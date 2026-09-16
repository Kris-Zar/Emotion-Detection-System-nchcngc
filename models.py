"""
models.py
=========
Unimodal emotion models (Audio/Face/Text/Video) with frozen backbones +
trainable heads, and the QuadraFusionModel used for Stage 2 fusion.

NOTE: unlike the original notebook cell, this file does NOT instantiate a
global `quadra_model` at import time. A model should only be created where
it's actually used (inside main() in train.py) — instantiating it here
would re-create the model (and re-print its param counts) every single
time anything imports this module.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, AutoModel, DebertaV2Model, TimesformerModel


class AudioEmotionModel(nn.Module):
    """
    HuBERT-large with frozen backbone + trainable classification head.
    Uses MASKED POOLING (never ordinary mean-pool over padded regions).
    """
    def __init__(self, num_classes: int = 7, model_name: str = "facebook/hubert-large-ls960-ft"):
        super().__init__()
        self.backbone = None
        self.model_name = model_name
        self.num_classes = num_classes
        self.hidden_dim = 1024
        self.num_unfrozen_blocks = 0  # set by unfreeze_last_n_blocks()

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_pretrained(self):
        print(f"[Audio] Loading {self.model_name}...")
        self.backbone = HubertModel.from_pretrained(self.model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Learned softmax weights over every HuBERT layer's output (embeddings +
        # each transformer block), instead of only using the final layer.
        # The final layer of an ASR-finetuned HuBERT is optimized to discard
        # exactly the prosody/pitch/tone cues emotion recognition needs, so
        # letting the model learn which layers actually carry emotion signal
        # (a SUPERB-style weighted layer sum) is the fix for that.
        num_layers = self.backbone.config.num_hidden_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers + 1))
        print(f"[Audio] Layer-weighting over {num_layers + 1} hidden states (embeddings + {num_layers} blocks) initialized.")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Audio] Backbone frozen. Trainable params: {trainable:,}")

    def unfreeze_last_n_blocks(self, n: int):
        """
        Partially fine-tune the backbone: unfreeze only the last n HuBERT
        transformer blocks (leaving the feature extractor and earlier blocks
        frozen). This keeps memory in budget on a 6GB card while still letting
        the most task-relevant layers adapt, instead of being stuck with a
        fully-frozen pooled representation that may not separate emotions well.
        """
        if self.backbone is None:
            raise RuntimeError("Call load_pretrained() before unfreeze_last_n_blocks().")
        if n <= 0:
            self.num_unfrozen_blocks = 0
            return

        encoder_layers = self.backbone.encoder.layers
        n = min(n, len(encoder_layers))
        for layer in encoder_layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.num_unfrozen_blocks = n
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Audio] Unfroze last {n} transformer block(s). Trainable params now: {trainable:,}")

    def masked_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked mean pooling accounting for HuBERT's temporal downsampling.
        Args:
            hidden_states: [batch, seq_len_feat, hidden_dim]
            attention_mask: [batch, seq_len_raw]
        Returns:
            pooled: [batch, hidden_dim]
        """
        batch_size, seq_len_feat, hidden_dim = hidden_states.shape
        mask_len = attention_mask.shape[1]

        # Downsample mask using max pooling (any valid frame = valid)
        pooled_mask = F.adaptive_max_pool1d(
            attention_mask.unsqueeze(1).float(),
            output_size=seq_len_feat
        ).squeeze(1)

        mask_expanded = pooled_mask.unsqueeze(-1).expand_as(hidden_states)
        masked_sum = (hidden_states * mask_expanded).sum(dim=1)
        masked_count = mask_expanded.sum(dim=1).clamp(min=1e-10)
        return masked_sum / masked_count

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None,
                return_embedding: bool = False):
        if self.training and self.num_unfrozen_blocks > 0:
            # Partial fine-tuning path: keep everything except the last N
            # blocks in eval mode (so frozen-layer dropout/layerdrop stays
            # off), but let the unfrozen tail train normally so gradients
            # flow into it. requires_grad=False on every earlier param means
            # no graph is built (and no memory spent) for the frozen prefix,
            # so this stays within a 6GB budget without needing torch.no_grad()
            # to wrap the whole call.
            self.backbone.eval()
            for layer in self.backbone.encoder.layers[-self.num_unfrozen_blocks:]:
                layer.train()
            outputs = self.backbone(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            with torch.no_grad():
                outputs = self.backbone(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

        # Weighted sum over all layers (must happen outside torch.no_grad() so
        # self.layer_weights still receives gradients even when the backbone
        # itself is fully frozen).
        hidden_states_tuple = outputs.hidden_states
        if hidden_states_tuple is not None and len(hidden_states_tuple) == self.layer_weights.shape[0]:
            stacked = torch.stack(hidden_states_tuple, dim=0)  # [num_layers+1, B, T, H]
            weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            hidden_states = (stacked * weights).sum(dim=0)
        else:
            # Defensive fallback if a different transformers version doesn't
            # return the expected number of hidden states.
            hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            pooled = self.masked_pool(hidden_states, attention_mask)
        else:
            pooled = hidden_states.mean(dim=1)

        if return_embedding:
            return pooled

        logits = self.classifier(pooled)
        return logits


class FaceEmotionModel(nn.Module):
    """DINOv2-large with frozen backbone + trainable classification head. CLS token pooling."""
    def __init__(self, num_classes: int = 7, model_name: str = "facebook/dinov2-large"):
        super().__init__()
        self.backbone = None
        self.model_name = model_name
        self.num_classes = num_classes
        self.hidden_dim = 1024

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_pretrained(self):
        print(f"[Face] Loading {self.model_name}...")
        self.backbone = AutoModel.from_pretrained(self.model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Face] Backbone frozen. Trainable params: {trainable:,}")

    def forward(self, pixel_values: torch.Tensor, return_embedding: bool = False):
        with torch.no_grad():
            outputs = self.backbone(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state

        cls_token = hidden_states[:, 0, :]

        if return_embedding:
            return cls_token

        logits = self.classifier(cls_token)
        return logits


class TextEmotionModel(nn.Module):
    """DeBERTa-v3-large with frozen backbone + trainable classification head. Masked mean pooling."""
    def __init__(self, num_classes: int = 7, model_name: str = "microsoft/deberta-v3-large"):
        super().__init__()
        self.backbone = None
        self.model_name = model_name
        self.num_classes = num_classes
        self.hidden_dim = 1024

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_pretrained(self):
        print(f"[Text] Loading {self.model_name}...")
        self.backbone = DebertaV2Model.from_pretrained(self.model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Text] Backbone frozen. Trainable params: {trainable:,}")

    def masked_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        masked_sum = (hidden_states * mask_expanded).sum(dim=1)
        masked_count = mask_expanded.sum(dim=1).clamp(min=1e-10)
        return masked_sum / masked_count

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                return_embedding: bool = False):
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            pooled = self.masked_pool(hidden_states, attention_mask)
        else:
            pooled = hidden_states.mean(dim=1)

        if return_embedding:
            return pooled

        logits = self.classifier(pooled)
        return logits


class VideoEmotionModel(nn.Module):
    """TimeSformer-base with frozen backbone + trainable classification head. CLS token pooling."""
    def __init__(self, num_classes: int = 7, model_name: str = "facebook/timesformer-base-finetuned-k400"):
        super().__init__()
        self.backbone = None
        self.model_name = model_name
        self.num_classes = num_classes
        self.hidden_dim = 768

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_pretrained(self):
        print(f"[Video] Loading {self.model_name}...")
        self.backbone = TimesformerModel.from_pretrained(self.model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Video] Backbone frozen. Trainable params: {trainable:,}")

    def forward(self, pixel_values: torch.Tensor, return_embedding: bool = False):
        with torch.no_grad():
            outputs = self.backbone(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state

        cls_token = hidden_states[:, 0, :]

        if return_embedding:
            return cls_token

        logits = self.classifier(cls_token)
        return logits


class QuadraFusionModel(nn.Module):
    """
    Quadra Contextual Fusion Model for Multimodal Emotion Detection.

    Components:
    - Modality projections: Audio (1024), Face (1024), Text (1024), Video (768) -> hidden_dim (768)
    - Learned modality type embeddings (Context, Audio, Face, Text, Video)
    - Explicit availability embeddings + availability mask projection
    - Learned global context token
    - Transformer encoder: pre-LN, 2 layers, 8 heads with src_key_padding_mask
    - Reliability gate: per-modality reliability scoring with proper softmax masking
    - Fused classifier head: [context_out (hidden_dim) + modality_summary (hidden_dim)] -> hidden_dim * 2
    - Modality dropout (train-only, guarantees >= 1 available modality retained)
    - Safe all-missing abstention support
    """
    def __init__(self, num_classes: int = 7, hidden_dim: int = 768,
                 num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.1, modality_dropout: float = 0.2,
                 audio_dim: int = 1024, face_dim: int = 1024,
                 text_dim: int = 1024, video_dim: int = 768):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        self.modalities = ["audio", "face", "text", "video"]

        # Modality projections to common hidden dimension
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim)
        )
        self.face_proj = nn.Sequential(
            nn.LayerNorm(face_dim),
            nn.Linear(face_dim, hidden_dim)
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim)
        )
        self.video_proj = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim)
        )

        # Modality type embeddings (0=context, 1=audio, 2=face, 3=text, 4=video)
        self.modality_embed = nn.Embedding(5, hidden_dim)

        # Explicit available-vs-missing embedding per modality [4, 2, hidden_dim]
        self.availability_embeddings = nn.Parameter(
            torch.randn(4, 2, hidden_dim) * 0.02
        )

        # Learned context token [1, 1, hidden_dim]
        self.context_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Availability mask projection
        self.availability_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Transformer encoder with pre-LN for stable convergence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        # Reliability gate: per-modality reliability scoring
        self.reliability_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Classifier head: takes [context_out (hidden_dim) + modality_summary (hidden_dim)] = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self,
                audio_emb: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
                face_emb: Optional[torch.Tensor] = None,
                text_emb: Optional[torch.Tensor] = None,
                video_emb: Optional[torch.Tensor] = None,
                availability_mask: Optional[torch.Tensor] = None,
                reliability_scores: Optional[torch.Tensor] = None,
                return_details: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for Quadra Contextual Fusion.
        Supports both individual positional tensors and a single batch dictionary:
          model(audio_emb, face_emb, text_emb, video_emb, availability_mask)
          model({'audio': ..., 'face': ..., 'text': ..., 'video': ..., 'mask': ...})
        """
        # Handle batch dict input
        if isinstance(audio_emb, dict):
            batch = audio_emb
            a_emb = batch.get('audio')
            f_emb = batch.get('face')
            t_emb = batch.get('text')
            v_emb = batch.get('video')
            avail_mask = batch.get('mask', batch.get('availability_mask'))
            rel_scores = batch.get('reliability_scores', None)
        else:
            a_emb = audio_emb
            f_emb = face_emb
            t_emb = text_emb
            v_emb = video_emb
            avail_mask = availability_mask
            rel_scores = reliability_scores

        # Determine batch size and device
        for t in [a_emb, f_emb, t_emb, v_emb, avail_mask]:
            if t is not None and torch.is_tensor(t):
                batch_size = t.shape[0]
                device = t.device
                break
        else:
            raise ValueError("At least one tensor must be provided to QuadraFusionModel.")

        # Fallback empty tensors if not provided
        if a_emb is None:
            a_emb = torch.zeros(batch_size, 1024, device=device)
        if f_emb is None:
            f_emb = torch.zeros(batch_size, 1024, device=device)
        if t_emb is None:
            t_emb = torch.zeros(batch_size, 1024, device=device)
        if v_emb is None:
            v_emb = torch.zeros(batch_size, 768, device=device)
        if avail_mask is None:
            avail_mask = torch.ones(batch_size, 4, device=device)

        avail_mask = avail_mask.float().clone()
        effective_mask = avail_mask.clone()

        # Modality dropout (training only) — guaranteed to preserve >=1 available modality per sample
        if self.training and self.modality_dropout > 0:
            random_drop = torch.rand_like(effective_mask) < self.modality_dropout
            effective_mask = effective_mask * (~random_drop).float()
            for i in range(batch_size):
                if avail_mask[i].sum() > 0 and effective_mask[i].sum() == 0:
                    available_indices = torch.where(avail_mask[i] > 0)[0]
                    keep_idx = available_indices[torch.randint(0, len(available_indices), (1,), device=device)]
                    effective_mask[i, keep_idx] = 1.0

        all_missing = (effective_mask.sum(dim=1) == 0)

        # Project each modality to common hidden dimension
        projs = [
            self.audio_proj(a_emb),
            self.face_proj(f_emb),
            self.text_proj(t_emb),
            self.video_proj(v_emb)
        ]

        # Availability projection added to global context
        avail_context = self.availability_proj(effective_mask)  # [B, hidden_dim]

        tokens = []
        for idx in range(4):
            z = projs[idx]
            # Modality token (type embedding 1..4)
            mod_type_id = torch.full((batch_size,), idx + 1, device=device, dtype=torch.long)
            mod_embed = self.modality_embed(mod_type_id)
            # Availability embedding (0=missing, 1=available)
            avail_idx = effective_mask[:, idx].long().clamp(0, 1)
            avail_embed = self.availability_embeddings[idx][avail_idx]

            z = z + mod_embed + avail_embed
            # Zero out inactive modalities
            z = z * effective_mask[:, idx:idx+1]
            tokens.append(z.unsqueeze(1))

        modality_tokens = torch.cat(tokens, dim=1)  # [B, 4, hidden_dim]

        # Context token at position 0
        ctx = self.context_token.expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
        ctx = ctx + self.modality_embed(torch.zeros(batch_size, device=device, dtype=torch.long)).unsqueeze(1)
        ctx = ctx + avail_context.unsqueeze(1)

        # Full sequence: [Context, Audio, Face, Text, Video] -> [B, 5, hidden_dim]
        sequence = torch.cat([ctx, modality_tokens], dim=1)

        # Attention padding mask: True = masked (ignored by self-attention)
        # Context token (idx 0) is never masked; modality tokens are masked if unavailable
        src_key_padding_mask = torch.cat([
            torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
            effective_mask.eq(0)
        ], dim=1)

        # Transformer encoding with proper attention masking
        transformed = self.transformer(sequence, src_key_padding_mask=src_key_padding_mask)

        context_out = transformed[:, 0, :]        # [B, hidden_dim]
        mod_tokens_out = transformed[:, 1:, :]    # [B, 4, hidden_dim]

        # Reliability scoring over the 4 modality tokens
        raw_gate_scores = self.reliability_gate(mod_tokens_out).squeeze(-1)  # [B, 4]

        if rel_scores is not None:
            raw_gate_scores = raw_gate_scores + torch.log(rel_scores.clamp(min=1e-6))

        # Mask unavailable modalities with large negative value before softmax
        masked_scores = raw_gate_scores.masked_fill(effective_mask <= 0, -1e4)
        weights = torch.softmax(masked_scores, dim=1)  # [B, 4]

        if all_missing.any():
            weights[all_missing] = 0.0

        # Weighted modality summary
        modality_summary = (mod_tokens_out * weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]

        # Concatenate context token and modality summary -> [B, hidden_dim * 2]
        fused = torch.cat([context_out, modality_summary], dim=-1)
        logits = self.classifier(fused)

        # Safe abstention: if all modalities are missing, set logits to NaN
        if all_missing.any():
            logits = logits.clone()
            logits[all_missing] = float('nan')

        if return_details:
            return logits, weights, effective_mask, all_missing
        return logits
