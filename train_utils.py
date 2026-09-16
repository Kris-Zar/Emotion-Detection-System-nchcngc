"""
train_utils.py
===============
Metrics tracking, checkpointing, early stopping, the Stage 1 unimodal
training loop, frozen-embedding extraction, and the Stage 2 Quadra
training loop.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from config import TrainingConfig
from datasets import parse_label


def build_loader_kwargs(config: TrainingConfig) -> Dict[str, Any]:
    """
    Common DataLoader kwargs built from CFG, shared by every train_<modality>.py
    script and extract_embeddings.py / train_quadra.py. Was previously only
    duplicated inline inside train.py's _loader_kwargs() — the standalone
    per-modality scripts imported this name but it never existed here, which
    is why they could not even be imported.
    """
    kwargs = dict(
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
    )
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


def save_embeddings(embeddings: Dict[str, torch.Tensor], path: str) -> None:
    """Save a {sample_id: embedding_tensor} dict produced by extract_embeddings() to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)
    print(f"[Embeddings] Saved {len(embeddings)} embeddings -> {path}")


def load_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """Load a {sample_id: embedding_tensor} dict previously written by save_embeddings()."""
    embeddings = torch.load(path, map_location='cpu')
    print(f"[Embeddings] Loaded {len(embeddings)} embeddings <- {path}")
    return embeddings


def _build_label_map(config: TrainingConfig) -> Dict[str, int]:
    label_map = {}
    for idx, cls in enumerate(config.emotion_classes):
        label_map[cls] = idx
        label_map[cls.lower()] = idx
        label_map[cls.capitalize()] = idx
        label_map[cls.upper()] = idx
    return label_map


def compute_class_weights(df: pd.DataFrame, config: TrainingConfig) -> torch.Tensor:
    """
    Inverse-frequency class weights computed from the REAL training label
    distribution (uses the same parse_label() logic datasets.py uses, so it
    stays consistent whether the manifest has an 'emotion_id' column or a
    string 'emotion' column). Fixes the class-imbalance issue (a
    Neutral-dominant 7-class distribution silently starving the minority
    classes of gradient signal) that was capping audio accuracy at ~24%.
    """
    label_map = _build_label_map(config)
    counts = np.zeros(config.num_emotions, dtype=np.float64)
    for _, row in df.iterrows():
        label = parse_label(row, label_map)
        counts[label] += 1

    counts = np.clip(counts, 1.0, None)  # guard against div-by-zero for any unseen class
    weights = counts.sum() / (config.num_emotions * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_optimizer_param_groups(model: nn.Module, base_lr: float, weight_decay: float,
                                  backbone_lr: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Split a model's trainable parameters into two AdamW param groups:
      - params under 'backbone.' that have been selectively unfrozen (e.g. via
        AudioEmotionModel.unfreeze_last_n_blocks) get a lower learning rate,
        since they're pretrained weights that only need gentle nudging.
      - everything else (classification head, the audio model's learned
        layer-weighting parameter, etc.) trains at the normal unimodal LR.
    For a fully-frozen backbone (the original setup) the backbone group is
    empty and this is equivalent to a single param group at base_lr.
    """
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_params.append(p)
        else:
            head_params.append(p)

    groups = [{'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay}]
    if backbone_params:
        groups.append({
            'params': backbone_params,
            'lr': backbone_lr if backbone_lr is not None else base_lr * 0.1,
            'weight_decay': weight_decay,
        })
    return groups


class MetricsTracker:
    """Track and compute classification metrics."""

    def __init__(self, num_classes: int = 7, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float = 0.0):
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probs = F.softmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        self.all_preds.extend(preds)
        self.all_labels.extend(labels)
        self.all_probs.extend(probs)
        self.total_loss += loss
        self.num_batches += 1

    def compute(self) -> Dict[str, Any]:
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
        per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))

        return {
            'loss': self.total_loss / max(self.num_batches, 1),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_precision': {self.class_names[i]: per_class_precision[i] for i in range(self.num_classes)},
            'per_class_recall': {self.class_names[i]: per_class_recall[i] for i in range(self.num_classes)},
            'per_class_f1': {self.class_names[i]: per_class_f1[i] for i in range(self.num_classes)},
            'confusion_matrix': cm.tolist()
        }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                     metrics: Dict, path: str, is_best: bool = False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = str(Path(path).parent / "best_model.pt")
        torch.save(checkpoint, best_path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(model: nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"[Checkpoint] Loaded from {path} (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = 'max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = score > self.best_score if self.mode == 'max' else score < self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_unimodal_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    modality: str,
    config: TrainingConfig,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    accum_steps: Optional[int] = None,
    microbatch: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
    backbone_lr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Train a single unimodal model with a (possibly partially unfrozen) backbone.

    class_weights: optional per-class weight tensor (e.g. from compute_class_weights())
        passed straight into nn.CrossEntropyLoss to counter class imbalance.
    backbone_lr: optional lower learning rate for any backbone parameters the
        model has selectively unfrozen (e.g. AudioEmotionModel.unfreeze_last_n_blocks).
        Ignored (no effect) if the backbone is fully frozen.
    """
    epochs = epochs or config.unimodal_epochs
    lr = lr or config.unimodal_lr
    accum_steps = accum_steps or getattr(config, f"{modality}_accum_steps", 4)
    microbatch = microbatch or getattr(config, f"{modality}_microbatch", 8)

    device = config.device
    model = model.to(device)

    # Trainable params, split into a head/backbone param group so a partially
    # unfrozen backbone can train at a gentler LR than the classification head.
    param_groups = build_optimizer_param_groups(
        model, base_lr=lr, weight_decay=config.unimodal_weight_decay, backbone_lr=backbone_lr
    )
    optimizer = torch.optim.AdamW(param_groups)
    trainable_params = [p for group in param_groups for p in group['params']]

    # Scheduler
    total_steps = len(train_loader) * epochs // accum_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # AMP — enabled only on CUDA; silently falls back to float32 on CPU so
    # nothing breaks if device is 'cpu'. GradScaler handles the loss scaling
    # needed to keep float16 gradients from underflowing.
    use_amp = getattr(config, 'use_amp', False) and device != 'cpu'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = 0.0
    best_checkpoint_path = None
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode='max')

    print(f"\n{'='*60}")
    print(f"STAGE 1: Training {modality.upper()} Model")
    print(f"{'='*60}")
    print(f"Epochs: {epochs} | Accum: {accum_steps} | Microbatch: {microbatch}")
    print(f"AMP (mixed precision): {'ON' if use_amp else 'OFF'}")
    for i, g in enumerate(param_groups):
        print(f"  Param group {i}: {sum(p.numel() for p in g['params']):,} params @ lr={g['lr']}")
    if class_weights is not None:
        print(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        train_tracker = MetricsTracker(num_classes=config.num_emotions, class_names=config.emotion_classes)
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # autocast is a no-op when use_amp=False (dtype stays float32)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if modality == 'audio':
                    inputs = batch['input_values'].to(device)
                    masks = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(inputs, masks)
                elif modality == 'face':
                    inputs = batch['pixel_values'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(inputs)
                elif modality == 'text':
                    inputs = batch['input_ids'].to(device)
                    masks = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(inputs, masks)
                elif modality == 'video':
                    inputs = batch['pixel_values'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(inputs)

                loss = criterion(logits, labels) / accum_steps

            # scaler.scale() is a no-op when use_amp=False
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # unscale before clip so the gradient norm is in the original scale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_tracker.update(logits.detach(), labels, loss.item() * accum_steps)
            pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})

        train_metrics = train_tracker.compute()

        # ---- VALIDATION ----
        model.eval()
        val_tracker = MetricsTracker(num_classes=config.num_emotions, class_names=config.emotion_classes)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if modality == 'audio':
                        inputs = batch['input_values'].to(device)
                        masks = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)
                        logits = model(inputs, masks)
                    elif modality == 'face':
                        inputs = batch['pixel_values'].to(device)
                        labels = batch['label'].to(device)
                        logits = model(inputs)
                    elif modality == 'text':
                        inputs = batch['input_ids'].to(device)
                        masks = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)
                        logits = model(inputs, masks)
                    elif modality == 'video':
                        inputs = batch['pixel_values'].to(device)
                        labels = batch['label'].to(device)
                        logits = model(inputs)

                    loss = criterion(logits, labels)
                val_tracker.update(logits, labels, loss.item())

        val_metrics = val_tracker.compute()

        history['train_loss'].append(train_metrics['loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_metrics['loss']:.4f} | "
              f"Train F1: {train_metrics['f1']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        # Save best by validation macro-F1
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_checkpoint_path = f"{config.checkpoint_dir}/{modality}_best.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, best_checkpoint_path)
            print(f"  -> New best Val F1: {best_val_f1:.4f}")

        if early_stopping(val_metrics['f1']):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    print(f"\n{modality.upper()} Training Complete. Best Val F1: {best_val_f1:.4f}")

    return {
        'history': history,
        'best_checkpoint': best_checkpoint_path,
        'best_val_f1': best_val_f1
    }


def extract_embeddings(model: nn.Module, dataloader: DataLoader, modality: str,
                        device: str, desc: str = "Extracting") -> Dict[str, torch.Tensor]:
    """
    Extract frozen embeddings from a trained unimodal model.
    Returns a dictionary mapping sample_id -> embedding tensor.
    """
    model.eval()
    model = model.to(device)
    embeddings = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            if modality == 'audio':
                inputs = batch['input_values'].to(device)
                masks = batch['attention_mask'].to(device)
                embs = model(inputs, masks, return_embedding=True)
            elif modality == 'face':
                inputs = batch['pixel_values'].to(device)
                embs = model(inputs, return_embedding=True)
            elif modality == 'text':
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                embs = model(inputs, masks, return_embedding=True)
            elif modality == 'video':
                inputs = batch['pixel_values'].to(device)
                embs = model(inputs, return_embedding=True)

            embs = embs.cpu()
            for i, sid in enumerate(batch['sample_id']):
                embeddings[sid] = embs[i]

    return embeddings


def train_quadra_model(
    quadra_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Train the Quadra fusion model using frozen unimodal embeddings.
    """
    epochs = epochs or config.quadra_epochs
    lr = lr or config.quadra_lr

    device = config.device
    quadra_model = quadra_model.to(device)

    # Optimize all Quadra parameters (unimodal models are NOT here; embeddings are pre-computed)
    optimizer = torch.optim.AdamW(quadra_model.parameters(), lr=lr, weight_decay=config.quadra_weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = 0.0
    best_checkpoint_path = None
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode='max')

    print(f"\n{'='*60}")
    print(f"STAGE 2: Training QUADRA Fusion Model")
    print(f"{'='*60}")
    print(f"Epochs: {epochs} | LR: {lr} | Hidden: {config.hidden_dim}")
    print(f"Layers: {config.quadra_layers} | Heads: {config.quadra_heads}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(epochs):
        # ---- TRAIN ----
        quadra_model.train()
        train_tracker = MetricsTracker(num_classes=config.num_emotions, class_names=config.emotion_classes)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            audio_emb = batch['audio_emb'].to(device)
            face_emb = batch['face_emb'].to(device)
            text_emb = batch['text_emb'].to(device)
            video_emb = batch['video_emb'].to(device)
            labels = batch['label'].to(device)

            # All modalities available during training
            availability = torch.ones(audio_emb.size(0), 4, device=device)
            reliability = torch.ones(audio_emb.size(0), 4, device=device)

            logits = quadra_model(audio_emb, face_emb, text_emb, video_emb, availability, reliability)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(quadra_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_tracker.update(logits.detach(), labels, loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_metrics = train_tracker.compute()

        # ---- VALIDATION ----
        quadra_model.eval()
        val_tracker = MetricsTracker(num_classes=config.num_emotions, class_names=config.emotion_classes)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                audio_emb = batch['audio_emb'].to(device)
                face_emb = batch['face_emb'].to(device)
                text_emb = batch['text_emb'].to(device)
                video_emb = batch['video_emb'].to(device)
                labels = batch['label'].to(device)

                availability = torch.ones(audio_emb.size(0), 4, device=device)
                reliability = torch.ones(audio_emb.size(0), 4, device=device)

                logits = quadra_model(audio_emb, face_emb, text_emb, video_emb, availability, reliability)
                loss = criterion(logits, labels)
                val_tracker.update(logits, labels, loss.item())

        val_metrics = val_tracker.compute()

        history['train_loss'].append(train_metrics['loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_metrics['loss']:.4f} | "
              f"Train F1: {train_metrics['f1']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_checkpoint_path = f"{config.checkpoint_dir}/quadra_best.pt"
            save_checkpoint(quadra_model, optimizer, epoch, val_metrics, best_checkpoint_path)
            print(f"  -> New best Val F1: {best_val_f1:.4f}")

        if early_stopping(val_metrics['f1']):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    print(f"\nQUADRA Training Complete. Best Val F1: {best_val_f1:.4f}")

    return {
        'history': history,
        'best_checkpoint': best_checkpoint_path,
        'best_val_f1': best_val_f1
    }