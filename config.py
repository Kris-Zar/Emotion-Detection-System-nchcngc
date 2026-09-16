"""
config.py
=========
Central configuration for the Emotion Detection System training pipeline.

Every other file (datasets.py, models.py, train_utils.py, train.py) imports
CFG from here. Do not redefine TrainingConfig anywhere else — a single
shared instance is what keeps every module in sync.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class TrainingConfig:
    """Master configuration for the entire training pipeline."""

    # Project paths — ADJUST THESE TO MATCH YOUR SETUP
    project_root: str = r"C:\New folder\New Emodect"
    artifact_dir: str = r"C:\New folder\New Emodect\processed"           # Output from Preprocessing
    split_dir: str = r"C:\New folder\New Emodect\cleaned_metadata\splits"  # Output from Metadata & Splits
    metadata_dir: str = r"C:\New folder\New Emodect\cleaned_metadata"
    checkpoint_dir: str = r"C:\New folder\New Emodect\checkpoints_final"
    log_dir: str = r"C:\New folder\New Emodect\logs_final"
    embeddings_dir: str = r"C:\New folder\New Emodect\embeddings_final"
    reports_dir: str = r"C:\New folder\New Emodect\reports_final"
    outputs_dir: str = r"C:\New folder\New Emodect\outputs"

    # Emotion classes (exactly 7)
    emotion_classes: List[str] = field(default_factory=lambda: [
        "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"
    ])
    num_emotions: int = 7

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # RTX 3050 6GB Optimized Microbatch Settings
    audio_microbatch: int = 8
    face_microbatch: int = 8
    text_microbatch: int = 8
    video_microbatch: int = 4
    effective_batch_size: int = 32

    # Gradient accumulation steps
    audio_accum_steps: int = 4   # 8 * 4 = 32
    face_accum_steps: int = 4    # 8 * 4 = 32
    text_accum_steps: int = 4    # 8 * 4 = 32
    video_accum_steps: int = 8   # 4 * 8 = 32

    # DataLoader — safe to use workers now that Dataset/collate code
    # lives in a real, importable module instead of a notebook cell.
    num_workers: int = 2
    pin_memory: bool = True if torch.cuda.is_available() else False
    persistent_workers: bool = True
    prefetch_factor: int = 4  # only used when num_workers > 0

    # Training epochs
    unimodal_epochs: int = 15
    quadra_epochs: int = 20
    early_stopping_patience: int = 5

    # Optimizer settings
    unimodal_lr: float = 1e-4
    unimodal_weight_decay: float = 1e-5
    quadra_lr: float = 5e-5
    quadra_weight_decay: float = 1e-5

    # Scheduler
    warmup_ratio: float = 0.1

    # Model architecture
    hidden_dim: int = 768
    quadra_layers: int = 2
    quadra_heads: int = 8
    quadra_dropout: float = 0.1
    modality_dropout: float = 0.20

    # Video preprocessing params (must match Notebook 5)
    video_frames: int = 16
    video_size: Tuple[int, int] = (224, 224)

    # Audio preprocessing params
    audio_sample_rate: int = 16000
    audio_max_length: int = 160000  # 10 seconds at 16kHz

    # Face preprocessing params
    face_size: Tuple[int, int] = (224, 224)

    # Text preprocessing params
    text_max_length: int = 256
    text_model_name: str = "microsoft/deberta-v3-large"

    # Backbone model names
    hubert_model_name: str = "facebook/hubert-large-ls960-ft"
    dinov2_model_name: str = "facebook/dinov2-large"
    timesformer_model_name: str = "facebook/timesformer-base-finetuned-k400"

    # Augmentation (TRAIN-ONLY)
    # audio_snr_min was 5.0 — that's very heavy noise (comparable in power to
    # the signal itself) and was washing out the subtle prosodic/pitch cues
    # emotion recognition depends on. Raised the floor so worst-case
    # augmentation is still audibly clean.
    audio_snr_min: float = 15.0
    audio_snr_max: float = 30.0
    audio_gain_min: float = 0.75
    audio_gain_max: float = 1.25
    face_brightness_min: float = 0.15
    face_brightness_max: float = 0.55

    # Audio fine-tuning fixes (see AudioEmotionModel in models.py)
    audio_unfreeze_blocks: int = 0      # partially fine-tune the last N HuBERT transformer blocks
    audio_backbone_lr: float = 2e-6     # lower LR for those unfrozen blocks than the head's unimodal_lr
    use_class_weights: bool = True       # class-weighted CrossEntropyLoss from the real train label distribution

    # Mixed precision (AMP) — halves epoch time on RTX 30-series Tensor Cores
    # with no accuracy impact. Automatically disabled on CPU.
    use_amp: bool = True

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        for d in [self.checkpoint_dir, self.log_dir, self.embeddings_dir, self.reports_dir, self.outputs_dir]:
            os.makedirs(d, exist_ok=True)

        # Enable TF32 on Ampere+ (RTX 30-series and above)
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("[Config] TF32 enabled for Ampere architecture")

        torch.set_float32_matmul_precision('high')
        print(f"[Config] Device: {self.device}")
        print(f"[Config] Project Root: {self.project_root}")
        print(f"[Config] Artifacts Dir: {self.artifact_dir}")
        print(f"[Config] Splits Dir: {self.split_dir}")
        print(f"[Config] Checkpoints Dir: {self.checkpoint_dir}")
        print(f"[Config] Logs Dir: {self.log_dir}")
        print(f"[Config] Effective batch size: {self.effective_batch_size}")


# Single shared config instance — everything else imports CFG from here.
CFG = TrainingConfig()


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(CFG.seed)