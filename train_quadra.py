"""
train_quadra.py
================
Stage 2 — Quadra fusion model. Run this AFTER extract_embeddings.py has
produced the 8 embedding files in CFG.embeddings_dir.

    python train_quadra.py

This never touches the unimodal models — it only reads the embedding
files extract_embeddings.py wrote to disk, so it can be run in a
completely separate session (even on a different day) from Stage 1.
"""

import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from config import CFG
from datasets import QuadraDataset, collate_quadra
from models import QuadraFusionModel
from train_utils import build_loader_kwargs, train_quadra_model, load_embeddings


REQUIRED_EMBEDDINGS = [
    "audio_train.pt", "face_train.pt", "text_train.pt", "video_train.pt",
    "audio_val.pt", "face_val.pt", "text_val.pt", "video_val.pt",
]


def check_prerequisites():
    missing = [f for f in REQUIRED_EMBEDDINGS if not (Path(CFG.embeddings_dir) / f).exists()]
    if missing:
        print("Missing embedding files: " + ", ".join(missing))
        print("Run: python extract_embeddings.py first.")
        sys.exit(1)


def main():
    print(f"\n{'#'*70}")
    print("# STAGE 2 — QUADRA FUSION MODEL TRAINING")
    print(f"# Device: {CFG.device}")
    print(f"{'#'*70}\n")

    check_prerequisites()

    print("[Step 0] Loading split manifests...")
    emotion_train = pd.read_csv(f"{CFG.split_dir}/emotion_train.csv")
    emotion_val = pd.read_csv(f"{CFG.split_dir}/emotion_validation.csv")

    print("\n[Step 1] Loading pre-extracted embeddings...")
    audio_embs_train = load_embeddings(f"{CFG.embeddings_dir}/audio_train.pt")
    face_embs_train = load_embeddings(f"{CFG.embeddings_dir}/face_train.pt")
    text_embs_train = load_embeddings(f"{CFG.embeddings_dir}/text_train.pt")
    video_embs_train = load_embeddings(f"{CFG.embeddings_dir}/video_train.pt")

    audio_embs_val = load_embeddings(f"{CFG.embeddings_dir}/audio_val.pt")
    face_embs_val = load_embeddings(f"{CFG.embeddings_dir}/face_val.pt")
    text_embs_val = load_embeddings(f"{CFG.embeddings_dir}/text_val.pt")
    video_embs_val = load_embeddings(f"{CFG.embeddings_dir}/video_val.pt")

    print("\n[Step 2] Building Quadra datasets...")
    quadra_train_ds = QuadraDataset(emotion_train, audio_embs_train, face_embs_train,
                                     text_embs_train, video_embs_train, split="train")
    quadra_val_ds = QuadraDataset(emotion_val, audio_embs_val, face_embs_val,
                                   text_embs_val, video_embs_val, split="validation")

    loader_kwargs = build_loader_kwargs(CFG)
    quadra_train_loader = DataLoader(quadra_train_ds, batch_size=32, shuffle=True,
                                      collate_fn=collate_quadra, drop_last=True, **loader_kwargs)
    quadra_val_loader = DataLoader(quadra_val_ds, batch_size=32, shuffle=False,
                                    collate_fn=collate_quadra, **loader_kwargs)

    print("\n[Step 3] Training Quadra fusion model...")
    quadra_model = QuadraFusionModel(
        num_classes=CFG.num_emotions,
        hidden_dim=CFG.hidden_dim,
        num_layers=CFG.quadra_layers,
        num_heads=CFG.quadra_heads,
        dropout=CFG.quadra_dropout,
        modality_dropout=CFG.modality_dropout
    )

    results = train_quadra_model(quadra_model, quadra_train_loader, quadra_val_loader, config=CFG)

    print(f"\n{'='*70}")
    print(f"QUADRA STAGE COMPLETE — Best Val F1: {results['best_val_f1']:.4f}")
    print(f"Checkpoint: {results['best_checkpoint']}")
    print(f"{'='*70}")
    print("\nFull pipeline finished. Final fusion model checkpoint saved above.")

    return results


if __name__ == "__main__":
    results = main()
