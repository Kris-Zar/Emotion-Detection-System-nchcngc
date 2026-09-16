"""
train_face.py
==============
Stage 1 — FACE modality only. Run on its own, whenever you like:

    python train_face.py

Independent of train_audio.py — run this before or after it, same day or
a week later. Saves its best checkpoint to CFG.checkpoint_dir/face_best.pt.
"""

import pandas as pd
from torch.utils.data import DataLoader

from config import CFG
from datasets import FaceEmotionDataset, collate_face
from models import FaceEmotionModel
from train_utils import train_unimodal_model, build_loader_kwargs


def main():
    print(f"\n{'#'*70}")
    print("# STAGE 1 — FACE MODEL TRAINING")
    print(f"# Device: {CFG.device}")
    print(f"{'#'*70}\n")

    print("[Step 0] Loading split manifests...")
    emotion_train = pd.read_csv(f"{CFG.split_dir}/emotion_train.csv")
    emotion_val = pd.read_csv(f"{CFG.split_dir}/emotion_validation.csv")
    print(f"Train: {len(emotion_train)} | Val: {len(emotion_val)}")

    print("\n[Step 1] Creating face datasets...")
    loader_kwargs = build_loader_kwargs(CFG)

    face_train_ds = FaceEmotionDataset(emotion_train, CFG.artifact_dir, "train", augment=True)
    face_val_ds = FaceEmotionDataset(emotion_val, CFG.artifact_dir, "validation", augment=False)

    face_train_loader = DataLoader(face_train_ds, batch_size=CFG.face_microbatch,
                                    shuffle=True, collate_fn=collate_face, drop_last=True,
                                    **loader_kwargs)
    face_val_loader = DataLoader(face_val_ds, batch_size=CFG.face_microbatch,
                                  shuffle=False, collate_fn=collate_face,
                                  **loader_kwargs)

    print("\n[Step 2] Training face model...")
    face_model = FaceEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.dinov2_model_name)
    face_model.load_pretrained()

    results = train_unimodal_model(
        face_model, face_train_loader, face_val_loader,
        modality='face', config=CFG
    )

    print(f"\n{'='*70}")
    print(f"FACE STAGE COMPLETE — Best Val F1: {results['best_val_f1']:.4f}")
    print(f"Checkpoint: {results['best_checkpoint']}")
    print(f"{'='*70}")
    print("\nYou can stop here. Whenever you're ready, run: python train_text.py")

    return results


if __name__ == "__main__":
    results = main()
