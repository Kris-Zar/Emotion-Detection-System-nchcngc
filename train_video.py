"""
train_video.py
===============
Stage 1 — VIDEO modality only. Run on its own, whenever you like:

    python train_video.py

Independent of the other modality scripts. Saves its best checkpoint to
CFG.checkpoint_dir/video_best.pt. Once all four modality scripts
(audio/face/text/video) have each been run and produced a *_best.pt
checkpoint, move on to extract_embeddings.py.
"""

import pandas as pd
from torch.utils.data import DataLoader

from config import CFG
from datasets import VideoEmotionDataset, collate_video
from models import VideoEmotionModel
from train_utils import train_unimodal_model, build_loader_kwargs


def main():
    print(f"\n{'#'*70}")
    print("# STAGE 1 — VIDEO MODEL TRAINING")
    print(f"# Device: {CFG.device}")
    print(f"{'#'*70}\n")

    print("[Step 0] Loading split manifests...")
    emotion_train = pd.read_csv(f"{CFG.split_dir}/emotion_train.csv")
    emotion_val = pd.read_csv(f"{CFG.split_dir}/emotion_validation.csv")
    print(f"Train: {len(emotion_train)} | Val: {len(emotion_val)}")

    print("\n[Step 1] Creating video datasets...")
    loader_kwargs = build_loader_kwargs(CFG)

    video_train_ds = VideoEmotionDataset(emotion_train, CFG.artifact_dir, "train", augment=True)
    video_val_ds = VideoEmotionDataset(emotion_val, CFG.artifact_dir, "validation", augment=False)

    video_train_loader = DataLoader(video_train_ds, batch_size=CFG.video_microbatch,
                                     shuffle=True, collate_fn=collate_video, drop_last=True,
                                     **loader_kwargs)
    video_val_loader = DataLoader(video_val_ds, batch_size=CFG.video_microbatch,
                                   shuffle=False, collate_fn=collate_video,
                                   **loader_kwargs)

    print("\n[Step 2] Training video model...")
    video_model = VideoEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.timesformer_model_name)
    video_model.load_pretrained()

    results = train_unimodal_model(
        video_model, video_train_loader, video_val_loader,
        modality='video', config=CFG
    )

    print(f"\n{'='*70}")
    print(f"VIDEO STAGE COMPLETE — Best Val F1: {results['best_val_f1']:.4f}")
    print(f"Checkpoint: {results['best_checkpoint']}")
    print(f"{'='*70}")
    print("\nOnce audio/face/text/video are ALL done, run: python extract_embeddings.py")

    return results


if __name__ == "__main__":
    results = main()
