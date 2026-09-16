"""
train_text.py
==============
Stage 1 — TEXT modality only. Run on its own, whenever you like:

    python train_text.py

Independent of the other modality scripts. Saves its best checkpoint to
CFG.checkpoint_dir/text_best.pt.
"""

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import CFG
from datasets import TextEmotionDataset, collate_text
from models import TextEmotionModel
from train_utils import train_unimodal_model, build_loader_kwargs


def main():
    print(f"\n{'#'*70}")
    print("# STAGE 1 — TEXT MODEL TRAINING")
    print(f"# Device: {CFG.device}")
    print(f"{'#'*70}\n")

    print("[Step 0] Loading split manifests...")
    emotion_train = pd.read_csv(f"{CFG.split_dir}/emotion_train.csv")
    emotion_val = pd.read_csv(f"{CFG.split_dir}/emotion_validation.csv")
    print(f"Train: {len(emotion_train)} | Val: {len(emotion_val)}")

    print("\n[Step 1] Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)

    print("\n[Step 2] Creating text datasets...")
    loader_kwargs = build_loader_kwargs(CFG)

    text_train_ds = TextEmotionDataset(emotion_train, CFG.artifact_dir, "train",
                                        tokenizer=tokenizer, augment=False)
    text_val_ds = TextEmotionDataset(emotion_val, CFG.artifact_dir, "validation",
                                      tokenizer=tokenizer, augment=False)

    text_train_loader = DataLoader(text_train_ds, batch_size=CFG.text_microbatch,
                                    shuffle=True, collate_fn=collate_text, drop_last=True,
                                    **loader_kwargs)
    text_val_loader = DataLoader(text_val_ds, batch_size=CFG.text_microbatch,
                                  shuffle=False, collate_fn=collate_text,
                                  **loader_kwargs)

    print("\n[Step 3] Training text model...")
    text_model = TextEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.text_model_name)
    text_model.load_pretrained()

    results = train_unimodal_model(
        text_model, text_train_loader, text_val_loader,
        modality='text', config=CFG
    )

    print(f"\n{'='*70}")
    print(f"TEXT STAGE COMPLETE — Best Val F1: {results['best_val_f1']:.4f}")
    print(f"Checkpoint: {results['best_checkpoint']}")
    print(f"{'='*70}")
    print("\nYou can stop here. Whenever you're ready, run: python train_video.py")

    return results


if __name__ == "__main__":
    results = main()
