"""
train_audio.py
===============
Stage 1 — AUDIO modality only. Run on its own, whenever you like:

    python train_audio.py

Saves its best checkpoint to CFG.checkpoint_dir/audio_best.pt. Once this
finishes, you can close the terminal / shut down the machine — nothing
needs to stay in memory. Come back later (in any order, any gap in
between) and run train_face.py, train_text.py, train_video.py. Each
script only reads what the previous one wrote to disk.
"""

import pandas as pd
from torch.utils.data import DataLoader

from config import CFG
from datasets import AudioEmotionDataset, collate_audio
from models import AudioEmotionModel
from train_utils import train_unimodal_model, build_loader_kwargs, compute_class_weights


def main():
    print(f"\n{'#'*70}")
    print("# STAGE 1 — AUDIO MODEL TRAINING")
    print(f"# Device: {CFG.device}")
    print(f"{'#'*70}\n")

    print("[Step 0] Loading split manifests...")
    emotion_train = pd.read_csv(f"{CFG.split_dir}/emotion_train.csv")
    emotion_val = pd.read_csv(f"{CFG.split_dir}/emotion_validation.csv")
    print(f"Train: {len(emotion_train)} | Val: {len(emotion_val)}")

    print("\n[Step 0b] Computing class weights from training label distribution...")
    class_weights = compute_class_weights(emotion_train, CFG) if CFG.use_class_weights else None
    if class_weights is not None:
        for cls, w in zip(CFG.emotion_classes, class_weights.tolist()):
            print(f"    {cls:12s} weight={w:.3f}")

    print("\n[Step 1] Creating audio datasets...")
    loader_kwargs = build_loader_kwargs(CFG)

    audio_train_ds = AudioEmotionDataset(emotion_train, CFG.artifact_dir, "train", augment=True)
    audio_val_ds = AudioEmotionDataset(emotion_val, CFG.artifact_dir, "validation", augment=False)

    audio_train_loader = DataLoader(audio_train_ds, batch_size=CFG.audio_microbatch,
                                     shuffle=True, collate_fn=collate_audio, drop_last=True,
                                     **loader_kwargs)
    audio_val_loader = DataLoader(audio_val_ds, batch_size=CFG.audio_microbatch,
                                   shuffle=False, collate_fn=collate_audio,
                                   **loader_kwargs)

    print("\n[Step 2] Training audio model...")
    audio_model = AudioEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.hubert_model_name)
    audio_model.load_pretrained()
    audio_model.unfreeze_last_n_blocks(CFG.audio_unfreeze_blocks)

    results = train_unimodal_model(
        audio_model, audio_train_loader, audio_val_loader,
        modality='audio', config=CFG,
        class_weights=class_weights,
        backbone_lr=CFG.audio_backbone_lr,
    )

    print(f"\n{'='*70}")
    print(f"AUDIO STAGE COMPLETE — Best Val F1: {results['best_val_f1']:.4f}")
    print(f"Checkpoint: {results['best_checkpoint']}")
    print(f"{'='*70}")
    print("\nYou can stop here. Whenever you're ready, run: python train_face.py")

    return results


if __name__ == "__main__":
    results = main()
