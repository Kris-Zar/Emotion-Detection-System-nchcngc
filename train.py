"""
train.py
========
Entry point for the full training pipeline.

RUN THIS FROM A TERMINAL, NOT FROM INSIDE A JUPYTER NOTEBOOK:

    python train.py

Running it as a real script (with the `if __name__ == "__main__":` guard
below) is what lets Windows correctly spawn the DataLoader worker
processes defined by CFG.num_workers. Running the equivalent code inside
a notebook cell is what caused the original hang at "0/1436 [00:00<?, ?it/s]".
"""

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import CFG
from datasets import (
    AudioEmotionDataset, FaceEmotionDataset, TextEmotionDataset, VideoEmotionDataset,
    collate_audio, collate_face, collate_text, collate_video,
    QuadraDataset, collate_quadra
)
from models import (
    AudioEmotionModel, FaceEmotionModel, TextEmotionModel, VideoEmotionModel, QuadraFusionModel
)
from train_utils import (
    train_unimodal_model, extract_embeddings, load_checkpoint, train_quadra_model
)


def _loader_kwargs():
    """Common DataLoader kwargs built from CFG, reused for every loader below."""
    kwargs = dict(
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        persistent_workers=CFG.persistent_workers,
    )
    if CFG.num_workers > 0:
        kwargs["prefetch_factor"] = CFG.prefetch_factor
    return kwargs


def main():
    """
    Execute the full training pipeline:
    1. Load manifests
    2. Train 4 unimodal models (Stage 1)
    3. Extract frozen embeddings
    4. Train Quadra fusion (Stage 2)
    5. Save all checkpoints
    """
    print(f"\n{'#'*70}")
    print(f"# EMOTION DETECTION SYSTEM V2 — NOTEBOOK 6: TRAINING")
    print(f"# Device: {CFG.device}")
    print(f"# Checkpoint Dir: {CFG.checkpoint_dir}")
    print(f"{'#'*70}\n")

    # ============================================================
    # STEP 0: Load Manifests
    # ============================================================
    print("[Step 0] Loading split manifests...")

    emotion_train = pd.read_csv(f"{CFG.split_dir}/emotion_train.csv")
    emotion_val = pd.read_csv(f"{CFG.split_dir}/emotion_validation.csv")
    emotion_test = pd.read_csv(f"{CFG.split_dir}/emotion_test.csv")

    print(f"Train: {len(emotion_train)} | Val: {len(emotion_val)} | Test: {len(emotion_test)}")

    # ============================================================
    # STEP 1: Initialize Tokenizer (for text fallback)
    # ============================================================
    print("\n[Step 1] Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)

    # ============================================================
    # STEP 2: Create Datasets & Loaders for Stage 1
    # ============================================================
    print("\n[Step 2] Creating Stage 1 datasets...")

    loader_kwargs = _loader_kwargs()

    # Audio
    audio_train_ds = AudioEmotionDataset(emotion_train, CFG.artifact_dir, "train", augment=True)
    audio_val_ds = AudioEmotionDataset(emotion_val, CFG.artifact_dir, "validation", augment=False)
    audio_train_loader = DataLoader(audio_train_ds, batch_size=CFG.audio_microbatch,
                                     shuffle=True, collate_fn=collate_audio, drop_last=True,
                                     **loader_kwargs)
    audio_val_loader = DataLoader(audio_val_ds, batch_size=CFG.audio_microbatch,
                                   shuffle=False, collate_fn=collate_audio,
                                   **loader_kwargs)

    # Face
    face_train_ds = FaceEmotionDataset(emotion_train, CFG.artifact_dir, "train", augment=True)
    face_val_ds = FaceEmotionDataset(emotion_val, CFG.artifact_dir, "validation", augment=False)
    face_train_loader = DataLoader(face_train_ds, batch_size=CFG.face_microbatch,
                                    shuffle=True, collate_fn=collate_face, drop_last=True,
                                    **loader_kwargs)
    face_val_loader = DataLoader(face_val_ds, batch_size=CFG.face_microbatch,
                                  shuffle=False, collate_fn=collate_face,
                                  **loader_kwargs)

    # Text
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

    # Video
    video_train_ds = VideoEmotionDataset(emotion_train, CFG.artifact_dir, "train", augment=True)
    video_val_ds = VideoEmotionDataset(emotion_val, CFG.artifact_dir, "validation", augment=False)
    video_train_loader = DataLoader(video_train_ds, batch_size=CFG.video_microbatch,
                                     shuffle=True, collate_fn=collate_video, drop_last=True,
                                     **loader_kwargs)
    video_val_loader = DataLoader(video_val_ds, batch_size=CFG.video_microbatch,
                                   shuffle=False, collate_fn=collate_video,
                                   **loader_kwargs)

    # ============================================================
    # STEP 3: Train Unimodal Models (Stage 1)
    # ============================================================
    print("\n[Step 3] Stage 1: Training Unimodal Models...")

    # Audio
    audio_model = AudioEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.hubert_model_name)
    audio_model.load_pretrained()
    audio_results = train_unimodal_model(
        audio_model, audio_train_loader, audio_val_loader,
        modality='audio', config=CFG
    )

    # Face
    face_model = FaceEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.dinov2_model_name)
    face_model.load_pretrained()
    face_results = train_unimodal_model(
        face_model, face_train_loader, face_val_loader,
        modality='face', config=CFG
    )

    # Text
    text_model = TextEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.text_model_name)
    text_model.load_pretrained()
    text_results = train_unimodal_model(
        text_model, text_train_loader, text_val_loader,
        modality='text', config=CFG
    )

    # Video
    video_model = VideoEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.timesformer_model_name)
    video_model.load_pretrained()
    video_results = train_unimodal_model(
        video_model, video_train_loader, video_val_loader,
        modality='video', config=CFG
    )

    # ============================================================
    # STEP 4: Extract Frozen Embeddings for Quadra
    # ============================================================
    print("\n[Step 4] Extracting frozen embeddings for Quadra...")

    # Reload best checkpoints
    load_checkpoint(audio_model, audio_results['best_checkpoint'])
    load_checkpoint(face_model, face_results['best_checkpoint'])
    load_checkpoint(text_model, text_results['best_checkpoint'])
    load_checkpoint(video_model, video_results['best_checkpoint'])

    # Extract from train
    print("Extracting train embeddings...")
    audio_embs_train = extract_embeddings(audio_model, audio_train_loader, 'audio', CFG.device, "Audio Train")
    face_embs_train = extract_embeddings(face_model, face_train_loader, 'face', CFG.device, "Face Train")
    text_embs_train = extract_embeddings(text_model, text_train_loader, 'text', CFG.device, "Text Train")
    video_embs_train = extract_embeddings(video_model, video_train_loader, 'video', CFG.device, "Video Train")

    # Extract from val
    print("Extracting validation embeddings...")
    audio_embs_val = extract_embeddings(audio_model, audio_val_loader, 'audio', CFG.device, "Audio Val")
    face_embs_val = extract_embeddings(face_model, face_val_loader, 'face', CFG.device, "Face Val")
    text_embs_val = extract_embeddings(text_model, text_val_loader, 'text', CFG.device, "Text Val")
    video_embs_val = extract_embeddings(video_model, video_val_loader, 'video', CFG.device, "Video Val")

    # ============================================================
    # STEP 5: Create Quadra Datasets
    # ============================================================
    print("\n[Step 5] Creating Quadra datasets...")

    quadra_train_ds = QuadraDataset(emotion_train, audio_embs_train, face_embs_train,
                                     text_embs_train, video_embs_train, split="train")
    quadra_val_ds = QuadraDataset(emotion_val, audio_embs_val, face_embs_val,
                                   text_embs_val, video_embs_val, split="validation")

    quadra_train_loader = DataLoader(quadra_train_ds, batch_size=32, shuffle=True,
                                      collate_fn=collate_quadra, drop_last=True,
                                      **loader_kwargs)
    quadra_val_loader = DataLoader(quadra_val_ds, batch_size=32, shuffle=False,
                                    collate_fn=collate_quadra,
                                    **loader_kwargs)

    # ============================================================
    # STEP 6: Train Quadra (Stage 2)
    # ============================================================
    print("\n[Step 6] Stage 2: Training Quadra Fusion Model...")

    quadra_model = QuadraFusionModel(
        num_classes=CFG.num_emotions,
        hidden_dim=CFG.hidden_dim,
        num_layers=CFG.quadra_layers,
        num_heads=CFG.quadra_heads,
        dropout=CFG.quadra_dropout,
        modality_dropout=CFG.modality_dropout
    )

    quadra_results = train_quadra_model(
        quadra_model, quadra_train_loader, quadra_val_loader, config=CFG
    )

    # ============================================================
    # STEP 7: Final Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Audio Best Val F1:  {audio_results['best_val_f1']:.4f}")
    print(f"Face Best Val F1:   {face_results['best_val_f1']:.4f}")
    print(f"Text Best Val F1:   {text_results['best_val_f1']:.4f}")
    print(f"Video Best Val F1:  {video_results['best_val_f1']:.4f}")
    print(f"Quadra Best Val F1: {quadra_results['best_val_f1']:.4f}")
    print(f"\nCheckpoints saved to: {CFG.checkpoint_dir}")
    print(f"{'='*70}\n")

    return {
        'audio': audio_results,
        'face': face_results,
        'text': text_results,
        'video': video_results,
        'quadra': quadra_results
    }


# ============================================================
# RUN THE PIPELINE
# ============================================================
if __name__ == "__main__":
    results = main()
