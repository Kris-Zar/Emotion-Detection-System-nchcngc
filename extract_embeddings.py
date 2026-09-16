"""
extract_embeddings.py
======================
Run this AFTER all four modality scripts (train_audio.py, train_face.py,
train_text.py, train_video.py) have each produced a *_best.pt checkpoint
in CFG.checkpoint_dir — in any order, across any number of separate
sessions.

    python extract_embeddings.py

Loads each trained model's best checkpoint, runs one forward pass over
train+val data (backbone still frozen, no training happening here), and
saves the resulting embeddings to CFG.embeddings_dir as .pt files.
train_quadra.py reads these files back later — it never touches the
unimodal models directly, so you can shut everything down after this
script finishes.
"""

import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import CFG
from datasets import (
    AudioEmotionDataset, FaceEmotionDataset, TextEmotionDataset, VideoEmotionDataset,
    collate_audio, collate_face, collate_text, collate_video
)
from models import AudioEmotionModel, FaceEmotionModel, TextEmotionModel, VideoEmotionModel
from train_utils import build_loader_kwargs, load_checkpoint, extract_embeddings, save_embeddings


REQUIRED_CHECKPOINTS = {
    "audio": f"{CFG.checkpoint_dir}/audio_best.pt",
    "face": f"{CFG.checkpoint_dir}/face_best.pt",
    "text": f"{CFG.checkpoint_dir}/text_best.pt",
    "video": f"{CFG.checkpoint_dir}/video_best.pt",
}


def check_prerequisites():
    missing = [m for m, p in REQUIRED_CHECKPOINTS.items() if not Path(p).exists()]
    if missing:
        print("Missing checkpoints for: " + ", ".join(missing))
        print("Run the corresponding script(s) first, e.g.:")
        for m in missing:
            print(f"    python train_{m}.py")
        sys.exit(1)


def main():
    print(f"\n{'#'*70}")
    print("# EXTRACTING FROZEN EMBEDDINGS FOR QUADRA FUSION")
    print(f"{'#'*70}\n")

    check_prerequisites()

    print("[Step 0] Loading split manifests...")
    emotion_train = pd.read_csv(f"{CFG.split_dir}/emotion_train.csv")
    emotion_val = pd.read_csv(f"{CFG.split_dir}/emotion_validation.csv")

    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
    loader_kwargs = build_loader_kwargs(CFG)

    # ---- Eval-mode loaders (no augmentation, no shuffle) for all 4 modalities ----
    print("\n[Step 1] Building loaders...")

    audio_train_ds = AudioEmotionDataset(emotion_train, CFG.artifact_dir, "train", augment=False)
    audio_val_ds = AudioEmotionDataset(emotion_val, CFG.artifact_dir, "validation", augment=False)
    audio_train_loader = DataLoader(audio_train_ds, batch_size=CFG.audio_microbatch, shuffle=False,
                                     collate_fn=collate_audio, **loader_kwargs)
    audio_val_loader = DataLoader(audio_val_ds, batch_size=CFG.audio_microbatch, shuffle=False,
                                   collate_fn=collate_audio, **loader_kwargs)

    face_train_ds = FaceEmotionDataset(emotion_train, CFG.artifact_dir, "train", augment=False)
    face_val_ds = FaceEmotionDataset(emotion_val, CFG.artifact_dir, "validation", augment=False)
    face_train_loader = DataLoader(face_train_ds, batch_size=CFG.face_microbatch, shuffle=False,
                                    collate_fn=collate_face, **loader_kwargs)
    face_val_loader = DataLoader(face_val_ds, batch_size=CFG.face_microbatch, shuffle=False,
                                  collate_fn=collate_face, **loader_kwargs)

    text_train_ds = TextEmotionDataset(emotion_train, CFG.artifact_dir, "train", tokenizer=tokenizer, augment=False)
    text_val_ds = TextEmotionDataset(emotion_val, CFG.artifact_dir, "validation", tokenizer=tokenizer, augment=False)
    text_train_loader = DataLoader(text_train_ds, batch_size=CFG.text_microbatch, shuffle=False,
                                    collate_fn=collate_text, **loader_kwargs)
    text_val_loader = DataLoader(text_val_ds, batch_size=CFG.text_microbatch, shuffle=False,
                                  collate_fn=collate_text, **loader_kwargs)

    video_train_ds = VideoEmotionDataset(emotion_train, CFG.artifact_dir, "train", augment=False)
    video_val_ds = VideoEmotionDataset(emotion_val, CFG.artifact_dir, "validation", augment=False)
    video_train_loader = DataLoader(video_train_ds, batch_size=CFG.video_microbatch, shuffle=False,
                                     collate_fn=collate_video, **loader_kwargs)
    video_val_loader = DataLoader(video_val_ds, batch_size=CFG.video_microbatch, shuffle=False,
                                   collate_fn=collate_video, **loader_kwargs)

    # ---- Load trained heads ----
    print("\n[Step 2] Loading trained checkpoints...")
    audio_model = AudioEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.hubert_model_name)
    audio_model.load_pretrained()
    load_checkpoint(audio_model, REQUIRED_CHECKPOINTS["audio"])

    face_model = FaceEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.dinov2_model_name)
    face_model.load_pretrained()
    load_checkpoint(face_model, REQUIRED_CHECKPOINTS["face"])

    text_model = TextEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.text_model_name)
    text_model.load_pretrained()
    load_checkpoint(text_model, REQUIRED_CHECKPOINTS["text"])

    video_model = VideoEmotionModel(num_classes=CFG.num_emotions, model_name=CFG.timesformer_model_name)
    video_model.load_pretrained()
    load_checkpoint(video_model, REQUIRED_CHECKPOINTS["video"])

    # ---- Extract ----
    print("\n[Step 3] Extracting embeddings...")

    audio_embs_train = extract_embeddings(audio_model, audio_train_loader, 'audio', CFG.device, "Audio Train")
    face_embs_train = extract_embeddings(face_model, face_train_loader, 'face', CFG.device, "Face Train")
    text_embs_train = extract_embeddings(text_model, text_train_loader, 'text', CFG.device, "Text Train")
    video_embs_train = extract_embeddings(video_model, video_train_loader, 'video', CFG.device, "Video Train")

    audio_embs_val = extract_embeddings(audio_model, audio_val_loader, 'audio', CFG.device, "Audio Val")
    face_embs_val = extract_embeddings(face_model, face_val_loader, 'face', CFG.device, "Face Val")
    text_embs_val = extract_embeddings(text_model, text_val_loader, 'text', CFG.device, "Text Val")
    video_embs_val = extract_embeddings(video_model, video_val_loader, 'video', CFG.device, "Video Val")

    # ---- Save to disk ----
    print("\n[Step 4] Saving embeddings to disk...")
    save_embeddings(audio_embs_train, f"{CFG.embeddings_dir}/audio_train.pt")
    save_embeddings(face_embs_train, f"{CFG.embeddings_dir}/face_train.pt")
    save_embeddings(text_embs_train, f"{CFG.embeddings_dir}/text_train.pt")
    save_embeddings(video_embs_train, f"{CFG.embeddings_dir}/video_train.pt")

    save_embeddings(audio_embs_val, f"{CFG.embeddings_dir}/audio_val.pt")
    save_embeddings(face_embs_val, f"{CFG.embeddings_dir}/face_val.pt")
    save_embeddings(text_embs_val, f"{CFG.embeddings_dir}/text_val.pt")
    save_embeddings(video_embs_val, f"{CFG.embeddings_dir}/video_val.pt")

    print(f"\n{'='*70}")
    print("EMBEDDING EXTRACTION COMPLETE")
    print(f"Saved to: {CFG.embeddings_dir}")
    print(f"{'='*70}")
    print("\nYou can stop here. Whenever you're ready, run: python train_quadra.py")


if __name__ == "__main__":
    main()
