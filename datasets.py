"""
datasets.py
===========
Dataset classes and collate functions for all four modalities, plus the
Quadra fusion dataset.

THIS IS THE FILE THAT FIXES THE WINDOWS DATALOADER HANG.
Because these classes now live in a real, importable .py module (instead
of a Jupyter notebook cell), Windows can correctly spawn DataLoader worker
processes (num_workers > 0) — each worker just does
`from datasets import AudioEmotionDataset` etc., which works fine for a
real module but silently deadlocks when the classes only exist inside a
notebook's __main__ namespace.
"""

import random
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import torchvision.transforms as VT
from PIL import Image
from torch.utils.data import Dataset

from config import CFG


def resolve_artifact_dir(artifact_dir: str, split: str, modality: str) -> Path:
    """Resolve artifact directory supporting both split/modality and modality/split."""
    p1 = Path(artifact_dir) / split / modality
    if p1.exists():
        return p1
    p2 = Path(artifact_dir) / modality / split
    if p2.exists():
        return p2
    return p1


def parse_label(row: Any, label_map: Dict[str, int]) -> int:
    """Safely extract integer emotion label from dataframe row."""
    if 'emotion_id' in row and pd.notna(row['emotion_id']):
        return int(row['emotion_id'])
    emotion_val = str(row['emotion']).strip().lower()
    return label_map.get(emotion_val, label_map.get(str(row['emotion']), 0))


class AudioEmotionDataset(Dataset):
    """
    Load preprocessed audio artifacts from Notebook 5 / Preprocessing.
    Artifact: .pt tensor [seq_len] (1D waveform) or dict with 'input_values'.
    """
    def __init__(self, manifest_df: pd.DataFrame, artifact_dir: str, split: str, augment: bool = False):
        self.manifest = manifest_df.reset_index(drop=True)
        self.artifact_dir = resolve_artifact_dir(artifact_dir, split, "audio")
        self.augment = augment and (split == "train")

        self.label_map = {}
        for idx, cls in enumerate(CFG.emotion_classes):
            self.label_map[cls] = idx
            self.label_map[cls.lower()] = idx
            self.label_map[cls.capitalize()] = idx
            self.label_map[cls.upper()] = idx

        if 'modality' in self.manifest.columns:
            self.manifest = self.manifest[self.manifest['modality'] == 'audio'].reset_index(drop=True)

        valid_rows = []
        for _, row in self.manifest.iterrows():
            artifact_path = self.artifact_dir / f"{row['sample_id']}.pt"
            if artifact_path.exists():
                valid_rows.append(row)
        self.manifest = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[Audio {split}] Valid samples: {len(self.manifest)}")

    def __len__(self):
        return len(self.manifest)

    def _add_noise(self, waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Gaussian noise at specified SNR."""
        signal_power = waveform.pow(2).mean()
        noise = torch.randn_like(waveform)
        noise_power = noise.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = (signal_power / (snr_linear * noise_power + 1e-10)).sqrt()
        return waveform + noise_scale * noise

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        artifact_path = self.artifact_dir / f"{row['sample_id']}.pt"

        data = torch.load(artifact_path, map_location='cpu')
        if isinstance(data, dict):
            input_values = data.get('input_values', data.get('waveform', data.get('audio'))).float()
            attention_mask = data.get('attention_mask', torch.ones(input_values.shape[0], dtype=torch.long))
        elif torch.is_tensor(data):
            input_values = data.squeeze().float()
            attention_mask = torch.ones(input_values.shape[0], dtype=torch.long)
        else:
            input_values = torch.tensor(data, dtype=torch.float32).squeeze()
            attention_mask = torch.ones(input_values.shape[0], dtype=torch.long)

        # Augmentation: noise + gain (train only)
        if self.augment:
            snr = random.uniform(CFG.audio_snr_min, CFG.audio_snr_max)
            input_values = self._add_noise(input_values, snr)
            gain = random.uniform(CFG.audio_gain_min, CFG.audio_gain_max)
            input_values = input_values * gain
            input_values = torch.clamp(input_values, -1.0, 1.0)

        label = parse_label(row, self.label_map)

        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': row['sample_id']
        }


class FaceEmotionDataset(Dataset):
    """
    Load preprocessed face artifacts from Notebook 5 / Preprocessing.
    Artifact: .pt tensor [3, 224, 224] or image file.
    """
    def __init__(self, manifest_df: pd.DataFrame, artifact_dir: str, split: str, augment: bool = False):
        self.manifest = manifest_df.reset_index(drop=True)
        self.artifact_dir = resolve_artifact_dir(artifact_dir, split, "face")
        self.augment = augment and (split == "train")

        self.label_map = {}
        for idx, cls in enumerate(CFG.emotion_classes):
            self.label_map[cls] = idx
            self.label_map[cls.lower()] = idx
            self.label_map[cls.capitalize()] = idx
            self.label_map[cls.upper()] = idx

        if 'modality' in self.manifest.columns:
            self.manifest = self.manifest[self.manifest['modality'].isin(['face', 'image'])].reset_index(drop=True)

        valid_rows = []
        for _, row in self.manifest.iterrows():
            for ext in ['.pt', '.jpg', '.png', '.jpeg']:
                if (self.artifact_dir / f"{row['sample_id']}{ext}").exists():
                    valid_rows.append(row)
                    break
        self.manifest = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[Face {split}] Valid samples: {len(self.manifest)}")

        self.normalize = VT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = VT.ColorJitter(brightness=(CFG.face_brightness_min, CFG.face_brightness_max), contrast=(0.8, 1.2))
        self.gaussian_blur = VT.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        artifact_path = None
        for ext in ['.pt', '.jpg', '.png', '.jpeg']:
            path = self.artifact_dir / f"{row['sample_id']}{ext}"
            if path.exists():
                artifact_path = path
                break

        if artifact_path is None or not artifact_path.exists():
            image = torch.zeros((3, CFG.face_size[0], CFG.face_size[1]), dtype=torch.float32)
        elif artifact_path.suffix == '.pt':
            image = torch.load(artifact_path, map_location='cpu')
            if isinstance(image, dict):
                image = image.get('pixel_values', image.get('image', image.get('face')))
            if not torch.is_tensor(image):
                image = torch.tensor(image, dtype=torch.float32)
            if image.dim() == 2:
                image = image.unsqueeze(0).repeat(3, 1, 1)
            elif image.dim() == 3 and image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            if image.max() > 1.0:
                image = image / 255.0
            image = image.float()

            if self.augment:
                image = self.color_jitter(image)
                if random.random() < 0.3:
                    image = self.gaussian_blur(image)
            image = self.normalize(image)
        else:
            img = Image.open(artifact_path).convert('RGB')
            if self.augment:
                img = self.color_jitter(img)
                if random.random() < 0.3:
                    img = self.gaussian_blur(img)
            image = VT.ToTensor()(img)
            image = self.normalize(image)

        label = parse_label(row, self.label_map)

        return {
            'pixel_values': image,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': row['sample_id']
        }


class TextEmotionDataset(Dataset):
    """
    Load preprocessed text artifacts from Notebook 5 / Preprocessing.
    Artifact: .pt file containing {'input_ids': tensor, 'attention_mask': tensor}.
    """
    def __init__(self, manifest_df: pd.DataFrame, artifact_dir: str, split: str,
                 tokenizer=None, augment: bool = False):
        self.manifest = manifest_df.reset_index(drop=True)
        self.artifact_dir = resolve_artifact_dir(artifact_dir, split, "text")
        self.tokenizer = tokenizer
        self.augment = False  # Text augmentation disabled per architecture contract

        self.label_map = {}
        for idx, cls in enumerate(CFG.emotion_classes):
            self.label_map[cls] = idx
            self.label_map[cls.lower()] = idx
            self.label_map[cls.capitalize()] = idx
            self.label_map[cls.upper()] = idx

        if 'modality' in self.manifest.columns:
            self.manifest = self.manifest[self.manifest['modality'] == 'text'].reset_index(drop=True)

        valid_rows = []
        for _, row in self.manifest.iterrows():
            if (self.artifact_dir / f"{row['sample_id']}.pt").exists():
                valid_rows.append(row)
            elif 'text' in row and pd.notna(row['text']):
                valid_rows.append(row)
        self.manifest = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[Text {split}] Valid samples: {len(self.manifest)}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        artifact_path = self.artifact_dir / f"{row['sample_id']}.pt"

        if artifact_path.exists():
            data = torch.load(artifact_path, map_location='cpu')
            if isinstance(data, dict):
                input_ids = data['input_ids'].squeeze().long()
                attention_mask = data['attention_mask'].squeeze().long()
            else:
                input_ids = data.squeeze().long()
                attention_mask = torch.ones_like(input_ids)
        else:
            text = str(row['text'])
            if self.tokenizer is not None:
                encoded = self.tokenizer(
                    text,
                    max_length=CFG.text_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].squeeze(0).long()
                attention_mask = encoded['attention_mask'].squeeze(0).long()
            else:
                input_ids = torch.zeros(CFG.text_max_length, dtype=torch.long)
                attention_mask = torch.zeros(CFG.text_max_length, dtype=torch.long)

        label = parse_label(row, self.label_map)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': row['sample_id']
        }


class VideoEmotionDataset(Dataset):
    """
    Load preprocessed video artifacts from Notebook 5 / Preprocessing.
    Artifact: .pt file containing [T, C, H, W] tensor of 16 frames.
    """
    def __init__(self, manifest_df: pd.DataFrame, artifact_dir: str, split: str, augment: bool = False):
        self.manifest = manifest_df.reset_index(drop=True)
        self.artifact_dir = resolve_artifact_dir(artifact_dir, split, "video")
        self.augment = augment and (split == "train")

        self.label_map = {}
        for idx, cls in enumerate(CFG.emotion_classes):
            self.label_map[cls] = idx
            self.label_map[cls.lower()] = idx
            self.label_map[cls.capitalize()] = idx
            self.label_map[cls.upper()] = idx

        if 'modality' in self.manifest.columns:
            self.manifest = self.manifest[self.manifest['modality'] == 'video'].reset_index(drop=True)

        valid_rows = []
        for _, row in self.manifest.iterrows():
            if (self.artifact_dir / f"{row['sample_id']}.pt").exists():
                valid_rows.append(row)
        self.manifest = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[Video {split}] Valid samples: {len(self.manifest)}")

        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
        self.color_jitter = VT.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        artifact_path = self.artifact_dir / f"{row['sample_id']}.pt"

        video = torch.load(artifact_path, map_location='cpu')
        if isinstance(video, dict):
            video = video.get('pixel_values', video.get('video', video.get('frames')))
        if not torch.is_tensor(video):
            video = torch.tensor(video, dtype=torch.float32)

        # Ensure 4D [T, C, H, W]
        if video.dim() == 3:
            video = video.unsqueeze(0)
        if video.shape[1] not in [1, 3] and video.shape[-1] in [1, 3]:
            video = video.permute(0, 3, 1, 2)

        if video.max() > 1.0:
            video = video / 255.0
        video = video.float()

        if self.augment:
            if random.random() < 0.5:
                video = torch.flip(video, dims=[-1])
            frames = []
            for t in range(video.shape[0]):
                frame = self.color_jitter(video[t])
                frames.append(frame)
            video = torch.stack(frames, dim=0)

        video = (video - self.mean) / self.std
        label = parse_label(row, self.label_map)

        return {
            'pixel_values': video,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': row['sample_id']
        }


def collate_audio(batch):
    """Collate function for audio with variable lengths."""
    input_values = [item['input_values'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    sample_ids = [item['sample_id'] for item in batch]

    max_len = max(iv.shape[0] for iv in input_values)
    padded = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, iv in enumerate(input_values):
        length = iv.shape[0]
        padded[i, :length] = iv
        attention_mask[i, :length] = 1

    return {
        'input_values': padded,
        'attention_mask': attention_mask,
        'label': labels,
        'sample_id': sample_ids
    }


def collate_text(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'sample_id': [item['sample_id'] for item in batch]
    }


def collate_face(batch):
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'sample_id': [item['sample_id'] for item in batch]
    }


def collate_video(batch):
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'sample_id': [item['sample_id'] for item in batch]
    }


class QuadraDataset(Dataset):
    """
    Dataset for Quadra training.
    Loads pre-extracted embeddings from all four modalities.
    """
    def __init__(self, manifest_df: pd.DataFrame,
                 audio_embs: Dict[str, torch.Tensor],
                 face_embs: Dict[str, torch.Tensor],
                 text_embs: Dict[str, torch.Tensor],
                 video_embs: Dict[str, torch.Tensor],
                 split: str = "train"):
        self.label_map = {cls: idx for idx, cls in enumerate(CFG.emotion_classes)}

        # Find intersection of all available embeddings
        valid_ids = set(manifest_df['sample_id'])
        valid_ids = valid_ids.intersection(audio_embs.keys(), face_embs.keys(),
                                           text_embs.keys(), video_embs.keys())

        self.samples = []
        for _, row in manifest_df.iterrows():
            if row['sample_id'] in valid_ids:
                self.samples.append(row)

        self.samples = pd.DataFrame(self.samples).reset_index(drop=True)
        self.audio_embs = audio_embs
        self.face_embs = face_embs
        self.text_embs = text_embs
        self.video_embs = video_embs
        self.split = split

        print(f"[Quadra {split}] Valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        sid = row['sample_id']

        return {
            'audio_emb': self.audio_embs[sid],
            'face_emb': self.face_embs[sid],
            'text_emb': self.text_embs[sid],
            'video_emb': self.video_embs[sid],
            'label': torch.tensor(self.label_map[row['emotion']], dtype=torch.long),
            'sample_id': sid
        }


def collate_quadra(batch):
    return {
        'audio_emb': torch.stack([item['audio_emb'] for item in batch]),
        'face_emb': torch.stack([item['face_emb'] for item in batch]),
        'text_emb': torch.stack([item['text_emb'] for item in batch]),
        'video_emb': torch.stack([item['video_emb'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'sample_id': [item['sample_id'] for item in batch]
    }
