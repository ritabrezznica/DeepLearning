#!/usr/bin/env python3
"""
Inference for Intel image classification (Challenge 2).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

NUM_CLASSES = 6
IMAGE_SIZE = 150

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


class LeanCNN(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_ckpt() -> Path:
    env = os.environ.get("CHALLENGE2_CKPT")
    if env:
        return Path(env)
    return _script_dir() / "best_model_exp_b.pth"


def _default_image_dir() -> Path:
    env = os.environ.get("CHALLENGE2_IMAGE_DIR")
    if env:
        return Path(env)
    base = _script_dir() / "data" / "seg_pred"
    nested = base / "seg_pred"
    if nested.is_dir():
        return nested
    return base


def _infer_image_column(df: pd.DataFrame) -> str:
    for name in ("path", "filepath", "filename", "image", "file"):
        if name in df.columns:
            return name
    return df.columns[0]


def _resolve_path(raw: str, image_dir: Path) -> Path:
    s = str(raw).strip()
    p = Path(s)
    if p.is_file():
        return p
    if p.is_absolute():
        return p
    script = _script_dir()
    for base in (Path.cwd(), script, image_dir):
        cand = base / s
        if cand.is_file():
            return cand
    return image_dir / s


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    ckpt_path = _default_ckpt()
    image_dir = _default_image_dir()

    if not ckpt_path.is_file():
        print(f"Missing checkpoint: {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeanCNN()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    df = pd.read_csv(input_path)
    col = _infer_image_column(df)

    preds: list[str] = []
    with torch.no_grad():
        for raw in df[col].astype(str):
            path = _resolve_path(raw, image_dir)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path} (from CSV value {raw!r})")
            img = Image.open(path).convert("RGB")
            x = val_transform(img).unsqueeze(0).to(device)
            logits = model(x)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            preds.append(CLASS_NAMES[pred_idx])

    out = pd.DataFrame({"prediction": preds})
    if "id" in df.columns:
        out.insert(0, "id", df["id"])
    out.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
