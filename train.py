import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from terratorch import BACKBONE_REGISTRY
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Edit values directly here.
CONFIG = {
    "train_processed_dir": "data/processed_chips",
    "val_processed_dir": "data/processed_val_chips",
    "metadata_file": "chips_metadata.csv",
    "epochs": 100,
    "batch_size": 16,
    "lr": 1e-4,
    "num_workers": 0,
    "output": "terramind_tiny_ship_classifier.pth",
    "seed": 42,
    "use_lora": True,
    "decision_threshold": 0.5,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_binary_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict:
    preds = (probs >= threshold).long()
    labels_i = labels.long()

    tp = int(((preds == 1) & (labels_i == 1)).sum().item())
    tn = int(((preds == 0) & (labels_i == 0)).sum().item())
    fp = int(((preds == 1) & (labels_i == 0)).sum().item())
    fn = int(((preds == 0) & (labels_i == 1)).sum().item())

    total = max(tp + tn + fp + fn, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


class ProcessedClassificationDataset(Dataset):
    def __init__(self, processed_dir: str, metadata_file: str):
        self.processed_dir = Path(processed_dir)
        meta_path = self.processed_dir / metadata_file
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_csv(meta_path)
        required = {"chip_file", "ship_present"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"Metadata missing required columns: {sorted(missing)}")

        df = df[df["ship_present"].isin([0, 1])].copy()
        df["chip_path"] = df["chip_file"].apply(lambda x: str(self.processed_dir / str(x)))
        df = df[df["chip_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError(f"No valid chips found in {self.processed_dir}")

        self.records = df

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        chip = np.load(row["chip_path"]).astype(np.float32)

        if chip.ndim != 3 or chip.shape[0] != 2:
            raise RuntimeError(f"Invalid chip shape {chip.shape} in {row['chip_path']}")

        # Channel-wise robust normalization.
        for c in range(chip.shape[0]):
            p2, p98 = np.percentile(chip[c], [2, 98])
            chip[c] = np.clip((chip[c] - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)

        image = torch.from_numpy(chip)
        label = torch.tensor(float(row["ship_present"]), dtype=torch.float32)
        return image, label


class TerraMindTinyShipClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BACKBONE_REGISTRY.build(
            "terramind_v1_tiny",
            pretrained=True,
            modalities=["S1GRD"],
        )

        with torch.no_grad():
            dummy = torch.randn(1, 2, 224, 224)
            feat = self._forward_features(dummy)
            hidden_dim = self._infer_feature_dim(feat)

        print(f"Detected backbone feature dimension: {hidden_dim}")

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone({"S1GRD": x})
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if isinstance(out, dict):
            out = list(out.values())[-1]
        return out

    @staticmethod
    def _infer_feature_dim(features: torch.Tensor) -> int:
        if features.ndim == 2:
            return int(features.shape[-1])
        if features.ndim == 3:
            return int(features.shape[-1])
        if features.ndim == 4:
            return int(features.shape[1])
        raise RuntimeError(f"Unsupported feature tensor shape: {tuple(features.shape)}")

    @staticmethod
    def _pool_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            return features
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 4:
            return features.mean(dim=(2, 3))
        raise RuntimeError(f"Unsupported feature tensor shape: {tuple(features.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._forward_features(x)
        pooled = self._pool_features(feat)
        logits = self.cls_head(pooled).squeeze(-1)
        return logits


def apply_lora(model: nn.Module) -> nn.Module:
    target_keywords = ("qkv", "q_proj", "k_proj", "v_proj", "query_key_value")
    target_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            target_modules.add(name.split(".")[-1])

    if not target_modules:
        print("LoRA skipped: no matching attention projection modules were found.")
        return model

    config = LoraConfig(
        r=1,
        lora_alpha=32,
        target_modules=sorted(target_modules),
        lora_dropout=0.1,
        bias="none",
    )

    try:
        lora_model = get_peft_model(model, config)
        print(f"LoRA enabled on modules: {', '.join(sorted(target_modules))}")
        return lora_model
    except Exception as exc:
        print(f"LoRA application failed ({exc}); continuing without LoRA.")
        return model


def build_loader(processed_dir: str, metadata_file: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = ProcessedClassificationDataset(processed_dir=processed_dir, metadata_file=metadata_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    amp_enabled: bool,
) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    probs_all = []
    labels_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            bs = images.size(0)
            total += bs
            total_loss += float(loss.item()) * bs
            probs_all.append(probs.detach().cpu())
            labels_all.append(labels.detach().cpu())

    probs_cat = torch.cat(probs_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    metrics = compute_binary_metrics(probs_cat, labels_cat, threshold)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def main() -> None:
    set_seed(int(CONFIG["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    print("Building train loader...")
    train_loader = build_loader(
        processed_dir=str(CONFIG["train_processed_dir"]),
        metadata_file=str(CONFIG["metadata_file"]),
        batch_size=int(CONFIG["batch_size"]),
        num_workers=int(CONFIG["num_workers"]),
        shuffle=True,
    )

    val_loader = None
    val_dir = str(CONFIG["val_processed_dir"]).strip()
    if val_dir:
        print("Building val loader...")
        val_loader = build_loader(
            processed_dir=val_dir,
            metadata_file=str(CONFIG["metadata_file"]),
            batch_size=int(CONFIG["batch_size"]),
            num_workers=int(CONFIG["num_workers"]),
            shuffle=False,
        )

    # Class imbalance handling with pos_weight for BCE loss.
    train_labels = train_loader.dataset.records["ship_present"].astype(int)
    pos_count = int((train_labels == 1).sum())
    neg_count = int((train_labels == 0).sum())
    pos_weight_value = float(neg_count / max(pos_count, 1))
    print(f"Train label counts: pos={pos_count}, neg={neg_count}, pos_weight={pos_weight_value:.4f}")

    print("Initializing TerraMind Tiny ship classifier...")
    model = TerraMindTinyShipClassifier()

    if bool(CONFIG["use_lora"]):
        print("Applying LoRA adapters...")
        model = apply_lora(model)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["lr"]))
    scaler = GradScaler(device.type, enabled=amp_enabled)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device, dtype=torch.float32)
    )

    threshold = float(CONFIG["decision_threshold"])
    epochs = int(CONFIG["epochs"])
    print(f"Training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        probs_all = []
        labels_all = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            probs = torch.sigmoid(logits)
            bs = images.size(0)
            total += bs
            running_loss += float(loss.item()) * bs
            probs_all.append(probs.detach().cpu())
            labels_all.append(labels.detach().cpu())

            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        probs_cat = torch.cat(probs_all, dim=0)
        labels_cat = torch.cat(labels_all, dim=0)
        train_metrics = compute_binary_metrics(probs_cat, labels_cat, threshold)
        train_metrics["loss"] = running_loss / max(total, 1)

        msg = (
            f"Epoch {epoch + 1} train | loss={train_metrics['loss']:.5f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"prec={train_metrics['precision']:.4f} "
            f"rec={train_metrics['recall']:.4f} "
            f"f1={train_metrics['f1']:.4f}"
        )

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, threshold, amp_enabled)
            msg += (
                f" || val_loss={val_metrics['loss']:.5f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_prec={val_metrics['precision']:.4f} "
                f"val_rec={val_metrics['recall']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )

        print(msg)

    output_path = str(CONFIG["output"])
    torch.save(model.state_dict(), output_path)
    print(f"Training finished. Weights saved to: {output_path}")


if __name__ == "__main__":
    main()
