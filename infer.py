"""
Inference entry point for TerraMind Tiny Ship Classifier.

Assumes dataset + model are already placed in expected directories.
Running `python infer.py` will:
1) Load model
2) Run inference on validation chips
3) Save predictions CSV + metrics JSON
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train import TerraMindTinyShipClassifier, apply_lora


# ----------------------
# Config (unchanged)
# ----------------------
CONFIG = {
    "model_path": "terramind_tiny_ship_classifier.pth",
    "val_processed_dir": "data/processed_val_chips",
    "metadata_file": "chips_metadata.csv",
    "batch_size": 64,
    "num_workers": 0,
    "decision_threshold": 0.5,
    "use_lora": True,
    "predictions_csv": "val_predictions.csv",
    "metrics_json": "val_metrics.json",
}


# ----------------------
# Metrics
# ----------------------
def compute_binary_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())

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


# ----------------------
# Dataset
# ----------------------
class ProcessedValClassificationDataset(Dataset):
    def __init__(self, processed_dir: str, metadata_file: str):
        self.processed_dir = Path(processed_dir)
        meta_path = self.processed_dir / metadata_file

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_csv(meta_path)

        if "chip_file" not in df.columns:
            raise RuntimeError("Missing chip_file column")
        if "ship_present" not in df.columns:
            raise RuntimeError("Missing ship_present column")

        df = df[df["ship_present"].isin([0, 1])].copy()
        df["chip_path"] = df["chip_file"].apply(lambda x: str(self.processed_dir / str(x)))
        df = df[df["chip_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError("No valid chips found")

        for col in ["draw_x1", "draw_y1", "draw_x2", "draw_y2"]:
            if col not in df.columns:
                df[col] = np.nan

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chip = np.load(row["chip_path"]).astype(np.float32)

        if chip.ndim != 3 or chip.shape[0] != 2:
            raise RuntimeError(f"Invalid chip shape {chip.shape}")

        # Normalize
        for c in range(chip.shape[0]):
            p2, p98 = np.percentile(chip[c], [2, 98])
            chip[c] = np.clip((chip[c] - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)

        image = torch.from_numpy(chip)
        label = torch.tensor(float(row["ship_present"]), dtype=torch.float32)

        draw_box = (
            float(row["draw_x1"]) if pd.notna(row["draw_x1"]) else np.nan,
            float(row["draw_y1"]) if pd.notna(row["draw_y1"]) else np.nan,
            float(row["draw_x2"]) if pd.notna(row["draw_x2"]) else np.nan,
            float(row["draw_y2"]) if pd.notna(row["draw_y2"]) else np.nan,
        )

        return image, label, row["chip_file"], draw_box


# ----------------------
# Collate
# ----------------------
def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    chip_files = [b[2] for b in batch]
    draw_boxes = [b[3] for b in batch]
    return images, labels, chip_files, draw_boxes


# ----------------------
# Main
# ----------------------
def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    threshold = float(CONFIG["decision_threshold"])

    dataset = ProcessedValClassificationDataset(
        CONFIG["val_processed_dir"],
        CONFIG["metadata_file"],
    )

    loader = DataLoader(
        dataset,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=False,
        num_workers=int(CONFIG["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(int(CONFIG["num_workers"]) > 0),
        collate_fn=collate_fn,
    )

    # Model
    model = TerraMindTinyShipClassifier()
    if CONFIG["use_lora"]:
        model = apply_lora(model)

    model_path = Path(CONFIG["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    # Inference
    all_rows, probs_all, targets_all = [], [], []

    with torch.no_grad():
        for images, labels, chip_files, draw_boxes in tqdm(loader):
            images = images.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=amp_enabled):
                probs = torch.sigmoid(model(images))

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            probs_all.append(probs_np)
            targets_all.append(labels_np)

            for i, chip_file in enumerate(chip_files):
                prob = float(probs_np[i])
                pred = int(prob >= threshold)
                dx1, dy1, dx2, dy2 = draw_boxes[i]

                all_rows.append(
                    {
                        "chip_file": chip_file,
                        "target_ship_present": int(labels_np[i]),
                        "pred_ship_prob": prob,
                        "pred_ship_present": pred,
                        "draw_box": int(pred == 1),
                        "draw_x1": float(dx1) if pred == 1 and not np.isnan(dx1) else np.nan,
                        "draw_y1": float(dy1) if pred == 1 and not np.isnan(dy1) else np.nan,
                        "draw_x2": float(dx2) if pred == 1 and not np.isnan(dx2) else np.nan,
                        "draw_y2": float(dy2) if pred == 1 and not np.isnan(dy2) else np.nan,
                    }
                )

    # Save outputs
    preds_df = pd.DataFrame(all_rows)
    out_dir = Path(CONFIG["val_processed_dir"])

    out_csv = out_dir / CONFIG["predictions_csv"]
    preds_df.to_csv(out_csv, index=False)

    probs_cat = np.concatenate(probs_all, axis=0)
    targets_cat = np.concatenate(targets_all, axis=0).astype(int)
    preds_cat = (probs_cat >= threshold).astype(int)

    metrics = compute_binary_metrics(preds_cat, targets_cat)
    metrics["num_samples"] = int(len(targets_cat))
    metrics["decision_threshold"] = threshold
    metrics["predictions_csv"] = str(out_csv)

    out_json = out_dir / CONFIG["metrics_json"]
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    return 0


# ----------------------
# Entry
# ----------------------
if __name__ == "__main__":
    raise SystemExit(main())