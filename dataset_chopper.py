import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# --- CONFIG ---
OUTPUT_META_CSV = "chips_metadata.csv"
CHIP_SIZE = 224
DEFAULT_BOX_SIZE = 20.0
DEFAULT_DRAW_BOX_SIZE = 24.0
BALANCE_DATASET = True
NEGATIVE_TO_POSITIVE_RATIO = 1.0
RANDOM_SEED = 42
CLEAN_OUTPUT = True

JOBS = [
    {
        "name": "train",
        "csv_path": "GRD_train.csv",
        "full_scene_dir": "data/chips",
        "output_dir": "data/processed_chips",
    },
    {
        "name": "val",
        "csv_path": "GRD_validation.csv",
        "full_scene_dir": "data/val_chips",
        "output_dir": "data/processed_val_chips",
    },
]


def extract_timestamp(product_id: str) -> Optional[str]:
    match = re.search(r"\d{8}t\d{6}", str(product_id).lower())
    return match.group(0) if match else None


def parse_is_vessel(value) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    return None


def get_scene_bbox(row: pd.Series, default_box_size: float) -> Optional[list[float]]:
    if (
        pd.notna(row.get("left"))
        and pd.notna(row.get("top"))
        and pd.notna(row.get("right"))
        and pd.notna(row.get("bottom"))
    ):
        left = float(row["left"])
        top = float(row["top"])
        right = float(row["right"])
        bottom = float(row["bottom"])
        x1, x2 = min(left, right), max(left, right)
        y1, y2 = min(top, bottom), max(top, bottom)
    else:
        if pd.isna(row.get("detect_scene_column")) or pd.isna(row.get("detect_scene_row")):
            return None
        col = float(row["detect_scene_column"])
        r = float(row["detect_scene_row"])
        half = float(default_box_size) / 2.0
        x1, y1, x2, y2 = col - half, r - half, col + half, r + half

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def build_scene_index(full_scene_dir: str) -> Dict[str, Dict[str, str]]:
    print("Finding satellite scenes...")
    root = Path(full_scene_dir)
    if not root.exists():
        raise FileNotFoundError(f"Scene directory not found: {full_scene_dir}")

    index: Dict[str, Dict[str, str]] = {}
    all_tiffs = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]

    for p in all_tiffs:
        name = p.name.lower()
        ts = extract_timestamp(name)
        if not ts:
            continue

        pol = None
        if "-vh-" in name or "_vh_" in name or "vh" in name:
            pol = "vh"
        elif "-vv-" in name or "_vv_" in name or "vv" in name:
            pol = "vv"

        if pol is None:
            continue

        if ts not in index:
            index[ts] = {}
        index[ts][pol] = str(p)

    print(f"Indexed {len(all_tiffs)} TIFF files and {len(index)} timestamps.")
    return index


def load_rows(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = df["GRD_product_identifier"].apply(extract_timestamp)
    df["ship_present"] = df["is_vessel"].apply(parse_is_vessel) if "is_vessel" in df.columns else None

    df = df[df["timestamp"].notna()]
    df = df[df["ship_present"].isin([0, 1])]
    df = df[df["detect_scene_column"].notna() & df["detect_scene_row"].notna()]
    return df.reset_index(drop=True)


def balance_rows(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df["ship_present"] == 1]
    neg = df[df["ship_present"] == 0]

    if len(pos) == 0 or len(neg) == 0:
        print("Skipping balancing: one class is empty.")
        return df

    target_neg = int(min(len(neg), round(len(pos) * NEGATIVE_TO_POSITIVE_RATIO)))
    if BALANCE_DATASET:
        neg = neg.sample(n=target_neg, random_state=RANDOM_SEED)

    out = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    print(
        f"Balanced rows: positives={int((out['ship_present'] == 1).sum())}, "
        f"negatives={int((out['ship_present'] == 0).sum())}"
    )
    return out


def clear_previous_outputs(output_dir: str) -> None:
    if not CLEAN_OUTPUT:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("ship_*.npy"):
        p.unlink(missing_ok=True)
    meta = out / OUTPUT_META_CSV
    if meta.exists():
        meta.unlink()


def run_job(job: Dict[str, str]) -> None:
    csv_path = job["csv_path"]
    full_scene_dir = job["full_scene_dir"]
    output_dir = job["output_dir"]
    job_name = job["name"]

    print(f"\n=== Processing {job_name} split ===")
    clear_previous_outputs(output_dir)

    chip_index = build_scene_index(full_scene_dir)
    df = load_rows(csv_path)

    # Keep rows where we have both VH and VV scenes.
    valid_ts = {k for k, v in chip_index.items() if "vh" in v and "vv" in v}
    df = df[df["timestamp"].isin(valid_ts)].reset_index(drop=True)

    if BALANCE_DATASET:
        df = balance_rows(df)

    print(f"Starting extraction from {len(df)} rows...")

    created = 0
    skipped_invalid_box = 0
    failed_reads = 0
    metadata_rows = []

    grouped = df.groupby("timestamp", sort=False)
    for ts, group in tqdm(grouped, total=grouped.ngroups, desc="Timestamps"):
        pair = chip_index.get(ts)
        if not pair or "vh" not in pair or "vv" not in pair:
            continue

        try:
            with rasterio.open(pair["vh"]) as vh_src, rasterio.open(pair["vv"]) as vv_src:
                for row in group.itertuples(index=False):
                    row_dict = row._asdict()
                    col = int(getattr(row, "detect_scene_column"))
                    r = int(getattr(row, "detect_scene_row"))
                    ship_present = int(getattr(row, "ship_present"))

                    window = Window(
                        col - CHIP_SIZE // 2,
                        r - CHIP_SIZE // 2,
                        CHIP_SIZE,
                        CHIP_SIZE,
                    )

                    try:
                        vh_chip = vh_src.read(
                            1,
                            window=window,
                            boundless=True,
                            fill_value=0,
                            out_shape=(CHIP_SIZE, CHIP_SIZE),
                        )
                        vv_chip = vv_src.read(
                            1,
                            window=window,
                            boundless=True,
                            fill_value=0,
                            out_shape=(CHIP_SIZE, CHIP_SIZE),
                        )
                    except Exception:
                        failed_reads += 1
                        continue

                    window_left = float(col - CHIP_SIZE // 2)
                    window_top = float(r - CHIP_SIZE // 2)

                    if ship_present == 1:
                        scene_bbox = get_scene_bbox(pd.Series(row_dict), DEFAULT_BOX_SIZE)
                        if scene_bbox is None:
                            skipped_invalid_box += 1
                            continue
                        bx1, by1, bx2, by2 = scene_bbox

                        cx1 = float(np.clip(bx1 - window_left, 0.0, CHIP_SIZE - 1.0))
                        cy1 = float(np.clip(by1 - window_top, 0.0, CHIP_SIZE - 1.0))
                        cx2 = float(np.clip(bx2 - window_left, 0.0, CHIP_SIZE - 1.0))
                        cy2 = float(np.clip(by2 - window_top, 0.0, CHIP_SIZE - 1.0))

                        if cx2 <= cx1 or cy2 <= cy1:
                            skipped_invalid_box += 1
                            continue

                        source_label_type = (
                            "explicit_bbox"
                            if (
                                pd.notna(row_dict.get("left"))
                                and pd.notna(row_dict.get("top"))
                                and pd.notna(row_dict.get("right"))
                                and pd.notna(row_dict.get("bottom"))
                            )
                            else "derived_center_box"
                        )
                    else:
                        cx1 = np.nan
                        cy1 = np.nan
                        cx2 = np.nan
                        cy2 = np.nan
                        source_label_type = "no_ship"

                    # Default draw box used for visualization when model predicts ship.
                    half_draw = DEFAULT_DRAW_BOX_SIZE / 2.0
                    draw_x1 = float(CHIP_SIZE / 2.0 - half_draw)
                    draw_y1 = float(CHIP_SIZE / 2.0 - half_draw)
                    draw_x2 = float(CHIP_SIZE / 2.0 + half_draw)
                    draw_y2 = float(CHIP_SIZE / 2.0 + half_draw)

                    chip_data = np.stack([vv_chip, vh_chip], axis=0).astype(np.float32)
                    output_name = f"ship_{ts}_{created:07d}.npy"
                    output_path = os.path.join(output_dir, output_name)
                    np.save(output_path, chip_data)

                    metadata_rows.append(
                        {
                            "chip_file": output_name,
                            "timestamp": ts,
                            "detect_scene_column": col,
                            "detect_scene_row": r,
                            "chip_size": CHIP_SIZE,
                            "ship_present": ship_present,
                            "center_x": CHIP_SIZE // 2,
                            "center_y": CHIP_SIZE // 2,
                            "x1": cx1,
                            "y1": cy1,
                            "x2": cx2,
                            "y2": cy2,
                            "x1_norm": (cx1 / float(CHIP_SIZE)) if ship_present == 1 else np.nan,
                            "y1_norm": (cy1 / float(CHIP_SIZE)) if ship_present == 1 else np.nan,
                            "x2_norm": (cx2 / float(CHIP_SIZE)) if ship_present == 1 else np.nan,
                            "y2_norm": (cy2 / float(CHIP_SIZE)) if ship_present == 1 else np.nan,
                            "draw_x1": draw_x1,
                            "draw_y1": draw_y1,
                            "draw_x2": draw_x2,
                            "draw_y2": draw_y2,
                            "source_label_type": source_label_type,
                        }
                    )
                    created += 1
        except Exception:
            failed_reads += len(group)

    meta_path = os.path.join(output_dir, OUTPUT_META_CSV)
    pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)

    out_df = pd.DataFrame(metadata_rows)
    pos_count = int((out_df["ship_present"] == 1).sum()) if len(out_df) else 0
    neg_count = int((out_df["ship_present"] == 0).sum()) if len(out_df) else 0

    print("Extraction complete.")
    print(f"Created chips: {created}")
    print(f"Positive chips: {pos_count}")
    print(f"Negative chips: {neg_count}")
    print(f"Skipped rows (invalid bbox for positives): {skipped_invalid_box}")
    print(f"Failed chip reads: {failed_reads}")
    print(f"Metadata CSV: {meta_path}")


if __name__ == "__main__":
    for job_cfg in JOBS:
        run_job(job_cfg)
