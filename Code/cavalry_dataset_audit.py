#!/usr/bin/env python3
"""
CAVALRY-AI Dataset Audit Script
--------------------------------
Purpose:
    Scan all CSE-CIC-IDS2018 CSV files in a folder, inspect structure,
    detect label columns, summarize rows/classes/missing values, and create
    an audit report for the next modeling stage.

Usage:
    python cavalry_dataset_audit.py --data_dir "D:\other\CAVALRY-AI\CSE-CIC-IDS2018" --sample_rows 200000

Outputs:
    dataset_audit_report.txt
    dataset_audit_report.json
    combined_label_distribution.csv
    per_file_summary.csv
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


POSSIBLE_LABEL_COLUMNS = [
    "Label", "label", "Class", "class", "Attack", "attack", "Attack_type",
    "attack_type", "Category", "category"
]


def safe_print(msg: str) -> None:
    print(msg, flush=True)


def find_csv_files(data_dir: Path):
    files = sorted(data_dir.glob("*.csv"))
    return files


def normalize_columns(columns):
    return [str(c).strip().replace("\ufeff", "") for c in columns]


def find_label_column(columns):
    stripped = normalize_columns(columns)

    for candidate in POSSIBLE_LABEL_COLUMNS:
        if candidate in stripped:
            return candidate

    lowered = {c.lower(): c for c in stripped}
    for candidate in POSSIBLE_LABEL_COLUMNS:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    likely = [c for c in stripped if "label" in c.lower() or "attack" in c.lower() or "class" in c.lower()]
    return likely[0] if likely else None


def read_csv_sample(path: Path, sample_rows: int):
    """
    Tries common encodings. Reads only sample_rows if provided.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error = None

    for enc in encodings:
        try:
            if sample_rows and sample_rows > 0:
                df = pd.read_csv(path, nrows=sample_rows, encoding=enc, low_memory=False)
            else:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
            df.columns = normalize_columns(df.columns)
            return df, enc, None
        except Exception as exc:
            last_error = str(exc)

    return None, None, last_error


def count_rows_fast(path: Path):
    """
    Counts file lines without loading the whole file.
    Returns approximate data rows, excluding header.
    """
    try:
        with open(path, "rb") as f:
            line_count = sum(1 for _ in f)
        return max(0, line_count - 1)
    except Exception:
        return None


def summarize_numeric(df: pd.DataFrame, max_cols: int = 20):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = {}

    for col in numeric_cols[:max_cols]:
        s = df[col].replace([np.inf, -np.inf], np.nan)
        summary[col] = {
            "mean": None if pd.isna(s.mean()) else float(s.mean()),
            "std": None if pd.isna(s.std()) else float(s.std()),
            "min": None if pd.isna(s.min()) else float(s.min()),
            "max": None if pd.isna(s.max()) else float(s.max()),
            "missing": int(s.isna().sum()),
            "infinite": int(np.isinf(df[col]).sum()) if np.issubdtype(df[col].dtype, np.number) else 0
        }

    return summary


def audit_file(path: Path, sample_rows: int):
    safe_print(f"\n[READING] {path.name}")

    row_count = count_rows_fast(path)
    file_size_mb = path.stat().st_size / (1024 * 1024)

    df, encoding, error = read_csv_sample(path, sample_rows)

    result = {
        "file": path.name,
        "path": str(path),
        "size_mb": round(file_size_mb, 2),
        "estimated_rows": row_count,
        "sample_rows_loaded": 0,
        "encoding": encoding,
        "read_error": error,
        "n_columns": None,
        "columns": [],
        "label_column": None,
        "label_distribution_sample": {},
        "missing_values_total_sample": None,
        "duplicate_rows_sample": None,
        "numeric_columns_count": None,
        "object_columns_count": None,
        "infinite_values_total_sample": None,
        "numeric_summary_sample": {}
    }

    if df is None:
        safe_print(f"[ERROR] Could not read {path.name}: {error}")
        return result

    result["sample_rows_loaded"] = int(len(df))
    result["n_columns"] = int(df.shape[1])
    result["columns"] = list(df.columns)

    label_col = find_label_column(df.columns)
    result["label_column"] = label_col

    missing_total = int(df.isna().sum().sum())
    result["missing_values_total_sample"] = missing_total
    result["duplicate_rows_sample"] = int(df.duplicated().sum())
    result["numeric_columns_count"] = int(len(df.select_dtypes(include=[np.number]).columns))
    result["object_columns_count"] = int(len(df.select_dtypes(include=["object"]).columns))

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 0:
        result["infinite_values_total_sample"] = int(np.isinf(numeric_df).sum().sum())
    else:
        result["infinite_values_total_sample"] = 0

    if label_col and label_col in df.columns:
        counts = df[label_col].astype(str).str.strip().value_counts(dropna=False).to_dict()
        result["label_distribution_sample"] = {str(k): int(v) for k, v in counts.items()}
        safe_print(f"[OK] Rows loaded: {len(df):,} | Columns: {df.shape[1]} | Label: {label_col}")
        safe_print(f"[LABELS] {result['label_distribution_sample']}")
    else:
        safe_print(f"[WARN] Rows loaded: {len(df):,} | Columns: {df.shape[1]} | No label column detected")

    result["numeric_summary_sample"] = summarize_numeric(df)

    return result


def build_reports(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "dataset_audit_report.json"
    txt_path = output_dir / "dataset_audit_report.txt"
    per_file_csv = output_dir / "per_file_summary.csv"
    label_csv = output_dir / "combined_label_distribution.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = []
    combined_labels = Counter()

    for r in results:
        rows.append({
            "file": r["file"],
            "size_mb": r["size_mb"],
            "estimated_rows": r["estimated_rows"],
            "sample_rows_loaded": r["sample_rows_loaded"],
            "n_columns": r["n_columns"],
            "label_column": r["label_column"],
            "missing_values_total_sample": r["missing_values_total_sample"],
            "duplicate_rows_sample": r["duplicate_rows_sample"],
            "numeric_columns_count": r["numeric_columns_count"],
            "object_columns_count": r["object_columns_count"],
            "infinite_values_total_sample": r["infinite_values_total_sample"],
            "read_error": r["read_error"]
        })

        for label, count in r.get("label_distribution_sample", {}).items():
            combined_labels[label] += count

    pd.DataFrame(rows).to_csv(per_file_csv, index=False)

    label_rows = [{"label": k, "sample_count": v} for k, v in combined_labels.most_common()]
    pd.DataFrame(label_rows).to_csv(label_csv, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("CAVALRY-AI DATASET AUDIT REPORT\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Files scanned: {len(results)}\n")
        f.write(f"Total estimated rows: {sum(r['estimated_rows'] or 0 for r in results):,}\n")
        f.write(f"Total sample rows loaded: {sum(r['sample_rows_loaded'] or 0 for r in results):,}\n\n")

        f.write("PER-FILE SUMMARY\n")
        f.write("-" * 40 + "\n")
        for r in results:
            f.write(f"\nFile: {r['file']}\n")
            f.write(f"  Size MB: {r['size_mb']}\n")
            f.write(f"  Estimated rows: {r['estimated_rows']}\n")
            f.write(f"  Sample rows loaded: {r['sample_rows_loaded']}\n")
            f.write(f"  Columns: {r['n_columns']}\n")
            f.write(f"  Label column: {r['label_column']}\n")
            f.write(f"  Missing values in sample: {r['missing_values_total_sample']}\n")
            f.write(f"  Duplicate rows in sample: {r['duplicate_rows_sample']}\n")
            f.write(f"  Infinite numeric values in sample: {r['infinite_values_total_sample']}\n")
            f.write(f"  Read error: {r['read_error']}\n")
            if r.get("label_distribution_sample"):
                f.write("  Label distribution in sample:\n")
                for label, count in sorted(r["label_distribution_sample"].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"    {label}: {count}\n")

        f.write("\nCOMBINED LABEL DISTRIBUTION FROM SAMPLES\n")
        f.write("-" * 40 + "\n")
        for label, count in combined_labels.most_common():
            f.write(f"{label}: {count}\n")

        f.write("\nRECOMMENDED NEXT STEPS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Send dataset_audit_report.txt and per_file_summary.csv output.\n")
        f.write("2. Confirm the detected label column.\n")
        f.write("3. Use sampled training first before full-file training because these CSV files are large.\n")
        f.write("4. Remove duplicate rows, clean NaN/Inf values, encode labels, and train baseline IDS models.\n")

    return {
        "json": str(json_path),
        "txt": str(txt_path),
        "per_file_csv": str(per_file_csv),
        "label_csv": str(label_csv)
    }


def main():
    parser = argparse.ArgumentParser(description="Audit CSE-CIC-IDS2018 CSV files for CAVALRY-AI.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Folder containing CSE-CIC-IDS2018 CSV files."
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=200000,
        help="Rows to load from each CSV. Use 0 to load full files, but this may require a lot of RAM."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder. Default: data_dir/audit_output"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        safe_print(f"[FATAL] Data directory does not exist: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "audit_output"

    csv_files = find_csv_files(data_dir)
    if not csv_files:
        safe_print(f"[FATAL] No CSV files found in: {data_dir}")
        sys.exit(1)

    safe_print("CAVALRY-AI DATASET AUDIT")
    safe_print("=" * 30)
    safe_print(f"Data directory: {data_dir}")
    safe_print(f"CSV files found: {len(csv_files)}")
    safe_print(f"Sample rows per file: {args.sample_rows}")

    start = time.time()
    results = []

    for file_path in csv_files:
        results.append(audit_file(file_path, args.sample_rows))

    outputs = build_reports(results, output_dir)

    elapsed = time.time() - start

    safe_print("\nDONE")
    safe_print("=" * 30)
    safe_print(f"Elapsed time: {elapsed:.2f} seconds")
    safe_print("Output files:")
    for name, path in outputs.items():
        safe_print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
