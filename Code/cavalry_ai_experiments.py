#!/usr/bin/env python3
"""
CAVALRY-AI Experimental Pipeline for CSE-CIC-IDS2018
====================================================

This script is designed for IEEE Transactions-style experimental reporting.

It performs:
    1. Multi-file CSE-CIC-IDS2018 loading
    2. Robust cleaning for NaN, Inf, duplicated headers, string numeric columns
    3. Binary and multiclass IDS experiments
    4. Multiple models
    5. Multiple ablation studies
    6. Multiple evaluation metrics
    7. Publication-ready tables and graphs
    8. Saved trained models and preprocessing artifacts

Recommended first run:
    python cavalry_ai_experiments.py --data_dir "D:\other\CAVALRY-AI\CSE-CIC-IDS2018" --max_rows_per_file 120000 --mode both

Faster test run:
    python cavalry_ai_experiments.py --data_dir "D:\other\CAVALRY-AI\CSE-CIC-IDS2018" --max_rows_per_file 20000 --mode both

Fuller run:
    python cavalry_ai_experiments.py --data_dir "D:\other\CAVALRY-AI\CSE-CIC-IDS2018" --max_rows_per_file 300000 --mode both

Notes:
    - The full dataset is large. Start with 20k to 120k rows per file.
    - Some CSE-CIC-IDS2018 files contain repeated header rows inside the data.
      This script removes rows where Label == "Label".
    - Some files store numeric columns as object because of dirty rows.
      This script converts feature columns to numeric safely.
"""

import argparse
import json
import os
import time
import warnings
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for Windows/Anaconda batch runs
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


RANDOM_STATE = 42

IDENTIFIER_COLUMNS = {
    "Flow ID", "Src IP", "Dst IP", "Src Port", "Timestamp"
}

GRAPH_SOURCE_COLUMNS = {
    "Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"
}

BASE_CORE_FEATURES = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Size Avg",
]

PORT_PROTOCOL_FEATURES = [
    "Dst Port",
    "Src Port",
    "Protocol",
]

TEMPORAL_FEATURES = [
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

TCP_FLAG_FEATURES = [
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_columns(columns) -> List[str]:
    return [str(c).strip().replace("\ufeff", "") for c in columns]


def find_csv_files(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])


def read_one_csv(path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    read_kwargs = {
        "low_memory": False,
        "encoding": "utf-8",
    }
    if max_rows and max_rows > 0:
        read_kwargs["nrows"] = max_rows

    try:
        df = pd.read_csv(path, **read_kwargs)
    except UnicodeDecodeError:
        read_kwargs["encoding"] = "latin1"
        df = pd.read_csv(path, **read_kwargs)

    df.columns = normalize_columns(df.columns)
    df["Source_File"] = path.name
    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "Label" not in df.columns:
        raise ValueError("No Label column found. Confirm the dataset format.")

    df["Label"] = df["Label"].astype(str).str.strip()

    bad_labels = {
        "",
        "nan",
        "NaN",
        "None",
        "Label",
    }
    df = df[~df["Label"].isin(bad_labels)].copy()

    # Standardize common spelling variants without changing class meaning.
    replacements = {
        "Infilteration": "Infiltration",
        "SSH-Bruteforce": "SSH-BruteForce",
        "DDoS attacks-LOIC-HTTP": "DDoS-LOIC-HTTP",
        "DDOS attack-HOIC": "DDoS-HOIC",
        "DDOS attack-LOIC-UDP": "DDoS-LOIC-UDP",
        "DoS attacks-Hulk": "DoS-Hulk",
        "DoS attacks-GoldenEye": "DoS-GoldenEye",
        "DoS attacks-Slowloris": "DoS-Slowloris",
        "DoS attacks-SlowHTTPTest": "DoS-SlowHTTPTest",
    }
    df["Label"] = df["Label"].replace(replacements)
    return df


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight Cyber Memory Graph-inspired features.

    These do not require Neo4j. They approximate graph context using
    frequency and degree-style counts from available flow identifiers.
    """
    df = df.copy()

    if "Src IP" in df.columns:
        src_counts = df["Src IP"].astype(str).value_counts()
        df["CMG_src_ip_frequency"] = df["Src IP"].astype(str).map(src_counts).astype(float)

    if "Dst IP" in df.columns:
        dst_counts = df["Dst IP"].astype(str).value_counts()
        df["CMG_dst_ip_frequency"] = df["Dst IP"].astype(str).map(dst_counts).astype(float)

    if "Src IP" in df.columns and "Dst IP" in df.columns:
        pair = df["Src IP"].astype(str) + "->" + df["Dst IP"].astype(str)
        pair_counts = pair.value_counts()
        df["CMG_edge_frequency"] = pair.map(pair_counts).astype(float)

        src_degree = df.groupby("Src IP")["Dst IP"].nunique()
        dst_degree = df.groupby("Dst IP")["Src IP"].nunique()
        df["CMG_src_unique_dst_degree"] = df["Src IP"].map(src_degree).astype(float)
        df["CMG_dst_unique_src_degree"] = df["Dst IP"].map(dst_degree).astype(float)

    if "Dst Port" in df.columns:
        port_counts = df["Dst Port"].astype(str).value_counts()
        df["CMG_dst_port_frequency"] = df["Dst Port"].astype(str).map(port_counts).astype(float)

    return df


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Risk-Weighted Agent Routing-inspired engineered features.

    These features approximate the Sentinel Agent's risk signal from flow
    intensity, port/protocol behavior, TCP flags, and packet statistics.
    """
    df = df.copy()

    numeric_source_cols = [
        "Flow Byts/s",
        "Flow Pkts/s",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "SYN Flag Cnt",
        "RST Flag Cnt",
        "ACK Flag Cnt",
        "Dst Port",
        "Protocol",
    ]

    for col in numeric_source_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Flow Byts/s" in df.columns and "Flow Pkts/s" in df.columns:
        df["RWAR_flow_intensity"] = np.log1p(df["Flow Byts/s"].clip(lower=0)) + np.log1p(df["Flow Pkts/s"].clip(lower=0))

    if "Tot Fwd Pkts" in df.columns and "Tot Bwd Pkts" in df.columns:
        df["RWAR_packet_asymmetry"] = (df["Tot Fwd Pkts"] - df["Tot Bwd Pkts"]).abs() / (
            df["Tot Fwd Pkts"] + df["Tot Bwd Pkts"] + 1.0
        )

    if "TotLen Fwd Pkts" in df.columns and "TotLen Bwd Pkts" in df.columns:
        df["RWAR_byte_asymmetry"] = (df["TotLen Fwd Pkts"] - df["TotLen Bwd Pkts"]).abs() / (
            df["TotLen Fwd Pkts"] + df["TotLen Bwd Pkts"] + 1.0
        )

    if "SYN Flag Cnt" in df.columns and "ACK Flag Cnt" in df.columns:
        df["RWAR_syn_ack_ratio"] = df["SYN Flag Cnt"] / (df["ACK Flag Cnt"] + 1.0)

    if "RST Flag Cnt" in df.columns and "ACK Flag Cnt" in df.columns:
        df["RWAR_rst_ack_ratio"] = df["RST Flag Cnt"] / (df["ACK Flag Cnt"] + 1.0)

    if "Dst Port" in df.columns:
        common_ports = {20, 21, 22, 23, 25, 53, 80, 110, 139, 143, 443, 445, 3389}
        df["RWAR_common_service_port"] = df["Dst Port"].isin(common_ports).astype(float)

    if "Protocol" in df.columns:
        df["RWAR_protocol_tcp"] = (df["Protocol"] == 6).astype(float)
        df["RWAR_protocol_udp"] = (df["Protocol"] == 17).astype(float)

    risk_cols = [c for c in df.columns if c.startswith("RWAR_")]
    if risk_cols:
        tmp = df[risk_cols].replace([np.inf, -np.inf], np.nan)
        tmp = tmp.fillna(tmp.median(numeric_only=True))
        # Normalize roughly for routing score.
        ranked = tmp.rank(pct=True)
        df["RWAR_risk_score"] = ranked.mean(axis=1)

    return df


def load_dataset(data_dir: Path, max_rows_per_file: int) -> pd.DataFrame:
    files = find_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    log(f"[INFO] Found {len(files)} CSV files.")

    for path in files:
        log(f"[LOAD] {path.name}")
        df = read_one_csv(path, max_rows_per_file)
        df = clean_labels(df)
        frames.append(df)
        log(f"       rows after label cleaning: {len(df):,}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    log(f"[INFO] Combined rows: {len(combined):,}")
    log(f"[INFO] Combined columns: {combined.shape[1]:,}")

    return combined


def make_binary_label(y: pd.Series) -> pd.Series:
    return np.where(y.astype(str).str.lower().eq("benign"), "Benign", "Attack")


def filter_min_classes(df: pd.DataFrame, label_col: str, min_class_count: int) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    keep = counts[counts >= min_class_count].index
    dropped = counts[counts < min_class_count]

    if len(dropped) > 0:
        log("[WARN] Dropping rare classes below min_class_count:")
        for label, count in dropped.items():
            log(f"       {label}: {count}")

    return df[df[label_col].isin(keep)].copy()


def get_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_numeric_candidates = []
    for col in df.columns:
        if col in {"Label", "Binary_Label", "Source_File"}:
            continue
        if col in IDENTIFIER_COLUMNS:
            continue
        all_numeric_candidates.append(col)

    available_core = [c for c in BASE_CORE_FEATURES if c in df.columns]
    available_port_proto = [c for c in PORT_PROTOCOL_FEATURES if c in df.columns and c not in IDENTIFIER_COLUMNS]
    available_temporal = [c for c in TEMPORAL_FEATURES if c in df.columns]
    available_flags = [c for c in TCP_FLAG_FEATURES if c in df.columns]
    graph_features = [c for c in df.columns if c.startswith("CMG_")]
    risk_features = [c for c in df.columns if c.startswith("RWAR_")]

    def unique(seq):
        return list(dict.fromkeys(seq))

    return {
        "A0_CoreFlow": unique(available_core),
        "A1_Core_PortProtocol": unique(available_core + available_port_proto),
        "A2_Core_PortProtocol_Temporal": unique(available_core + available_port_proto + available_temporal),
        "A3_AllNumeric_NoCMG_NoRWAR": unique([
            c for c in all_numeric_candidates
            if not c.startswith("CMG_") and not c.startswith("RWAR_")
        ]),
        "A4_RWAR_RiskRouting": unique([
            c for c in all_numeric_candidates
            if not c.startswith("CMG_")
        ]),
        "A5_CMG_GraphMemory": unique([
            c for c in all_numeric_candidates
            if not c.startswith("RWAR_")
        ]),
        "A6_Full_CAVALRY_AI": unique(all_numeric_candidates),
        "A7_TCPFlagsOnly": unique(available_flags),
    }


def coerce_features_numeric(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def build_models(task_mode: str) -> Dict[str, object]:
    """
    Models selected to provide classical, ensemble, linear, and boosting baselines.
    """
    models = {
        "GaussianNB": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GaussianNB()),
        ]),
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(
                max_iter=1200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                solver="saga" if task_mode == "multiclass" else "liblinear"
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=160,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=160,
                random_state=RANDOM_STATE
            )),
        ]),
    }

    return models


def predict_scores(model, X_test):
    """
    Returns probability-like scores when available.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X_test)
        except Exception:
            return None

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
            return scores
        except Exception:
            return None

    return None


def compute_auc_metrics(y_true_enc, y_pred_scores, labels_count: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns ROC-AUC macro, PR-AUC macro, log loss where possible.
    """
    roc_macro = None
    pr_macro = None
    ll = None

    if y_pred_scores is None:
        return roc_macro, pr_macro, ll

    try:
        if labels_count == 2:
            if len(np.asarray(y_pred_scores).shape) == 2:
                positive_scores = y_pred_scores[:, 1]
                ll = float(log_loss(y_true_enc, y_pred_scores))
            else:
                positive_scores = y_pred_scores
            roc_macro = float(roc_auc_score(y_true_enc, positive_scores))
            pr_macro = float(average_precision_score(y_true_enc, positive_scores))
        else:
            if len(np.asarray(y_pred_scores).shape) == 2 and y_pred_scores.shape[1] == labels_count:
                y_bin = label_binarize(y_true_enc, classes=np.arange(labels_count))
                roc_macro = float(roc_auc_score(y_bin, y_pred_scores, average="macro", multi_class="ovr"))
                pr_macro = float(average_precision_score(y_bin, y_pred_scores, average="macro"))
                ll = float(log_loss(y_true_enc, y_pred_scores, labels=np.arange(labels_count)))
    except Exception:
        pass

    return roc_macro, pr_macro, ll


def evaluate_one(model, X_train, X_test, y_train, y_test, label_encoder, model_name: str, feature_set: str, task_mode: str):
    start_train = time.perf_counter()
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_pred = fitted.predict(X_test)
    inference_seconds = time.perf_counter() - start_pred

    y_scores = predict_scores(fitted, X_test)

    labels_count = len(label_encoder.classes_)
    roc_macro, pr_macro, ll = compute_auc_metrics(y_test, y_scores, labels_count)

    result = {
        "task": task_mode,
        "feature_set": feature_set,
        "model": model_name,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(labels_count),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "roc_auc_macro_ovr": roc_macro,
        "pr_auc_macro": pr_macro,
        "log_loss": ll,
        "train_seconds": float(train_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_ms_per_sample": float((inference_seconds / max(1, X_test.shape[0])) * 1000),
    }

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(labels_count),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(labels_count))

    return fitted, result, report, cm


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, output_path: Path) -> None:
    plt.figure(figsize=(max(8, len(labels) * 0.65), max(6, len(labels) * 0.55)))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close('all')
    gc.collect()


def plot_metric_bars(df: pd.DataFrame, metric: str, group_col: str, title: str, output_path: Path, top_n: Optional[int] = None) -> None:
    tmp = df.copy()
    if top_n:
        tmp = tmp.sort_values(metric, ascending=False).head(top_n)

    labels = tmp[group_col].astype(str).tolist()
    values = tmp[metric].astype(float).tolist()

    plt.figure(figsize=(max(10, len(labels) * 0.55), 6))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close('all')
    gc.collect()


def plot_runtime_vs_f1(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(df["inference_ms_per_sample"], df["f1_macro"])
    for _, row in df.iterrows():
        label = f"{row['model']}|{row['feature_set'].replace('A', '')}"
        plt.annotate(label, (row["inference_ms_per_sample"], row["f1_macro"]), fontsize=7)
    plt.xlabel("Inference Time per Sample (ms)")
    plt.ylabel("Macro F1-Score")
    plt.title("Accuracy-Latency Tradeoff")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close('all')
    gc.collect()


def save_label_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["Label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts.to_csv(out_dir / "label_distribution_cleaned.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(counts)), counts["count"].values)
    plt.xticks(range(len(counts)), counts["label"].values, rotation=75, ha="right")
    plt.ylabel("Rows")
    plt.title("Cleaned Label Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_label_distribution_cleaned.png", dpi=300, bbox_inches="tight")
    plt.close('all')
    gc.collect()


def run_experiments(
    df: pd.DataFrame,
    output_dir: Path,
    task_mode: str,
    test_size: float,
    min_class_count: int,
    max_total_rows: int,
    top_k_features: int,
) -> None:
    task_dir = output_dir / task_mode
    figs_dir = task_dir / "figures"
    models_dir = task_dir / "models"
    ensure_dir(task_dir)
    ensure_dir(figs_dir)
    ensure_dir(models_dir)

    df = df.copy()

    if task_mode == "binary":
        df["Target"] = make_binary_label(df["Label"])
    elif task_mode == "multiclass":
        df["Target"] = df["Label"]
        df = filter_min_classes(df, "Target", min_class_count=min_class_count)
    else:
        raise ValueError("task_mode must be binary or multiclass")

    if max_total_rows and max_total_rows > 0 and len(df) > max_total_rows:
        # Stratified downsample for fair class representation.
        df, _ = train_test_split(
            df,
            train_size=max_total_rows,
            random_state=RANDOM_STATE,
            stratify=df["Target"],
        )
        df = df.copy()
        log(f"[INFO] Downsampled {task_mode} data to {len(df):,} rows.")

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df["Target"].astype(str))

    feature_sets = get_feature_sets(df)
    models = build_models(task_mode)

    all_results = []
    report_records = []

    for feature_set_name, feature_cols in feature_sets.items():
        if not feature_cols:
            log(f"[SKIP] {feature_set_name}: no available columns")
            continue

        log(f"\n[FEATURE SET] {feature_set_name}: {len(feature_cols)} candidate features")
        X_all = coerce_features_numeric(df, feature_cols)

        # Remove columns that are entirely missing after conversion.
        valid_cols = [c for c in X_all.columns if not X_all[c].isna().all()]
        X_all = X_all[valid_cols]

        if X_all.shape[1] == 0:
            log(f"[SKIP] {feature_set_name}: all features invalid after numeric conversion")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y_all,
        )

        # Feature-selection ablation variant for A6.
        current_models = models.copy()

        if feature_set_name == "A6_Full_CAVALRY_AI" and X_train.shape[1] > top_k_features:
            selected_name = f"A6b_Full_CAVALRY_AI_SelectKBest_{top_k_features}"
            log(f"[FEATURE SELECTION] Creating {selected_name}")
            selector_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", SelectKBest(mutual_info_classif, k=top_k_features)),
            ])
            X_train_selected = selector_pipe.fit_transform(X_train, y_train)
            X_test_selected = selector_pipe.transform(X_test)

            selected_mask = selector_pipe.named_steps["selector"].get_support()
            selected_features = X_train.columns[selected_mask].tolist()
            pd.DataFrame({"selected_feature": selected_features}).to_csv(
                task_dir / f"{selected_name}_selected_features.csv",
                index=False,
            )
            joblib.dump(selector_pipe, models_dir / f"{selected_name}_selector.joblib")

            for model_name, model in current_models.items():
                log(f"[TRAIN] {task_mode} | {selected_name} | {model_name}")
                fitted, result, report, cm = evaluate_one(
                    model,
                    X_train_selected,
                    X_test_selected,
                    y_train,
                    y_test,
                    label_encoder,
                    model_name,
                    selected_name,
                    task_mode,
                )
                all_results.append(result)

                for label_name, values in report.items():
                    if isinstance(values, dict):
                        rec = {"task": task_mode, "feature_set": selected_name, "model": model_name, "label": label_name}
                        rec.update(values)
                        report_records.append(rec)

                if result["f1_macro"] >= 0:
                    safe_name = f"{task_mode}_{selected_name}_{model_name}".replace("/", "_").replace(" ", "_")
                    plot_confusion_matrix(
                        cm,
                        list(label_encoder.classes_),
                        f"{task_mode} {selected_name} {model_name}",
                        figs_dir / f"cm_{safe_name}.png",
                    )
                    joblib.dump(fitted, models_dir / f"{safe_name}.joblib")

        for model_name, model in current_models.items():
            log(f"[TRAIN] {task_mode} | {feature_set_name} | {model_name}")
            try:
                fitted, result, report, cm = evaluate_one(
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    label_encoder,
                    model_name,
                    feature_set_name,
                    task_mode,
                )
            except Exception as exc:
                log(f"[ERROR] {task_mode} | {feature_set_name} | {model_name}: {exc}")
                continue

            all_results.append(result)

            for label_name, values in report.items():
                if isinstance(values, dict):
                    rec = {"task": task_mode, "feature_set": feature_set_name, "model": model_name, "label": label_name}
                    rec.update(values)
                    report_records.append(rec)

            safe_name = f"{task_mode}_{feature_set_name}_{model_name}".replace("/", "_").replace(" ", "_")
            plot_confusion_matrix(
                cm,
                list(label_encoder.classes_),
                f"{task_mode} {feature_set_name} {model_name}",
                figs_dir / f"cm_{safe_name}.png",
            )
            joblib.dump(fitted, models_dir / f"{safe_name}.joblib")

    results_df = pd.DataFrame(all_results)
    reports_df = pd.DataFrame(report_records)

    results_path = task_dir / "results_summary.csv"
    reports_path = task_dir / "classification_report_by_class.csv"
    results_df.to_csv(results_path, index=False)
    reports_df.to_csv(reports_path, index=False)

    joblib.dump(label_encoder, models_dir / f"{task_mode}_label_encoder.joblib")

    if len(results_df) > 0:
        sort_cols = ["f1_macro", "balanced_accuracy", "mcc"]
        results_ranked = results_df.sort_values(sort_cols, ascending=False)
        results_ranked.to_csv(task_dir / "results_ranked.csv", index=False)

        best = results_ranked.iloc[0].to_dict()
        with open(task_dir / "best_model.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

        plot_metric_bars(
            results_ranked.assign(combo=results_ranked["model"] + " | " + results_ranked["feature_set"]),
            "f1_macro",
            "combo",
            f"{task_mode} Macro F1 by Model and Ablation",
            figs_dir / "fig_macro_f1_all_experiments.png",
            top_n=30,
        )

        plot_metric_bars(
            results_ranked.assign(combo=results_ranked["model"] + " | " + results_ranked["feature_set"]),
            "balanced_accuracy",
            "combo",
            f"{task_mode} Balanced Accuracy by Model and Ablation",
            figs_dir / "fig_balanced_accuracy_all_experiments.png",
            top_n=30,
        )

        plot_runtime_vs_f1(results_ranked.head(40), figs_dir / "fig_latency_vs_macro_f1.png")

        # Ablation-only comparison: best model within each feature set.
        ablation_best = (
            results_df.sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
            .groupby("feature_set", as_index=False)
            .first()
            .sort_values("f1_macro", ascending=False)
        )
        ablation_best.to_csv(task_dir / "ablation_best_by_feature_set.csv", index=False)
        plot_metric_bars(
            ablation_best,
            "f1_macro",
            "feature_set",
            f"{task_mode} Ablation Study: Best Macro F1 per Feature Set",
            figs_dir / "fig_ablation_best_macro_f1.png",
        )

        # Model-only comparison: best feature set for each model.
        model_best = (
            results_df.sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
            .groupby("model", as_index=False)
            .first()
            .sort_values("f1_macro", ascending=False)
        )
        model_best.to_csv(task_dir / "model_best_comparison.csv", index=False)
        plot_metric_bars(
            model_best,
            "f1_macro",
            "model",
            f"{task_mode} Model Study: Best Macro F1 per Model",
            figs_dir / "fig_model_best_macro_f1.png",
        )

    log(f"[DONE] {task_mode} results saved in {task_dir}")


def write_experiment_design(output_dir: Path) -> None:
    text = """CAVALRY-AI Experimental Design
===============================

Main evaluation perspectives:
1. Detection perspective:
   - Binary classification: Benign vs Attack
   - Multiclass classification: attack-family classification

2. Model perspective:
   - GaussianNB
   - LogisticRegression
   - RandomForest
   - ExtraTrees
   - HistGradientBoosting

3. Ablation perspective:
   - A0_CoreFlow: basic flow statistics only
   - A1_Core_PortProtocol: core flow + destination/source port/protocol
   - A2_Core_PortProtocol_Temporal: adds inter-arrival and idle/active timing
   - A3_AllNumeric_NoCMG_NoRWAR: all available numeric flow features, no proposed modules
   - A4_RWAR_RiskRouting: adds risk-routing engineered features
   - A5_CMG_GraphMemory: adds graph-memory frequency/degree features where IP fields exist
   - A6_Full_CAVALRY_AI: all features including RWAR and CMG
   - A6b_Full_CAVALRY_AI_SelectKBest: full model with mutual-information feature selection
   - A7_TCPFlagsOnly: TCP flag-only diagnostic baseline

4. Metrics:
   - Accuracy
   - Balanced accuracy
   - Macro precision
   - Macro recall
   - Macro F1-score
   - Weighted precision
   - Weighted recall
   - Weighted F1-score
   - Matthews correlation coefficient
   - ROC-AUC macro OvR where available
   - PR-AUC macro where available
   - Log loss where available
   - Training time
   - Inference time
   - Inference milliseconds per sample

Recommended reporting tables:
1. Table I: Dataset distribution after cleaning
2. Table II: Binary classification model comparison
3. Table III: Multiclass classification model comparison
4. Table IV: Ablation study across A0-A7
5. Table V: Runtime and inference latency comparison
6. Table VI: Per-class precision, recall, F1-score

Recommended figures:
1. Label distribution
2. Ablation macro F1 bar chart
3. Model macro F1 bar chart
4. Confusion matrix for best binary model
5. Confusion matrix for best multiclass model
6. Latency vs macro F1 scatter plot
"""
    (output_dir / "experiment_design_ieee_style.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="CAVALRY-AI CSE-CIC-IDS2018 experimental pipeline.")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing CSE-CIC-IDS2018 CSV files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder. Default: data_dir/cavalry_results")
    parser.add_argument("--max_rows_per_file", type=int, default=120000, help="Rows loaded per CSV. Use 0 for full files.")
    parser.add_argument("--max_total_rows", type=int, default=800000, help="Total rows after combining. Use 0 for no cap.")
    parser.add_argument("--mode", type=str, default="both", choices=["binary", "multiclass", "both"], help="Experiment mode.")
    parser.add_argument("--test_size", type=float, default=0.25, help="Test split fraction.")
    parser.add_argument("--min_class_count", type=int, default=500, help="Drop multiclass labels below this count.")
    parser.add_argument("--top_k_features", type=int, default=35, help="SelectKBest feature count for A6b.")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "cavalry_results"
    ensure_dir(output_dir)

    log("CAVALRY-AI EXPERIMENTAL PIPELINE")
    log("=" * 45)
    log(f"Data directory: {data_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Rows per file: {args.max_rows_per_file}")
    log(f"Max total rows: {args.max_total_rows}")
    log(f"Mode: {args.mode}")

    start = time.perf_counter()

    df = load_dataset(data_dir, args.max_rows_per_file)

    # Proposed module feature construction.
    log("[INFO] Building Cyber Memory Graph-inspired features...")
    df = add_graph_features(df)

    log("[INFO] Building Risk-Weighted Agent Routing-inspired features...")
    df = add_risk_features(df)

    # Remove exact duplicate rows after cleaning. This is important for fair evaluation.
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    log(f"[INFO] Dropped duplicate rows: {before - after:,}")

    save_label_distribution(df, output_dir)
    write_experiment_design(output_dir)

    df[["Label", "Source_File"]].to_csv(output_dir / "cleaned_label_source_index.csv", index=False)

    if args.mode in {"binary", "both"}:
        run_experiments(
            df=df,
            output_dir=output_dir,
            task_mode="binary",
            test_size=args.test_size,
            min_class_count=args.min_class_count,
            max_total_rows=args.max_total_rows,
            top_k_features=args.top_k_features,
        )

    if args.mode in {"multiclass", "both"}:
        run_experiments(
            df=df,
            output_dir=output_dir,
            task_mode="multiclass",
            test_size=args.test_size,
            min_class_count=args.min_class_count,
            max_total_rows=args.max_total_rows,
            top_k_features=args.top_k_features,
        )

    elapsed = time.perf_counter() - start

    summary = {
        "elapsed_seconds": elapsed,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "max_rows_per_file": args.max_rows_per_file,
        "max_total_rows": args.max_total_rows,
        "mode": args.mode,
        "rows_after_cleaning": int(len(df)),
        "columns_after_feature_engineering": int(df.shape[1]),
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("\nALL DONE")
    log("=" * 45)
    log(f"Elapsed seconds: {elapsed:.2f}")
    log(f"Results folder: {output_dir}")
    log("Main files to send back:")
    log(f"  {output_dir / 'binary' / 'results_ranked.csv'}")
    log(f"  {output_dir / 'multiclass' / 'results_ranked.csv'}")
    log(f"  {output_dir / 'experiment_design_ieee_style.txt'}")


if __name__ == "__main__":
    main()
