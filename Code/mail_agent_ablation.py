#!/usr/bin/env python3
"""
MAIL Agent Ablation for CAVALRY-AI
==================================

This script runs a controlled Multi-Agent Intelligence Layer ablation.

It creates:
    1. Binary MAIL agent ablation CSV
    2. Multiclass MAIL agent ablation CSV
    3. Combined MAIL agent ablation CSV
    4. LaTeX table for the paper
    5. Saved best models for each ablation run

Important:
    This script does not invent agent scores.
    It maps each MAIL agent to the feature block it contributes and removes
    that block during ablation.

Agent to feature mapping:
    SA  Sentinel Agent    -> RWAR risk score and risk indicators
    HA  Hunter Agent      -> core flow behavior features
    OA  Oracle Agent      -> port and protocol evidence
    SPA Specter Agent     -> TCP flag behavior
    GMA GraphMind Agent   -> CMG graph memory features
    RA  Raptor Agent      -> RWAR retrieval/risk support indicators
    FA  ForensiX Agent    -> temporal and idle/active timing features
    SHA Shield Agent      -> response planning agent, no detection feature removed

Recommended run:
    python mail_agent_ablation.py --data_dir "D:\\other\\CAVALRY-AI\\CSE-CIC-IDS2018" --output_dir "D:\\other\\CAVALRY-AI\\mail_results" --max_rows_per_file 120000 --max_total_rows 800000 --mode both
"""

import argparse
import json
import time
import warnings
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42


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

IDENTIFIER_COLUMNS = {
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Source_File",
}


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_columns(columns: List[str]) -> List[str]:
    return [str(c).strip().replace("\ufeff", "") for c in columns]


def find_csv_files(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])


def read_one_csv(path: Path, max_rows: int) -> pd.DataFrame:
    kwargs = {"low_memory": False, "encoding": "utf-8"}
    if max_rows > 0:
        kwargs["nrows"] = max_rows

    try:
        df = pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        kwargs["encoding"] = "latin1"
        df = pd.read_csv(path, **kwargs)

    df.columns = normalize_columns(df.columns)
    df["Source_File"] = path.name
    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "Label" not in df.columns:
        raise ValueError("The dataset must contain a Label column.")

    df = df.copy()
    df["Label"] = df["Label"].astype(str).str.strip()

    bad_labels = {"", "nan", "NaN", "None", "Label"}
    df = df[~df["Label"].isin(bad_labels)].copy()

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
        "Brute Force -Web": "BruteForce-Web",
        "Brute Force -XSS": "BruteForce-XSS",
    }

    df["Label"] = df["Label"].replace(replacements)
    return df


def load_dataset(data_dir: Path, max_rows_per_file: int) -> pd.DataFrame:
    files = find_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []

    log(f"[INFO] Found {len(files)} CSV files.")

    for file_path in files:
        log(f"[LOAD] {file_path.name}")
        df = read_one_csv(file_path, max_rows_per_file)
        before = len(df)
        df = clean_labels(df)
        after = len(df)
        log(f"       rows loaded: {before:,}")
        log(f"       rows after label cleaning: {after:,}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    log(f"[INFO] Combined rows before duplicate removal: {len(combined):,}")

    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    log(f"[INFO] Duplicate rows removed: {before - after:,}")
    log(f"[INFO] Combined rows after cleaning: {after:,}")

    return combined


def add_cmg_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Src IP" in df.columns:
        src_counts = df["Src IP"].astype(str).value_counts()
        df["CMG_src_ip_frequency"] = df["Src IP"].astype(str).map(src_counts).astype(float)

    if "Dst IP" in df.columns:
        dst_counts = df["Dst IP"].astype(str).value_counts()
        df["CMG_dst_ip_frequency"] = df["Dst IP"].astype(str).map(dst_counts).astype(float)

    if "Src IP" in df.columns and "Dst IP" in df.columns:
        pair = df["Src IP"].astype(str) + "_to_" + df["Dst IP"].astype(str)
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


def add_rwar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_needed = [
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

    for col in numeric_needed:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Flow Byts/s" in df.columns and "Flow Pkts/s" in df.columns:
        df["RWAR_flow_intensity"] = (
            np.log1p(df["Flow Byts/s"].clip(lower=0))
            + np.log1p(df["Flow Pkts/s"].clip(lower=0))
        )

    if "Tot Fwd Pkts" in df.columns and "Tot Bwd Pkts" in df.columns:
        df["RWAR_packet_asymmetry"] = (
            (df["Tot Fwd Pkts"] - df["Tot Bwd Pkts"]).abs()
            / (df["Tot Fwd Pkts"] + df["Tot Bwd Pkts"] + 1.0)
        )

    if "TotLen Fwd Pkts" in df.columns and "TotLen Bwd Pkts" in df.columns:
        df["RWAR_byte_asymmetry"] = (
            (df["TotLen Fwd Pkts"] - df["TotLen Bwd Pkts"]).abs()
            / (df["TotLen Fwd Pkts"] + df["TotLen Bwd Pkts"] + 1.0)
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
        temp = df[risk_cols].replace([np.inf, -np.inf], np.nan)
        temp = temp.fillna(temp.median(numeric_only=True))
        df["RWAR_risk_score"] = temp.rank(pct=True).mean(axis=1)

    return df


def make_binary_label(y: pd.Series) -> np.ndarray:
    return np.where(y.astype(str).str.lower().eq("benign"), "Benign", "Attack")


def get_available(columns: List[str], desired: List[str]) -> List[str]:
    present = set(columns)
    return [c for c in desired if c in present]


def get_all_numeric_features(df: pd.DataFrame) -> List[str]:
    exclude = {"Label", "Target", "Binary_Label", "Source_File"} | IDENTIFIER_COLUMNS
    return [c for c in df.columns if c not in exclude]


def get_agent_feature_blocks(df: pd.DataFrame) -> Dict[str, List[str]]:
    columns = list(df.columns)

    core = get_available(columns, BASE_CORE_FEATURES)
    port_protocol = get_available(columns, PORT_PROTOCOL_FEATURES)
    temporal = get_available(columns, TEMPORAL_FEATURES)
    flags = get_available(columns, TCP_FLAG_FEATURES)
    cmg = [c for c in columns if c.startswith("CMG_")]
    rwar = [c for c in columns if c.startswith("RWAR_")]

    blocks = {
        "SA": rwar,
        "HA": core,
        "OA": port_protocol,
        "SPA": flags,
        "GMA": cmg,
        "RA": rwar,
        "FA": temporal,
        "SHA": [],
    }

    return blocks


def unique_list(values: List[str]) -> List[str]:
    return list(dict.fromkeys(values))


def build_mail_feature_sets(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    all_numeric = get_all_numeric_features(df)
    blocks = get_agent_feature_blocks(df)

    full_features = unique_list(all_numeric)

    settings = {
        "MAIL_Full": {
            "active_agents": "SA, HA, OA, SPA, GMA, RA, FA, SHA",
            "removed_agent": "None",
            "feature_description": "Full CAVALRY-AI representation with flow, port, protocol, timing, TCP flags, RWAR, and CMG features",
            "features": full_features,
        }
    }

    for agent, removed_features in blocks.items():
        setting_name = f"MAIL_without_{agent}"

        if agent == "SHA":
            remaining = full_features
            description = "Shield Agent removed. Detection feature space remains unchanged because SHA produces response recommendations after classification."
        else:
            removed_set = set(removed_features)
            remaining = [c for c in full_features if c not in removed_set]
            description = f"Full CAVALRY-AI representation with the {agent} feature contribution removed"

        active_agents = [a for a in ["SA", "HA", "OA", "SPA", "GMA", "RA", "FA", "SHA"] if a != agent]

        settings[setting_name] = {
            "active_agents": ", ".join(active_agents),
            "removed_agent": agent,
            "feature_description": description,
            "features": remaining,
        }

    settings["MAIL_Low_Risk_Route"] = {
        "active_agents": "SA, HA, OA",
        "removed_agent": "SPA, GMA, RA, FA, SHA",
        "feature_description": "Lightweight route using RWAR risk indicators, core flow behavior, port, and protocol evidence",
        "features": unique_list(blocks["SA"] + blocks["HA"] + blocks["OA"]),
    }

    settings["MAIL_Medium_Risk_Route"] = {
        "active_agents": "SA, HA, OA, GMA, RA",
        "removed_agent": "SPA, FA, SHA",
        "feature_description": "Medium route using risk indicators, core flow behavior, port and protocol evidence, CMG graph context, and RWAR support",
        "features": unique_list(blocks["SA"] + blocks["HA"] + blocks["OA"] + blocks["GMA"] + blocks["RA"]),
    }

    return settings


def coerce_numeric(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df[features].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    valid_cols = [c for c in X.columns if not X[c].isna().all()]
    return X[valid_cols]


def build_best_model(task: str):
    if task == "binary":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=160,
                random_state=RANDOM_STATE,
            )),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=160,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def compute_auc_and_loss(y_true: np.ndarray, scores, n_classes: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    roc_value = None
    pr_value = None
    loss_value = None

    if scores is None:
        return roc_value, pr_value, loss_value

    try:
        if n_classes == 2:
            if len(np.asarray(scores).shape) == 2:
                positive_scores = scores[:, 1]
                loss_value = float(log_loss(y_true, scores))
            else:
                positive_scores = scores

            roc_value = float(roc_auc_score(y_true, positive_scores))
            pr_value = float(average_precision_score(y_true, positive_scores))

        else:
            if len(np.asarray(scores).shape) == 2 and scores.shape[1] == n_classes:
                y_bin = label_binarize(y_true, classes=np.arange(n_classes))
                roc_value = float(roc_auc_score(y_bin, scores, average="macro", multi_class="ovr"))
                pr_value = float(average_precision_score(y_bin, scores, average="macro"))
                loss_value = float(log_loss(y_true, scores, labels=np.arange(n_classes)))

    except Exception:
        pass

    return roc_value, pr_value, loss_value


def get_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X_test)
        except Exception:
            return None
    return None


def evaluate_model(
    task: str,
    setting_name: str,
    setting_info: Dict[str, object],
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    label_encoder: LabelEncoder,
    models_dir: Path,
) -> Tuple[Dict[str, object], pd.DataFrame, np.ndarray]:
    fitted = clone(model)

    train_start = time.perf_counter()
    fitted.fit(X_train, y_train)
    train_seconds = time.perf_counter() - train_start

    pred_start = time.perf_counter()
    y_pred = fitted.predict(X_test)
    inference_seconds = time.perf_counter() - pred_start

    scores = get_scores(fitted, X_test)
    n_classes = len(label_encoder.classes_)
    roc_value, pr_value, loss_value = compute_auc_and_loss(y_test, scores, n_classes)

    result = {
        "task": task,
        "mail_setting": setting_name,
        "active_agents": setting_info["active_agents"],
        "removed_agent": setting_info["removed_agent"],
        "feature_description": setting_info["feature_description"],
        "model": fitted.named_steps["model"].__class__.__name__,
        "feature_count": int(X_train.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "roc_auc_macro_ovr": roc_value,
        "pr_auc_macro": pr_value,
        "log_loss": loss_value,
        "train_seconds": float(train_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_ms_per_sample": float((inference_seconds / max(1, X_test.shape[0])) * 1000),
    }

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(n_classes),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []
    for label_name, values in report_dict.items():
        if isinstance(values, dict):
            row = {
                "task": task,
                "mail_setting": setting_name,
                "label": label_name,
            }
            row.update(values)
            report_rows.append(row)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))

    model_file = models_dir / f"{task}_{setting_name}_{result['model']}.joblib"
    joblib.dump(fitted, model_file)

    return result, pd.DataFrame(report_rows), cm


def run_task(
    df: pd.DataFrame,
    task: str,
    output_dir: Path,
    test_size: float,
    max_total_rows: int,
    min_class_count: int,
) -> pd.DataFrame:
    task_dir = output_dir / task
    models_dir = task_dir / "models"
    reports_dir = task_dir / "reports"
    matrices_dir = task_dir / "confusion_matrices"

    ensure_dir(task_dir)
    ensure_dir(models_dir)
    ensure_dir(reports_dir)
    ensure_dir(matrices_dir)

    data = df.copy()

    if task == "binary":
        data["Target"] = make_binary_label(data["Label"])
    elif task == "multiclass":
        data["Target"] = data["Label"]
        counts = data["Target"].value_counts()
        keep = counts[counts >= min_class_count].index
        data = data[data["Target"].isin(keep)].copy()
    else:
        raise ValueError("task must be binary or multiclass")

    if max_total_rows > 0 and len(data) > max_total_rows:
        data, _ = train_test_split(
            data,
            train_size=max_total_rows,
            random_state=RANDOM_STATE,
            stratify=data["Target"],
        )
        data = data.copy()

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(data["Target"].astype(str))

    settings = build_mail_feature_sets(data)
    model = build_best_model(task)

    all_results = []
    all_reports = []

    for setting_name, setting_info in settings.items():
        features = setting_info["features"]

        if not features:
            log(f"[SKIP] {task} | {setting_name}: no features available")
            continue

        X_all = coerce_numeric(data, features)

        if X_all.shape[1] == 0:
            log(f"[SKIP] {task} | {setting_name}: no valid numeric features")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y_all,
        )

        log(f"[TRAIN] {task} | {setting_name} | features={X_train.shape[1]}")

        result, report_df, cm = evaluate_model(
            task=task,
            setting_name=setting_name,
            setting_info=setting_info,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=label_encoder,
            models_dir=models_dir,
        )

        all_results.append(result)
        all_reports.append(report_df)

        pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
            matrices_dir / f"{task}_{setting_name}_confusion_matrix.csv"
        )

        gc.collect()

    results_df = pd.DataFrame(all_results)

    if all_reports:
        reports_df = pd.concat(all_reports, ignore_index=True)
        reports_df.to_csv(reports_dir / f"{task}_mail_classification_reports.csv", index=False)

    results_df.to_csv(task_dir / f"{task}_mail_agent_ablation_results.csv", index=False)

    ranked = results_df.sort_values(["f1_macro", "balanced_accuracy", "mcc"], ascending=False)
    ranked.to_csv(task_dir / f"{task}_mail_agent_ablation_ranked.csv", index=False)

    joblib.dump(label_encoder, models_dir / f"{task}_label_encoder.joblib")

    return results_df


def pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.4f}"


def dec(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def make_latex_table(combined: pd.DataFrame, output_path: Path) -> None:
    keep_order = [
        "MAIL_Full",
        "MAIL_without_SA",
        "MAIL_without_HA",
        "MAIL_without_OA",
        "MAIL_without_SPA",
        "MAIL_without_GMA",
        "MAIL_without_RA",
        "MAIL_without_FA",
        "MAIL_without_SHA",
        "MAIL_Low_Risk_Route",
        "MAIL_Medium_Risk_Route",
    ]

    rows = []
    for setting in keep_order:
        binary = combined[(combined["task"] == "binary") & (combined["mail_setting"] == setting)]
        multi = combined[(combined["task"] == "multiclass") & (combined["mail_setting"] == setting)]

        if binary.empty and multi.empty:
            continue

        ref = binary.iloc[0] if not binary.empty else multi.iloc[0]

        b = binary.iloc[0] if not binary.empty else None
        m = multi.iloc[0] if not multi.empty else None

        rows.append(
            [
                setting.replace("_", "\\_"),
                str(ref["active_agents"]).replace("_", "\\_"),
                str(ref["removed_agent"]).replace("_", "\\_"),
                int(ref["feature_count"]) if not pd.isna(ref["feature_count"]) else "N/A",
                pct(b["accuracy"]) if b is not None else "N/A",
                pct(b["f1_macro"]) if b is not None else "N/A",
                dec(b["mcc"]) if b is not None else "N/A",
                pct(m["accuracy"]) if m is not None else "N/A",
                pct(m["f1_macro"]) if m is not None else "N/A",
                dec(m["mcc"]) if m is not None else "N/A",
                f"{ref['inference_ms_per_sample']:.4f}",
            ]
        )

    lines = []
    lines.append("\\begin{table*}[!t]")
    lines.append("\\centering")
    lines.append("\\caption{MAIL agent ablation results for binary and multiclass CAVALRY-AI detection. The active agents column lists the agents used in each run. The removed agent column identifies the disabled agent or route components. Feature count reports the numeric input features retained after that ablation. Binary and multiclass scores are reported using accuracy, macro F1 score, and MCC.}")
    lines.append("\\label{tab:mail_agent_ablation_results}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\setlength{\\tabcolsep}{3.2pt}")
    lines.append("\\scriptsize")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{p{2.6cm} p{4.4cm} p{2.4cm} r r r r r r r r}")
    lines.append("\\hline")
    lines.append("\\textbf{MAIL Setting} & \\textbf{Active Agents} & \\textbf{Removed Agent} & \\textbf{Feature Count} & \\textbf{Binary Acc.} & \\textbf{Binary F1} & \\textbf{Binary MCC} & \\textbf{Multi Acc.} & \\textbf{Multi F1} & \\textbf{Multi MCC} & \\textbf{Latency} \\\\")
    lines.append("\\hline")

    for row in rows:
        lines.append(" & ".join(map(str, row)) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="CAVALRY-AI MAIL agent ablation runner.")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing CSE-CIC-IDS2018 CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder where results and models will be saved.")
    parser.add_argument("--max_rows_per_file", type=int, default=120000, help="Rows loaded per CSV file. Use 0 for full files.")
    parser.add_argument("--max_total_rows", type=int, default=800000, help="Maximum rows used after cleaning. Use 0 for no cap.")
    parser.add_argument("--mode", choices=["binary", "multiclass", "both"], default="both", help="Which task to run.")
    parser.add_argument("--test_size", type=float, default=0.25, help="Test split fraction.")
    parser.add_argument("--min_class_count", type=int, default=500, help="Minimum class count for multiclass runs.")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)

    log("CAVALRY-AI MAIL AGENT ABLATION")
    log("=" * 45)
    log(f"Data directory: {data_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Rows per file: {args.max_rows_per_file}")
    log(f"Max total rows: {args.max_total_rows}")
    log(f"Mode: {args.mode}")

    start = time.perf_counter()

    df = load_dataset(data_dir, args.max_rows_per_file)

    log("[INFO] Building CMG graph memory features...")
    df = add_cmg_features(df)

    log("[INFO] Building RWAR risk routing features...")
    df = add_rwar_features(df)

    results = []

    if args.mode in {"binary", "both"}:
        binary_results = run_task(
            df=df,
            task="binary",
            output_dir=output_dir,
            test_size=args.test_size,
            max_total_rows=args.max_total_rows,
            min_class_count=args.min_class_count,
        )
        results.append(binary_results)

    if args.mode in {"multiclass", "both"}:
        multiclass_results = run_task(
            df=df,
            task="multiclass",
            output_dir=output_dir,
            test_size=args.test_size,
            max_total_rows=args.max_total_rows,
            min_class_count=args.min_class_count,
        )
        results.append(multiclass_results)

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output_dir / "mail_agent_ablation_combined.csv", index=False)

    make_latex_table(
        combined=combined,
        output_path=output_dir / "mail_agent_ablation_table.tex",
    )

    summary = {
        "elapsed_seconds": time.perf_counter() - start,
        "rows_after_cleaning_and_feature_engineering": int(len(df)),
        "columns_after_feature_engineering": int(df.shape[1]),
        "mode": args.mode,
        "output_dir": str(output_dir),
    }

    with open(output_dir / "mail_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("\nDONE")
    log("=" * 45)
    log(f"Combined results: {output_dir / 'mail_agent_ablation_combined.csv'}")
    log(f"LaTeX table: {output_dir / 'mail_agent_ablation_table.tex'}")
    log(f"Saved models: {output_dir / 'binary' / 'models'} and {output_dir / 'multiclass' / 'models'}")


if __name__ == "__main__":
    main()