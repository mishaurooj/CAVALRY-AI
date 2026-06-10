import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CAVALRY-AI Paper Figure Generator
# Full readable version with all results shown
# ------------------------------------------------------------
# Put these files in the SAME folder as this script:
#   results_ranked_binary.csv
#   results_ranked_multi.csv
#
# Run:
#   conda activate cavalry-ai
#   cd /d D:\other\CAVALRY-AI\Code
#   python plot_paper_figures_all_readable.py
#
# Output:
#   paper_figures\fig_binary_multiclass_2x3_all_readable.png
#   paper_figures\fig_binary_multiclass_2x3_all_readable.pdf
#   paper_figures\fig_binary_multiclass_2x3_all_readable.tiff
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BINARY_RESULTS = os.path.join(SCRIPT_DIR, "results_ranked_binary.csv")
MULTI_RESULTS = os.path.join(SCRIPT_DIR, "results_ranked_multi.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_MAP = {
    "GaussianNB": "PSE",
    "LogisticRegression": "LTE",
    "RandomForest": "EDE",
    "ExtraTrees": "CFE",
    "HistGradientBoosting": "ABE",
}

FEATURE_MAP = {
    "A0_CoreFlow": "A0-Core",
    "A1_Core_PortProtocol": "A1-Port",
    "A2_Core_PortProtocol_Temporal": "A2-Time",
    "A3_AllNumeric_NoCMG_NoRWAR": "A3-Flow",
    "A4_RWAR_RiskRouting": "A4-RWAR",
    "A5_CMG_GraphMemory": "A5-CMG",
    "A6_Full_CAVALRY_AI": "A6-Full",
    "A6b_Full_CAVALRY_AI_SelectKBest_35": "A6b-K35",
    "A7_TCPFlagsOnly": "A7-Flags",
}

CFG_ORDER = [
    "A0-Core",
    "A1-Port",
    "A2-Time",
    "A3-Flow",
    "A4-RWAR",
    "A5-CMG",
    "A6-Full",
    "A6b-K35",
    "A7-Flags",
]

COLORS = [
    "#0B6E4F",
    "#168AAD",
    "#34A0A4",
    "#52B788",
    "#76C893",
    "#99D98C",
    "#F4A261",
    "#E76F51",
    "#457B9D",
    "#1D3557",
]

ENGINE_COLORS = {
    "PSE": "#7D8597",
    "LTE": "#F4A261",
    "EDE": "#2A9D8F",
    "CFE": "#0B6E4F",
    "ABE": "#457B9D",
}


def short_feature_name(x):
    x = str(x)

    if x in FEATURE_MAP:
        return FEATURE_MAP[x]

    for key, value in FEATURE_MAP.items():
        if key in x:
            return value

    match = re.search(r"(A\d+b?|A\d)", x)
    return match.group(1) if match else x[:12]


def short_model_name(x):
    x = str(x)

    for key, value in MODEL_MAP.items():
        if key in x:
            return value

    return x[:8]


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "\nMissing file:\n"
            + path
            + "\n\nPut results_ranked_binary.csv and results_ranked_multi.csv "
            + "in the same folder as this script."
        )


def load_results(csv_path):
    check_file(csv_path)

    df = pd.read_csv(csv_path)

    required_cols = [
        "model",
        "feature_set",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "inference_ms_per_sample",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "\nMissing required columns in:\n"
            + csv_path
            + "\n\nMissing columns:\n"
            + str(missing_cols)
            + "\n\nAvailable columns:\n"
            + str(df.columns.tolist())
        )

    for col in ["accuracy", "balanced_accuracy", "f1_macro", "inference_ms_per_sample"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["accuracy", "balanced_accuracy", "f1_macro", "inference_ms_per_sample"]
    )

    df["engine"] = df["model"].astype(str).apply(short_model_name)
    df["cfg"] = df["feature_set"].astype(str).apply(short_feature_name)
    df["label2"] = df["engine"] + ":" + df["cfg"]

    return df


def build_ablation_from_results(df):
    ablation = (
        df.sort_values("f1_macro", ascending=False)
        .drop_duplicates(subset=["cfg"])
        .copy()
    )

    ablation["cfg_order"] = ablation["cfg"].apply(
        lambda x: CFG_ORDER.index(x) if x in CFG_ORDER else 999
    )

    return ablation.sort_values("cfg_order").drop(columns=["cfg_order"])


def repeat_colors(n):
    return [COLORS[i % len(COLORS)] for i in range(n)]


def style_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.30)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def add_engine_legend(ax, data, loc="lower right"):
    handles = []
    labels = []

    for engine, color in ENGINE_COLORS.items():
        if engine in set(data["engine"]):
            handle = ax.scatter(
                [],
                [],
                s=48,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            handles.append(handle)
            labels.append(engine)

    ax.legend(
        handles,
        labels,
        fontsize=7,
        frameon=True,
        loc=loc,
        borderpad=0.4,
        labelspacing=0.35,
        handletextpad=0.4,
    )


def plot_macro_f1_bar(ax, ablation_df, title):
    data = ablation_df.copy()
    x = np.arange(len(data))

    ax.bar(
        x,
        data["f1_macro"] * 100,
        color=repeat_colors(len(data)),
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel("Macro F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(data["cfg"], rotation=35, ha="right", fontsize=9)

    ymin = max(0, (data["f1_macro"].min() * 100) - 5)
    ax.set_ylim(ymin, 101)

    style_axis(ax, "y")


def plot_all_balanced_accuracy_barh(ax, results_df, title):
    # Shows all results. Extra y spacing prevents labels from touching.
    data = results_df.sort_values("balanced_accuracy", ascending=True).copy()

    row_gap = 1.35
    y = np.arange(len(data)) * row_gap

    values = data["balanced_accuracy"] * 100
    colors = [ENGINE_COLORS.get(engine, "#2A9D8F") for engine in data["engine"]]

    ax.barh(
        y,
        values,
        height=0.78,
        color=colors,
        edgecolor="black",
        linewidth=0.55,
        alpha=0.95,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data["label2"], fontsize=5.8)

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Balanced Accuracy (%)")

    xmin = max(0, values.min() - 4)
    xmax = min(101, values.max() + 1.5)
    ax.set_xlim(xmin, xmax)

    ax.set_ylim(y.min() - row_gap, y.max() + row_gap)

    style_axis(ax, "x")
    add_engine_legend(ax, data, loc="lower right")


def plot_latency_scatter_all(ax, results_df, title):
    # Shows all results, labels only top five to stop overlap.
    data = results_df.copy()

    acc_min = data["accuracy"].min()
    acc_max = data["accuracy"].max()

    sizes = 55 + ((data["accuracy"] - acc_min) / (acc_max - acc_min + 1e-9)) * 135

    for engine in sorted(data["engine"].unique()):
        part = data[data["engine"] == engine]
        part_sizes = sizes.loc[part.index]

        ax.scatter(
            part["inference_ms_per_sample"],
            part["f1_macro"] * 100,
            s=part_sizes,
            color=ENGINE_COLORS.get(engine, "#2A9D8F"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.82,
            label=engine,
        )

    top5 = data.sort_values("f1_macro", ascending=False).head(5)
    offsets = [(8, 8), (8, -14), (10, 20), (-54, 8), (-54, -14)]

    for i, (_, row) in enumerate(top5.iterrows()):
        ax.annotate(
            row["label2"],
            (row["inference_ms_per_sample"], row["f1_macro"] * 100),
            fontsize=7.2,
            xytext=offsets[i % len(offsets)],
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=0.7),
        )

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Latency (ms/sample)")
    ax.set_ylabel("Macro F1 (%)")

    ax.legend(
        fontsize=7,
        frameon=True,
        loc="lower right",
        borderpad=0.4,
        labelspacing=0.35,
        handletextpad=0.4,
    )

    style_axis(ax, "both")


def main():
    binary_results = load_results(BINARY_RESULTS)
    multi_results = load_results(MULTI_RESULTS)

    binary_ablation = build_ablation_from_results(binary_results)
    multi_ablation = build_ablation_from_results(multi_results)

    print("Binary CSV:", BINARY_RESULTS)
    print("Multiclass CSV:", MULTI_RESULTS)
    print("Binary rows loaded:", len(binary_results))
    print("Multiclass rows loaded:", len(multi_results))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # High figure height and wider middle column keep all results readable.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(23, 15),
        dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.45, 1.12]},
    )

    fig.patch.set_facecolor("white")

    plot_macro_f1_bar(
        axes[0, 0],
        binary_ablation,
        "(a) Binary Macro F1 by Configuration"
    )

    plot_all_balanced_accuracy_barh(
        axes[0, 1],
        binary_results,
        "(b) Binary Balanced Accuracy"
    )

    plot_latency_scatter_all(
        axes[0, 2],
        binary_results,
        "(c) Binary Accuracy-Latency Tradeoff"
    )

    plot_macro_f1_bar(
        axes[1, 0],
        multi_ablation,
        "(d) Multiclass Macro F1 by Configuration"
    )

    plot_all_balanced_accuracy_barh(
        axes[1, 1],
        multi_results,
        "(e) Multiclass Balanced Accuracy"
    )

    plot_latency_scatter_all(
        axes[1, 2],
        multi_results,
        "(f) Multiclass Accuracy-Latency Tradeoff"
    )

    plt.tight_layout(pad=2.4, w_pad=2.6, h_pad=3.0)

    png_path = os.path.join(OUT_DIR, "fig_binary_multiclass_2x3_all_readable.png")
    pdf_path = os.path.join(OUT_DIR, "fig_binary_multiclass_2x3_all_readable.pdf")
    tiff_path = os.path.join(OUT_DIR, "fig_binary_multiclass_2x3_all_readable.tiff")

    plt.savefig(png_path, dpi=900, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(
        tiff_path,
        dpi=900,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    plt.close()

    print("\nSaved:")
    print(png_path)
    print(pdf_path)
    print(tiff_path)


if __name__ == "__main__":
    main()
