import os
import csv
import json
import argparse
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from sklearn.metrics import roc_auc_score


# =========================================================
# Basic IO
# =========================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# Name normalization
# =========================================================

def canonical_name(x: str) -> str:
    x = str(x).replace("\\", "/")
    return os.path.basename(x)


def normalize_fault_value(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if v != 0 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "fault", "faulty", "noisy", "noise", "bad", "error"}:
            return 1
        if s in {"0", "false", "clean", "normal", "good", "correct"}:
            return 0
    raise ValueError(f"Unsupported fault flag: {v}")


def load_name2isfault(path: str) -> Dict[str, int]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"name2isfault must be dict, got {type(data)}")

    out = {}
    for k, v in data.items():
        out[canonical_name(k)] = normalize_fault_value(v)
    return out


# =========================================================
# Ranking parsers
# =========================================================

def parse_adversarial_sorted_score_list(path: str) -> Tuple[List[str], np.ndarray]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Adversarial file must be list, got {type(data)}")

    names = []
    scores = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(f"Unexpected Adversarial item: {item}")
        names.append(canonical_name(item[0]))
        scores.append(float(item[1]))

    return names, np.asarray(scores, dtype=float)


def parse_regular_method(results_list_path: str, sorted_score_list_path: str) -> Tuple[List[str], np.ndarray]:
    names = load_json(results_list_path)
    scores = load_json(sorted_score_list_path)

    if not isinstance(names, list):
        raise ValueError(f"results_list must be list, got {type(names)}")
    if not isinstance(scores, list):
        raise ValueError(f"sorted_score_list must be list, got {type(scores)}")
    if len(names) != len(scores):
        raise ValueError(
            f"Length mismatch: {results_list_path} has {len(names)}, "
            f"{sorted_score_list_path} has {len(scores)}"
        )

    names = [canonical_name(x) for x in names]
    scores = np.asarray([float(x) for x in scores], dtype=float)
    return names, scores


# =========================================================
# Metrics
# =========================================================

def localization_curve(ranked_names: List[str], name2isfault: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    ranked_faults = np.asarray([name2isfault.get(canonical_name(n), 0) for n in ranked_names], dtype=int)

    n = len(ranked_faults)
    total_faults = int(sum(name2isfault.values()))

    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    x = np.arange(1, n + 1, dtype=float) / n
    if total_faults == 0:
        y = np.zeros(n, dtype=float)
    else:
        y = np.cumsum(ranked_faults) / total_faults
    return x, y


def theory_best_curve(name2isfault: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(name2isfault)
    total_faults = int(sum(name2isfault.values()))
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.arange(1, n + 1, dtype=float) / n
    if total_faults == 0:
        return x, np.zeros(n, dtype=float)
    y = np.minimum(np.arange(1, n + 1, dtype=float) / total_faults, 1.0)
    return x, y


def random_curve(name2isfault: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(name2isfault)
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.arange(1, n + 1, dtype=float) / n
    y = x.copy()
    return x, y


def POBL(ranked_names: List[str], name2isfault: Dict[str, int], ratio: float = 0.1) -> float:
    x, y = localization_curve(ranked_names, name2isfault)
    if len(x) == 0 or len(y) == 0:
        return 0.0
    idx = np.searchsorted(x, ratio, side="left")
    idx = min(idx, len(y) - 1)
    return float(y[idx])


def APFD(ranked_names: List[str], name2isfault: Dict[str, int]) -> float:
    flags = np.asarray([name2isfault.get(canonical_name(n), 0) for n in ranked_names], dtype=int)
    n = len(flags)
    fault_pos = np.where(flags == 1)[0] + 1
    m = len(fault_pos)
    if n == 0 or m == 0:
        return 0.0
    return 1.0 - fault_pos.sum() / (n * m) + 1.0 / (2 * n)


def curve_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    x0 = np.concatenate(([0.0], x))
    y0 = np.concatenate(([0.0], y))
    return float(np.trapz(y0, x0))


def RAUC(ranked_names: List[str], name2isfault: Dict[str, int]) -> float:
    x, y = localization_curve(ranked_names, name2isfault)
    xb, yb = theory_best_curve(name2isfault)
    auc_m = curve_auc(x, y)
    auc_b = curve_auc(xb, yb)
    return 0.0 if auc_b <= 0 else auc_m / auc_b


def ROC_AUC(ranked_names: List[str], name2isfault: Dict[str, int], ranked_scores: np.ndarray) -> float:
    y_true = np.asarray([name2isfault.get(canonical_name(n), 0) for n in ranked_names], dtype=int)
    y_score = np.asarray(ranked_scores, dtype=float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


# =========================================================
# Method loader
# =========================================================

def load_method_ranking(results_dir: str, method: str) -> Tuple[List[str], np.ndarray]:
    if method == "Adversarial":
        path = os.path.join(results_dir, "Adversarial_sorted_score_list.json")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return parse_adversarial_sorted_score_list(path)

    if method == "DFauLo":
        results_list_path = os.path.join(results_dir, "DFauLo_Manual_results_list.json")
        sorted_score_list_path = os.path.join(results_dir, "DFauLo_Manual_sorted_score_list.json")
        if not os.path.exists(results_list_path):
            raise FileNotFoundError(results_list_path)
        if not os.path.exists(sorted_score_list_path):
            raise FileNotFoundError(sorted_score_list_path)
        return parse_regular_method(results_list_path, sorted_score_list_path)

    file_prefix_map = {
        "DeepGini": "DeepGini",
        "CleanLab": "CleanLab",
        "DIF": "DIF",
        "SimiFeat": "SimiFeat",
        "NCNV": "NCNV",
    }

    if method not in file_prefix_map:
        raise ValueError(f"Unsupported method: {method}")

    prefix = file_prefix_map[method]
    results_list_path = os.path.join(results_dir, f"{prefix}_results_list.json")
    sorted_score_list_path = os.path.join(results_dir, f"{prefix}_sorted_score_list.json")

    if not os.path.exists(results_list_path):
        raise FileNotFoundError(results_list_path)
    if not os.path.exists(sorted_score_list_path):
        raise FileNotFoundError(sorted_score_list_path)

    return parse_regular_method(results_list_path, sorted_score_list_path)


# =========================================================
# Plot config
# =========================================================

METHOD_ORDER = [
    "Theory Best",
    "DeepGini",
    "CleanLab",
    "DIF",
    "SimiFeat",
    "NCNV",
    "Random",
    "DFauLo",
    "ADDFL",
]

METHOD_PLOT_STYLE = {
    "Theory Best": {"linewidth": 1.2, "color": "tab:blue"},
    "DeepGini": {"linewidth": 1.2, "color": "tab:brown"},
    "CleanLab": {"linewidth": 1.2, "color": "tab:purple"},
    "DIF": {"linewidth": 1.2, "color": "tab:orange"},
    "SimiFeat": {"linewidth": 1.2, "color": "tab:pink"},
    "NCNV": {"linewidth": 1.2, "color": "tab:cyan"},
    "Random": {"linewidth": 1.0, "color": "gray"},
    "DFauLo": {"linewidth": 1.2, "color": "darkgreen"},
    "ADDFL": {"linewidth": 1.4, "color": "red"},
}

PANEL_INFO = [
    ("RandomLabelNoise", "(a) Random Label Noise"),
    ("SpecificLabelNoise", "(b) Specific Label Noise"),
    ("RandomDataNoise", "(c) Random Data Noise"),
    ("SpecificDataNoise", "(d) Specific Data Noise"),
]


# =========================================================
# Core panel processing
# =========================================================

def process_one_panel(dataset_root: str, model_name: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Dict[str, float]]]:
    name2isfault_path = os.path.join(dataset_root, "train", "name2isfault.json")
    results_dir = os.path.join(dataset_root, "results", model_name)

    if not os.path.exists(name2isfault_path):
        raise FileNotFoundError(name2isfault_path)
    if not os.path.exists(results_dir):
        raise FileNotFoundError(results_dir)

    name2isfault = load_name2isfault(name2isfault_path)

    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    xb, yb = theory_best_curve(name2isfault)
    curves["Theory Best"] = (xb, yb)

    xr, yr = random_curve(name2isfault)
    curves["Random"] = (xr, yr)

    methods = ["DeepGini", "CleanLab", "DIF", "SimiFeat", "NCNV", "DFauLo", "Adversarial"]

    for method in methods:
        try:
            ranked_names, ranked_scores = load_method_ranking(results_dir, method)
        except FileNotFoundError as e:
            print(f"[Warning] {method} missing in {results_dir}: {e}")
            continue

        x, y = localization_curve(ranked_names, name2isfault)

        if method == "Adversarial":
            curves["ADDFL"] = (x, y)
            metrics["ADDFL"] = {
                "PoBL(10%)": POBL(ranked_names, name2isfault, 0.1),
                "APFD": APFD(ranked_names, name2isfault),
                "RAUC": RAUC(ranked_names, name2isfault),
                "ROC_AUC": ROC_AUC(ranked_names, name2isfault, ranked_scores),
            }
        else:
            curves[method] = (x, y)
            metrics[method] = {
                "PoBL(10%)": POBL(ranked_names, name2isfault, 0.1),
                "APFD": APFD(ranked_names, name2isfault),
                "RAUC": RAUC(ranked_names, name2isfault),
                "ROC_AUC": ROC_AUC(ranked_names, name2isfault, ranked_scores),
            }

    return curves, metrics


def plot_panel(ax, curves: Dict[str, Tuple[np.ndarray, np.ndarray]], missing_message: str = "") -> None:
    if missing_message:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.text(0.5, 0.5, missing_message, ha="center", va="center", fontsize=12)
        ax.set_xlabel("Percentage of test case executed", fontsize=9, fontstyle="italic")
        ax.set_ylabel("Percentage of fault detected", fontsize=8, fontstyle="italic")
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        return

    for method in METHOD_ORDER:
        if method not in curves:
            continue
        x, y = curves[method]
        style = METHOD_PLOT_STYLE.get(method, {})
        ax.plot(x, y, label=method, **style)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Percentage of test case executed", fontsize=9, fontstyle="italic")
    ax.set_ylabel("Percentage of fault detected", fontsize=8, fontstyle="italic")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower right", frameon=True, fancybox=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# =========================================================
# Combo discovery
# =========================================================

def discover_all_combinations(root: str) -> List[Tuple[str, str]]:
    combos: Set[Tuple[str, str]] = set()

    for noise_folder, _ in PANEL_INFO:
        noise_root = os.path.join(root, noise_folder)
        if not os.path.isdir(noise_root):
            continue

        for dataset in sorted(os.listdir(noise_root)):
            dataset_root = os.path.join(noise_root, dataset)
            results_root = os.path.join(dataset_root, "results")
            if not os.path.isdir(results_root):
                continue

            for model in sorted(os.listdir(results_root)):
                model_root = os.path.join(results_root, model)
                if os.path.isdir(model_root):
                    combos.add((dataset, model))

    return sorted(combos, key=lambda x: (x[0], x[1]))


def filter_combinations(
    combos: List[Tuple[str, str]],
    datasets: List[str],
    models: List[str],
) -> List[Tuple[str, str]]:
    dataset_set = set(datasets) if datasets else None
    model_set = set(models) if models else None

    out = []
    for dataset, model in combos:
        if dataset_set is not None and dataset not in dataset_set:
            continue
        if model_set is not None and model not in model_set:
            continue
        out.append((dataset, model))
    return out


# =========================================================
# Saving metrics
# =========================================================

def save_global_metrics_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "dataset", "model", "noise_folder", "panel_title", "method",
        "PoBL(10%)", "APFD", "RAUC", "ROC_AUC"
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(dataset: str, model: str, all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print("\n" + "=" * 120)
    print(f"Metrics Summary | {dataset} | {model}")
    print("=" * 120)
    for panel_title, metrics in all_metrics.items():
        print(f"\n[{panel_title}]")
        if not metrics:
            print("No metrics available.")
            continue
        header = f"{'Method':<15} {'PoBL(10%)':>12} {'APFD':>12} {'RAUC':>12} {'ROC_AUC':>12}"
        print(header)
        print("-" * len(header))
        for method, vals in metrics.items():
            print(
                f"{method:<15} "
                f"{vals['PoBL(10%)']:>12.4f} "
                f"{vals['APFD']:>12.4f} "
                f"{vals['RAUC']:>12.4f} "
                f"{vals['ROC_AUC']:>12.4f}"
            )


# =========================================================
# Figure generation
# =========================================================

def generate_one_figure(root: str, dataset: str, model: str, output_dir: str) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Any]]:
    ensure_dir(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    fig.patch.set_facecolor("white")

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    status: Dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "panels": {},
    }

    for ax, (noise_folder, panel_title) in zip(axes, PANEL_INFO):
        dataset_root = os.path.join(root, noise_folder, dataset)
        try:
            curves, metrics = process_one_panel(dataset_root, model)
            plot_panel(ax, curves=curves)
            all_metrics[panel_title] = metrics
            status["panels"][noise_folder] = {"ok": True, "message": ""}
        except Exception as e:
            plot_panel(ax, curves={}, missing_message="Missing data")
            all_metrics[panel_title] = {}
            status["panels"][noise_folder] = {"ok": False, "message": str(e)}
            print(f"[Warning] Skip panel {noise_folder} | {dataset} | {model}: {e}")

        ax.text(
            0.5, -0.16, panel_title,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=16, fontweight="bold",
            clip_on=False
        )

    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10, wspace=0.28, hspace=0.35)

    base_name = f"{dataset}_{model}_fault_localization"
    png_path = os.path.join(output_dir, base_name + ".png")
    pdf_path = os.path.join(output_dir, base_name + ".pdf")
    json_path = os.path.join(output_dir, base_name + "_metrics.json")

    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    save_json(all_metrics, json_path)

    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")
    print(f"[Saved] {json_path}")
    print_metrics(dataset, model, all_metrics)

    status["png"] = png_path
    status["pdf"] = pdf_path
    status["metrics_json"] = json_path
    return all_metrics, status


# =========================================================
# Main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root dataset directory, e.g. /ywt/dataset")
    parser.add_argument("--output_dir", type=str, default="./plots_all", help="Output directory")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset filter, e.g. MNIST KMNIST")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model filter, e.g. LeNet1 LeNet5 ResNet18 VGG")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    combos = discover_all_combinations(args.root)
    combos = filter_combinations(combos, args.datasets, args.models)

    if not combos:
        raise RuntimeError("No dataset-model combinations were found. Please check --root or the optional filters.")

    print("\nDiscovered combinations:")
    for dataset, model in combos:
        print(f"  - {dataset} | {model}")

    global_rows: List[Dict[str, Any]] = []
    all_status: List[Dict[str, Any]] = []

    for dataset, model in combos:
        all_metrics, status = generate_one_figure(args.root, dataset, model, args.output_dir)
        all_status.append(status)

        for noise_folder, panel_title in PANEL_INFO:
            panel_metrics = all_metrics.get(panel_title, {})
            for method, vals in panel_metrics.items():
                global_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "noise_folder": noise_folder,
                    "panel_title": panel_title,
                    "method": method,
                    "PoBL(10%)": vals["PoBL(10%)"],
                    "APFD": vals["APFD"],
                    "RAUC": vals["RAUC"],
                    "ROC_AUC": vals["ROC_AUC"],
                })

    csv_path = os.path.join(args.output_dir, "all_combinations_metrics_summary.csv")
    json_path = os.path.join(args.output_dir, "all_combinations_status.json")

    save_global_metrics_csv(global_rows, csv_path)
    save_json(all_status, json_path)

    print("\n" + "=" * 120)
    print("Batch plotting completed")
    print("=" * 120)
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {json_path}")
    print(f"[Total combinations] {len(combos)}")


if __name__ == "__main__":
    main()