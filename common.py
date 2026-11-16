"""Shared, hand-written numerical routines for the PharmEasy capstone."""
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_DIR = ROOT / "outputs"
FEATURE_NAMES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]


def load_data():
    with DATA_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 800:
        raise ValueError(f"Expected 800 data rows, found {len(rows)}")
    x = np.array([[float(r[k]) for k in (
        "months_registered", "missed_visit_count_12m", "copay_burden_ratio",
        "avg_bill_inr_thousands", "adherence_score", "chronic_condition_flag",
        "insured_flag") ] for r in rows], dtype=float)
    y = np.array([int(r["no_show"]) for r in rows], dtype=int)
    value = np.array([float(r["care_plan_cost_inr_thousands"]) for r in rows], dtype=float)
    return rows, x, y, value


def stratified_split(y):
    train_parts, test_parts = [], []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        cut = (7 * len(idx)) // 10
        train_parts.extend(idx[:cut].tolist())
        test_parts.extend(idx[cut:].tolist())
    return np.array(sorted(train_parts), dtype=int), np.array(sorted(test_parts), dtype=int)


def standardise_train_test(x, train_idx, test_idx):
    mean = x[train_idx].mean(axis=0)
    sd = x[train_idx].std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError("A feature has zero training standard deviation")
    return (x[train_idx] - mean) / sd, (x[test_idx] - mean) / sd, mean, sd


def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y, p):
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_logistic(x, y, learning_rate=0.1, updates=1000, capture=()):
    n, d = x.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    losses = {}
    wanted = set(capture)
    if 0 in wanted:
        losses[0] = binary_cross_entropy(y, sigmoid(x @ w + b))
    for step in range(1, updates + 1):
        p = sigmoid(x @ w + b)
        error = p - y
        grad_w = (x.T @ error) / n
        grad_b = float(np.mean(error))
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        if step in wanted:
            losses[step] = binary_cross_entropy(y, sigmoid(x @ w + b))
    return w, b, losses


def confusion_metrics(y, pred):
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1, "tpr": recall, "fpr": fpr}


def knn_probabilities(reference_x, reference_y, query_x, k):
    probs = np.empty(len(query_x), dtype=float)
    for i, q in enumerate(query_x):
        distances = np.sqrt(np.sum((reference_x - q) ** 2, axis=1))
        order = np.argsort(distances, kind="stable")
        probs[i] = np.mean(reference_y[order[:k]])
    return probs


def write_results(path, values):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        for key, value in values.items():
            if isinstance(value, (bool, np.bool_)):
                rendered = "yes" if value else "no"
            elif isinstance(value, (int, np.integer)):
                rendered = str(int(value))
            elif isinstance(value, (float, np.floating)):
                rendered = f"{float(value):.4f}"
            else:
                rendered = str(value)
            f.write(f"{key} = {rendered}\n")


def write_csv(path, header, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{v:.4f}" if isinstance(v, (float, np.floating)) else v for v in row])


def ensure_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
