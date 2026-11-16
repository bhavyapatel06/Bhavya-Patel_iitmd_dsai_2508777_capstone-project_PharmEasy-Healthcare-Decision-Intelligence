"""Part 2: logistic regression, threshold analysis and KNN from scratch."""
import numpy as np
from common import (load_data, stratified_split, standardise_train_test, fit_logistic,
                    sigmoid, binary_cross_entropy, confusion_metrics, knn_probabilities,
                    write_results, write_csv, ensure_outputs, OUTPUT_DIR)


def main():
    ensure_outputs()
    _, x, y, _ = load_data()
    train_idx, test_idx = stratified_split(y)
    x_train, x_test, train_mean, train_sd = standardise_train_test(x, train_idx, test_idx)
    y_train, y_test = y[train_idx], y[test_idx]

    capture = (0, 1, 10, 100, 1000)
    w, b, losses = fit_logistic(x_train, y_train, 0.1, 1000, capture)
    train_prob = sigmoid(x_train @ w + b)
    test_prob = sigmoid(x_test @ w + b)
    test_bce = binary_cross_entropy(y_test, test_prob)

    lr_losses = {}
    for lr in (0.01, 0.1, 1.0):
        _, _, ls = fit_logistic(x_train, y_train, lr, 200, (200,))
        lr_losses[lr] = ls[200]

    out = {
        "loss_epoch_200_eta_0.01": lr_losses[0.01],
        "loss_epoch_200_eta_0.1": lr_losses[0.1],
        "loss_epoch_200_eta_1.0": lr_losses[1.0],
        "train_rows": len(train_idx), "train_positives": int(np.sum(y_train)),
        "test_rows": len(test_idx), "test_positives": int(np.sum(y_test)),
        "loss_epoch_0": losses[0], "loss_epoch_1": losses[1],
        "loss_epoch_10": losses[10], "loss_epoch_100": losses[100],
        "loss_epoch_1000": losses[1000], "test_bce": test_bce,
    }
    for i, val in enumerate(w, start=1):
        out[f"weight_X{i}"] = float(val)
        out[f"odds_ratio_X{i}"] = float(np.exp(val))
    out["bias"] = float(b)

    for tau in (0.5, 0.3):
        m = confusion_metrics(y_test, (test_prob >= tau).astype(int))
        for key in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity"):
            out[f"tau_{tau}_{key}"] = m[key]
    blind = np.zeros_like(y_test)
    out["blind_accuracy"] = confusion_metrics(y_test, blind)["accuracy"]

    threshold_rows = []
    for tau in np.arange(0.1, 1.0, 0.1):
        m = confusion_metrics(y_test, (test_prob >= tau).astype(int))
        threshold_rows.append((float(tau), m["tp"], m["fp"], m["tn"], m["fn"], m["tpr"], m["fpr"]))
    write_csv(OUTPUT_DIR / "part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], threshold_rows)

    # Cross-validation uses fixed training-set scaling and fold = training position modulo 5.
    cv_rows = []
    for k in range(1, 22, 2):
        fold_f1 = []
        positions = np.arange(len(y_train))
        for fold in range(5):
            val_mask = (positions % 5) == fold
            ref_mask = ~val_mask
            p = knn_probabilities(x_train[ref_mask], y_train[ref_mask], x_train[val_mask], k)
            pred = (p >= (k + 1) / (2 * k)).astype(int)
            fold_f1.append(confusion_metrics(y_train[val_mask], pred)["f1"])
        cv_rows.append((k, float(np.mean(fold_f1))))
    write_csv(OUTPUT_DIR / "part2_knn_cv.csv", ["K", "mean_cv_f1"], cv_rows)
    best_k = sorted(cv_rows, key=lambda row: (-row[1], row[0]))[0][0]
    knn_test_prob = knn_probabilities(x_train, y_train, x_test, best_k)
    knn_test_pred = (knn_test_prob >= (best_k + 1) / (2 * best_k)).astype(int)
    knn_metrics = confusion_metrics(y_test, knn_test_pred)
    out["knn_best_k"] = int(best_k)
    for key in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1"):
        out[f"knn_{key}"] = knn_metrics[key]

    # Select threshold using training data only. The first minimum gives the smaller tau on ties.
    candidate_costs = []
    for k in range(1, 20):
        tau = k / 20.0
        m = confusion_metrics(y_train, (train_prob >= tau).astype(int))
        cost = 50 * m["fp"] + 400 * m["fn"]
        candidate_costs.append((cost, tau))
    cost_train, cost_tau = min(candidate_costs, key=lambda pair: (pair[0], pair[1]))
    test_at_tau = confusion_metrics(y_test, (test_prob >= cost_tau).astype(int))
    test_at_half = confusion_metrics(y_test, (test_prob >= 0.5).astype(int))
    out["cost_tau"] = float(cost_tau)
    out["cost_train_at_tau"] = int(cost_train)
    out["cost_test_at_tau"] = int(50 * test_at_tau["fp"] + 400 * test_at_tau["fn"])
    out["cost_test_at_0.5"] = int(50 * test_at_half["fp"] + 400 * test_at_half["fn"])
    out["cost_test_flag_nobody"] = int(400 * np.sum(y_test))

    write_results(OUTPUT_DIR / "part2_results.txt", out)
    print("Part 2 complete:", OUTPUT_DIR / "part2_results.txt", "best K =", best_k)

if __name__ == "__main__":
    main()
