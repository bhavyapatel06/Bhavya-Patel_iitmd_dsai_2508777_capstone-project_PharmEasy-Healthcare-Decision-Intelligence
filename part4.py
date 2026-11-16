"""Part 4: split criteria, voting, AdaBoost arithmetic and architecture sizing."""
import numpy as np
from common import (load_data, stratified_split, standardise_train_test, fit_logistic,
                    sigmoid, confusion_metrics, knn_probabilities, write_results,
                    write_csv, ensure_outputs, OUTPUT_DIR)


def entropy(labels):
    if len(labels) == 0:
        return 0.0
    p = float(np.mean(labels))
    if p in (0.0, 1.0):
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = float(np.mean(labels))
    return 1.0 - p * p - (1.0 - p) ** 2


def split_scores(mask, y):
    yes, no = y[mask], y[~mask]
    n = len(y)
    ig = entropy(y) - (len(yes) / n) * entropy(yes) - (len(no) / n) * entropy(no)
    gg = gini(y) - (len(yes) / n) * gini(yes) - (len(no) / n) * gini(no)
    return len(yes), int(np.sum(yes)), len(no), int(np.sum(no)), float(ig), float(gg)


def main():
    ensure_outputs()
    _, x, y, _ = load_data()
    train_idx, test_idx = stratified_split(y)
    x_train_raw, x_test_raw = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    x_train, x_test, _, _ = standardise_train_test(x, train_idx, test_idx)

    split_masks = {
        "S1": x_train_raw[:, 5] == 1,
        "S2": x_train_raw[:, 6] == 1,
        "S3": x_train_raw[:, 2] > 0.5,
        "S4": x_train_raw[:, 0] <= 24,
    }
    split_rows = []
    split_details = {}
    for name, mask in split_masks.items():
        vals = split_scores(mask, y_train)
        split_details[name] = vals
        split_rows.append((name, *vals))
    best_info = max(split_details, key=lambda name: (split_details[name][4], -int(name[1:])))
    best_gini = max(split_details, key=lambda name: (split_details[name][5], -int(name[1:])))
    write_csv(OUTPUT_DIR / "part4_splits.csv",
              ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], split_rows)

    # Refit exactly the Part 2 logistic model and KNN selected by the pinned CV folds.
    w, b, _ = fit_logistic(x_train, y_train, 0.1, 1000, ())
    logistic_prob = sigmoid(x_test @ w + b)
    cv_rows = []
    positions = np.arange(len(y_train))
    for k in range(1, 22, 2):
        fold_f1 = []
        for fold in range(5):
            val_mask = (positions % 5) == fold
            ref_mask = ~val_mask
            p = knn_probabilities(x_train[ref_mask], y_train[ref_mask], x_train[val_mask], k)
            pred = (p >= (k + 1) / (2 * k)).astype(int)
            fold_f1.append(confusion_metrics(y_train[val_mask], pred)["f1"])
        cv_rows.append((k, float(np.mean(fold_f1))))
    best_k = sorted(cv_rows, key=lambda row: (-row[1], row[0]))[0][0]
    knn_prob = knn_probabilities(x_train, y_train, x_test, best_k)
    stump_mask_train = split_masks[best_info]
    stump_rate_yes = float(np.mean(y_train[stump_mask_train])) if np.any(stump_mask_train) else 0.0
    stump_rate_no = float(np.mean(y_train[~stump_mask_train])) if np.any(~stump_mask_train) else 0.0
    stump_mask_test = {
        "S1": x_test_raw[:, 5] == 1,
        "S2": x_test_raw[:, 6] == 1,
        "S3": x_test_raw[:, 2] > 0.5,
        "S4": x_test_raw[:, 0] <= 24,
    }[best_info]
    stump_prob = np.where(stump_mask_test, stump_rate_yes, stump_rate_no)
    logistic_pred = (logistic_prob >= 0.3).astype(int)
    knn_pred = (knn_prob >= (best_k + 1) / (2 * best_k)).astype(int)
    stump_pred = (stump_prob >= 0.3).astype(int)
    hard_pred = ((logistic_pred + knn_pred + stump_pred) >= 2).astype(int)
    soft_prob = (logistic_prob + knn_prob + stump_prob) / 3.0
    soft_pred = (soft_prob >= 0.3).astype(int)
    voting_rows = []
    for name, pred in (("logistic", logistic_pred), ("knn", knn_pred), ("stump", stump_pred),
                       ("hard_vote", hard_pred), ("soft_vote", soft_pred)):
        m = confusion_metrics(y_test, pred)
        voting_rows.append((name, m["tp"], m["fp"], m["tn"], m["fn"], m["precision"], m["recall"], m["f1"]))
    write_csv(OUTPUT_DIR / "part4_voting.csv",
              ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], voting_rows)

    y_boost = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=int)
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=int)
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=int)
    weights = np.full(8, 1.0 / 8.0)
    boost_rows = []
    alphas = []
    round1_weights_after = None
    for round_no, h in ((1, h1), (2, h2)):
        before = weights.copy()
        error = float(np.sum(before[h != y_boost]))
        alpha = 0.5 * np.log((1.0 - error) / error)
        unnormalised = before * np.exp(-alpha * y_boost * h)
        weights = unnormalised / np.sum(unnormalised)
        alphas.append(alpha)
        if round_no == 1:
            round1_weights_after = weights.copy()
        for point in range(8):
            boost_rows.append((round_no, point + 1, float(before[point]),
                               float(unnormalised[point]), float(weights[point])))
        if round_no == 1:
            round1_error, round1_alpha = error, alpha
        else:
            round2_error, round2_alpha = error, alpha
    ensemble_score = alphas[0] * h1 + alphas[1] * h2
    ensemble_pred = np.where(ensemble_score >= 0, 1, -1)
    ensemble_accuracy = float(np.mean(ensemble_pred == y_boost))
    write_csv(OUTPUT_DIR / "part4_adaboost.csv",
              ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], boost_rows)

    # Architecture parameter counts; biases are included in every affine layer.
    architecture = [
        ("mlp_hidden1", 784 * 16 + 16),
        ("mlp_hidden2", 16 * 16 + 16),
        ("mlp_output", 16 * 10 + 10),
        ("cnn_conv1", 5 * 5 * 1 * 6 + 6),
        ("cnn_conv2", 5 * 5 * 6 * 16 + 16),
        ("cnn_dense1", (5 * 5 * 16) * 32 + 32),
        ("cnn_dense2", 32 * 10 + 10),
    ]
    write_csv(OUTPUT_DIR / "part4_architecture.csv", ["layer", "parameters"], architecture)
    out = {
        "root_entropy": entropy(y_train), "root_gini": gini(y_train),
        "best_split_by_info_gain": best_info, "best_split_by_gini_gain": best_gini,
        "stump_rate_yes": stump_rate_yes, "stump_rate_no": stump_rate_no,
        "knn_best_k": int(best_k), "round1_error": round1_error,
        "round1_alpha": round1_alpha, "round2_error": round2_error,
        "round2_alpha": round2_alpha, "ensemble_accuracy": ensemble_accuracy,
        "round1_misclassified_weight_after": float(np.sum(round1_weights_after[h1 != y_boost])),
        "oob_fraction": float((1.0 - 1.0 / 560.0) ** 560),
        "mlp_total": int(sum(v for k, v in architecture if k.startswith("mlp_"))),
        "cnn_total": int(sum(v for k, v in architecture if k.startswith("cnn_"))),
        "fc_layer_params": int(128 * 128 * 3 * 1024 + 1024),
        "conv_layer_params": int(3 * 3 * 3 * 64 + 64),
        "rnn_total": int(64 * 128 + 128 * 128 + 128 * 16 + 128 + 16),
    }
    for name, vals in split_details.items():
        out[f"{name}_n_yes"] = vals[0]
        out[f"{name}_positives_yes"] = vals[1]
        out[f"{name}_n_no"] = vals[2]
        out[f"{name}_positives_no"] = vals[3]
        out[f"{name}_info_gain"] = vals[4]
        out[f"{name}_gini_gain"] = vals[5]
    write_results(OUTPUT_DIR / "part4_results.txt", out)
    print("Part 4 complete:", OUTPUT_DIR / "part4_results.txt")

if __name__ == "__main__":
    main()
