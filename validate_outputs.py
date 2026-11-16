"""Offline acceptance checks for generated files and pinned capstone checkpoints."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
EXPECTED_DATA_MD5 = "e26b46e82f1a1af44dcabc133050802e"
CSV_SPECS = {
    "part2_thresholds.csv": ("tau,tp,fp,tn,fn,tpr,fpr".split(","), 9),
    "part2_knn_cv.csv": (["K", "mean_cv_f1"], 11),
    "part3_segments.csv": ("segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value".split(","), 3),
    "part3_hierarchical.csv": ("linkage,merge_no,cluster_a,cluster_b,height".split(","), 15),
    "part4_splits.csv": ("split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain".split(","), 4),
    "part4_voting.csv": ("model,tp,fp,tn,fn,precision,recall,f1".split(","), 5),
    "part4_adaboost.csv": ("round,point,weight_before,weight_unnormalised,weight_after".split(","), 16),
    "part4_architecture.csv": (["layer", "parameters"], 7),
}
REQUIRED_KEYS = {
    "part1_results.txt": "n mean_value sd_value se_value ci95_low ci95_high z_stat reject_h0 n_required reg_slope reg_intercept tss ess rss tss_equals_ess_plus_rss r2 rmse mae gd_J0 gd_J1 gd_J10 gd_J100 gd_J500 gd_slope gd_intercept gd_slope_gap resid_sd_lower_raw resid_sd_upper_raw resid_sd_ratio_raw loglog_intercept loglog_slope loglog_r2 loglog_rmse resid_sd_lower_log resid_sd_upper_log resid_sd_ratio_log mr_beta_0 mr_beta_1 mr_beta_2 mr_beta_3 mr_beta_4 mr_beta_5 mr_beta_6 mr_beta_7 mr_r2 mr_adj_r2 loglog_adj_r2".split(),
    "part2_results.txt": "loss_epoch_200_eta_0.01 loss_epoch_200_eta_0.1 loss_epoch_200_eta_1.0 train_rows train_positives test_rows test_positives loss_epoch_0 loss_epoch_1 loss_epoch_10 loss_epoch_100 loss_epoch_1000 test_bce weight_X1 weight_X2 weight_X3 weight_X4 weight_X5 weight_X6 weight_X7 odds_ratio_X1 odds_ratio_X2 odds_ratio_X3 odds_ratio_X4 odds_ratio_X5 odds_ratio_X6 odds_ratio_X7 bias tau_0.5_tp tau_0.5_fp tau_0.5_tn tau_0.5_fn tau_0.5_accuracy tau_0.5_precision tau_0.5_recall tau_0.5_f1 tau_0.5_specificity tau_0.3_tp tau_0.3_fp tau_0.3_tn tau_0.3_fn tau_0.3_accuracy tau_0.3_precision tau_0.3_recall tau_0.3_f1 tau_0.3_specificity blind_accuracy knn_best_k knn_tp knn_fp knn_tn knn_fn knn_accuracy knn_precision knn_recall knn_f1 cost_tau cost_train_at_tau cost_test_at_tau cost_test_at_0.5 cost_test_flag_nobody".split(),
    "part3_results.txt": "wcss_k2 wcss_k3 wcss_k4 wcss_k5 wcss_k6 elbow_k size_segment_A size_segment_B size_segment_C single_clusters_at_cut complete_clusters_at_cut average_clusters_at_cut eigenvalue_sum eigenvalue_1 eigenvalue_2 eigenvalue_3 eigenvalue_4 eigenvalue_5 explained_ratio_1 explained_ratio_2 explained_ratio_3 explained_ratio_4 explained_ratio_5 cumulative_ratio_1 cumulative_ratio_2 cumulative_ratio_3 cumulative_ratio_4 cumulative_ratio_5 components_for_80pct pc1_loading_X1 pc1_loading_X2 pc1_loading_X3 pc1_loading_X4 pc1_loading_X5 pc2_loading_X1 pc2_loading_X2 pc2_loading_X3 pc2_loading_X4 pc2_loading_X5 row1_pc1 row1_pc2 row2_pc1 row2_pc2 row3_pc1 row3_pc2 mean_pc1_segment_A mean_pc1_segment_B mean_pc1_segment_C".split(),
    "part4_results.txt": "root_entropy root_gini best_split_by_info_gain best_split_by_gini_gain stump_rate_yes stump_rate_no round1_error round1_alpha round2_error round2_alpha ensemble_accuracy round1_misclassified_weight_after oob_fraction mlp_total cnn_total fc_layer_params conv_layer_params rnn_total".split(),
}


def read_results(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " not in line:
            raise AssertionError(f"Malformed result line in {path.name}: {line!r}")
        key, value = line.split(" = ", 1)
        values[key] = value
    return values


def main():
    data_path = ROOT / "data" / "records.csv"
    digest = hashlib.md5(data_path.read_bytes()).hexdigest()
    assert digest == EXPECTED_DATA_MD5, f"Dataset hash mismatch: {digest}"
    with data_path.open("r", newline="", encoding="utf-8") as f:
        data_rows = list(csv.reader(f))
    assert len(data_rows) == 801 and len(data_rows[0]) == 10

    results = {}
    for filename, required in REQUIRED_KEYS.items():
        path = OUT / filename
        assert path.is_file(), f"Missing {path}"
        values = read_results(path)
        missing = sorted(set(required) - set(values))
        assert not missing, f"{filename} missing keys: {missing}"
        results[filename] = values

    for filename, (header, expected_rows) in CSV_SPECS.items():
        path = OUT / filename
        assert path.is_file(), f"Missing {path}"
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == header, f"Header mismatch in {filename}: {rows[0]}"
        assert len(rows) - 1 == expected_rows, f"Row count mismatch in {filename}"

    assert results["part1_results.txt"]["n"] == "800"
    assert results["part1_results.txt"]["mean_value"] == "8.9382"
    assert results["part2_results.txt"]["train_rows"] == "560"
    assert results["part2_results.txt"]["test_rows"] == "240"
    assert results["part3_results.txt"]["wcss_k2"] == "2262.6753"
    assert results["part3_results.txt"]["eigenvalue_sum"] == "5.0000"
    assert results["part4_results.txt"]["root_entropy"] == "0.7462"
    assert results["part4_results.txt"]["round1_error"] == "0.2500"
    assert abs(float(results["part1_results.txt"]["gd_slope_gap"])) < 0.001
    assert float(results["part2_results.txt"]["cost_tau"]) == 0.15
    assert sum(int(results["part3_results.txt"][f"size_segment_{s}"]) for s in "ABC") == 800
    print("Validation passed: dataset hash, required result keys, CSV headers/row counts and assignment checkpoints.")


if __name__ == "__main__":
    main()
