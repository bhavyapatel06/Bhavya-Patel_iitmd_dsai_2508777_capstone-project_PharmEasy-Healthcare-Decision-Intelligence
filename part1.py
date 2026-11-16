"""Part 1: estimation, inference and regression baselines."""
import numpy as np
from common import load_data, write_results, ensure_outputs, OUTPUT_DIR


def simple_regression(x, y):
    xm, ym = float(np.mean(x)), float(np.mean(y))
    slope = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    intercept = ym - slope * xm
    fitted = intercept + slope * x
    residual = y - fitted
    tss = float(np.sum((y - ym) ** 2))
    ess = float(np.sum((fitted - ym) ** 2))
    rss = float(np.sum(residual ** 2))
    r2 = 1.0 - rss / tss
    rmse = float(np.sqrt(rss / len(y)))
    mae = float(np.mean(np.abs(residual)))
    return slope, intercept, fitted, residual, tss, ess, rss, r2, rmse, mae


def main():
    ensure_outputs()
    _, x, _, value = load_data()
    n = len(value)
    x4 = x[:, 3]
    mean_value = float(np.mean(value))
    sd_value = float(np.std(value, ddof=1))
    se_value = sd_value / np.sqrt(n)
    ci_low, ci_high = mean_value - 1.96 * se_value, mean_value + 1.96 * se_value
    z_stat = (mean_value - 8.0) / se_value
    n_required = int(np.ceil((1.96 * sd_value / 0.5) ** 2))

    (slope, intercept, fitted, residual, tss, ess, rss, r2, rmse, mae) = simple_regression(x4, value)
    z = (x4 - np.mean(x4)) / np.std(x4, ddof=1)
    m, c = 0.0, 0.0
    gd_costs = {0: float(np.mean(value ** 2))}
    for step in range(1, 501):
        err = value - (m * z + c)
        grad_m = -2.0 * np.mean(z * err)
        grad_c = -2.0 * np.mean(err)
        m -= 0.1 * grad_m
        c -= 0.1 * grad_c
        if step in (1, 10, 100, 500):
            gd_costs[step] = float(np.mean((value - (m * z + c)) ** 2))
    gd_slope = m / np.std(x4, ddof=1)
    gd_intercept = c - m * np.mean(x4) / np.std(x4, ddof=1)

    order = np.argsort(fitted, kind="stable")
    lower_raw = float(np.std(residual[order[:400]], ddof=1))
    upper_raw = float(np.std(residual[order[400:]], ddof=1))
    lx, ly = np.log(x4), np.log(value)
    (log_slope, log_intercept, log_fit, log_resid, log_tss, log_ess,
     log_rss, log_r2, log_rmse, _) = simple_regression(lx, ly)
    log_order = np.argsort(log_fit, kind="stable")
    lower_log = float(np.std(log_resid[log_order[:400]], ddof=1))
    upper_log = float(np.std(log_resid[log_order[400:]], ddof=1))
    log_adj_r2 = 1.0 - (1.0 - log_r2) * (n - 1) / (n - 1 - 1)

    design = np.column_stack((np.ones(n), np.log(x4), x[:, 0], x[:, 1], x[:, 2], x[:, 4], x[:, 5], x[:, 6]))
    beta = np.linalg.solve(design.T @ design, design.T @ ly)
    pred = design @ beta
    mr_r2 = 1.0 - np.sum((ly - pred) ** 2) / np.sum((ly - np.mean(ly)) ** 2)
    p = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p - 1)

    out = {
        "n": n, "mean_value": mean_value, "sd_value": sd_value, "se_value": se_value,
        "ci95_low": ci_low, "ci95_high": ci_high, "z_stat": z_stat,
        "reject_h0": abs(z_stat) > 1.96, "n_required": n_required,
        "reg_slope": slope, "reg_intercept": intercept, "tss": tss, "ess": ess, "rss": rss,
        "tss_equals_ess_plus_rss": abs(tss - ess - rss) < 1e-6 * tss,
        "r2": r2, "rmse": rmse, "mae": mae,
        "gd_J0": gd_costs[0], "gd_J1": gd_costs[1], "gd_J10": gd_costs[10],
        "gd_J100": gd_costs[100], "gd_J500": gd_costs[500],
        "gd_slope": gd_slope, "gd_intercept": gd_intercept,
        "gd_slope_gap": abs(gd_slope - slope),
        "resid_sd_lower_raw": lower_raw, "resid_sd_upper_raw": upper_raw,
        "resid_sd_ratio_raw": upper_raw / lower_raw,
        "loglog_intercept": log_intercept, "loglog_slope": log_slope,
        "loglog_r2": log_r2, "loglog_rmse": log_rmse,
        "resid_sd_lower_log": lower_log, "resid_sd_upper_log": upper_log,
        "resid_sd_ratio_log": upper_log / lower_log,
    }
    for i, val in enumerate(beta):
        out[f"mr_beta_{i}"] = float(val)
    out["mr_r2"] = float(mr_r2)
    out["mr_adj_r2"] = float(mr_adj_r2)
    out["loglog_adj_r2"] = float(log_adj_r2)
    write_results(OUTPUT_DIR / "part1_results.txt", out)
    print("Part 1 complete:", OUTPUT_DIR / "part1_results.txt")

if __name__ == "__main__":
    main()
