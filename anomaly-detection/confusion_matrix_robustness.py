"""
Reproduces the exact protocol of evaluate_anomaly_detection_robustness.py
(10 splits, seeds 0-9, same train/test/threshold logic) but additionally
captures TP/FP/FN/TN per split, for the raw-counts / confusion-matrix
table requested by the professor (Item #11) alongside Table 4.3.

Does NOT change methodology or invent numbers -- same feature, same
protocol, same seeds as the script that produced robustness_variance_results.csv
(already validated against Table 4.3's reported means).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve

INPUT_PATH = "augmented_real_dataset.csv"
FEATURE_COLUMNS = ["deviation_zscore_historical"]
TEST_SIZE = 0.30
VARIANCE_SEEDS = list(range(10))
CANONICAL_SEED = 42


def load_and_prepare(path=INPUT_PATH):
    df = pd.read_csv(path)
    df["deviation_zscore_historical"] = df["deviation_zscore_historical"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["deviation_zscore_historical"]).reset_index(drop=True)
    return df


def best_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0,
    )
    best_idx = np.argmax(f1[:-1]) if len(thresholds) else 0
    return thresholds[best_idx] if len(thresholds) else 0.0, f1[best_idx] if len(f1) else 0.0


def fit_and_predict(df, seed):
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=seed
    )
    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train["label"].values
    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test["label"].values
    true_rate_train = y_train.mean()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    preds = {}

    iso = IsolationForest(n_estimators=200, contamination=true_rate_train, random_state=seed)
    iso.fit(X_train_scaled)
    iso_scores_train = -iso.decision_function(X_train_scaled)
    iso_scores_test = -iso.decision_function(X_test_scaled)
    iso_pred_default = (iso.predict(X_test_scaled) == -1).astype(int)
    thr, _ = best_f1_threshold(y_train, iso_scores_train)
    iso_pred_best = (iso_scores_test >= thr).astype(int)
    preds["Isolation Forest | default"] = iso_pred_default
    preds["Isolation Forest | best-F1"] = iso_pred_best

    ae = MLPRegressor(hidden_layer_sizes=(3,), activation="relu", max_iter=1000, random_state=seed)
    ae.fit(X_train_scaled, X_train_scaled)
    recon_train = ae.predict(X_train_scaled)
    recon_error_train = np.mean((X_train_scaled - recon_train.reshape(X_train_scaled.shape)) ** 2, axis=1)
    recon_test = ae.predict(X_test_scaled)
    recon_error_test = np.mean((X_test_scaled - recon_test.reshape(X_test_scaled.shape)) ** 2, axis=1)
    ae_threshold_default = np.quantile(recon_error_train, 1 - true_rate_train)
    ae_pred_default = (recon_error_test >= ae_threshold_default).astype(int)
    thr_ae, _ = best_f1_threshold(y_train, recon_error_train)
    ae_pred_best = (recon_error_test >= thr_ae).astype(int)
    preds["Autoencoder | default"] = ae_pred_default
    preds["Autoencoder | best-F1"] = ae_pred_best

    metrics = {}
    for name, pred in preds.items():
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        metrics[name] = {
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "n_test": len(y_test),
        }
    return metrics


def main():
    df = load_and_prepare()
    all_runs = {}
    for seed in VARIANCE_SEEDS:
        m = fit_and_predict(df, seed)
        for name, vals in m.items():
            all_runs.setdefault(name, []).append(vals)

    # sanity check vs already-validated robustness_variance_results.csv
    print("Sanity check (should match robustness_variance_results.csv):")
    for name, runs in all_runs.items():
        p = np.array([r["precision"] for r in runs])
        r = np.array([r["recall"] for r in runs])
        f = np.array([r["f1"] for r in runs])
        print(f"  {name:32s} precision={p.mean():.3f}+/-{p.std():.3f}  recall={r.mean():.3f}+/-{r.std():.3f}  f1={f.mean():.3f}+/-{f.std():.3f}")

    print("\nRaw counts (mean +/- SD across the same 10 splits, seeds 0-9):")
    rows = []
    for name, runs in all_runs.items():
        tp = np.array([r["TP"] for r in runs], dtype=float)
        fp = np.array([r["FP"] for r in runs], dtype=float)
        fn = np.array([r["FN"] for r in runs], dtype=float)
        tn = np.array([r["TN"] for r in runs], dtype=float)
        n_test = runs[0]["n_test"]
        row = {
            "model_threshold": name, "n_test": n_test,
            "TP_mean": round(tp.mean(), 1), "TP_std": round(tp.std(), 1),
            "FP_mean": round(fp.mean(), 1), "FP_std": round(fp.std(), 1),
            "FN_mean": round(fn.mean(), 1), "FN_std": round(fn.std(), 1),
            "TN_mean": round(tn.mean(), 1), "TN_std": round(tn.std(), 1),
        }
        rows.append(row)
        print(f"  [{name}] n_test={n_test}")
        print(f"    TP={row['TP_mean']}+/-{row['TP_std']}  FP={row['FP_mean']}+/-{row['FP_std']}  "
              f"FN={row['FN_mean']}+/-{row['FN_std']}  TN={row['TN_mean']}+/-{row['TN_std']}")

    out = pd.DataFrame(rows)
    out.to_csv("confusion_matrix_robustness_results.csv", index=False)
    print("\nSaved confusion_matrix_robustness_results.csv")

    # also dump per-split raw rows for full traceability / appendix if needed
    detail_rows = []
    for name, runs in all_runs.items():
        for seed, r in zip(VARIANCE_SEEDS, runs):
            detail_rows.append({"model_threshold": name, "seed": seed, **{k: r[k] for k in ["TP","FP","FN","TN","n_test"]}})
    pd.DataFrame(detail_rows).to_csv("confusion_matrix_robustness_per_split.csv", index=False)
    print("Saved confusion_matrix_robustness_per_split.csv")


if __name__ == "__main__":
    main()
