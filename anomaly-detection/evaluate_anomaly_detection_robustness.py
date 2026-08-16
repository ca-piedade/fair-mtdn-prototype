"""
Robustness checks for the Section 4.5 augmented real-data evaluation (RQ1 / Table
4.3). Supplements evaluate_anomaly_detection_augmented.py (the canonical, single
seed=42 result reported in Table 4.3); does not replace it.

Two checks:
  1. VARIANCE across 10 independent train/test splits (seeds 0-9), reporting
     mean +/- std for precision/recall/F1 per model/threshold, instead of relying on
     a single point estimate.
  2. PER-CATEGORY RECALL on the canonical seed=42 split: what fraction of
     "consumption" (isolated) anomalies is caught vs. what fraction of "recipe"
     (systematic) anomalies is caught, to see whether one injection mechanism is
     harder to detect than the other.

Run
---
    python3 build_augmented_real_dataset.py   # if augmented_real_dataset.csv is stale
    python3 evaluate_anomaly_detection_robustness.py

Output
------
Console only: two summary tables. Also saves
robustness_variance_results.csv and robustness_per_category_recall.csv.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

INPUT_PATH = "augmented_real_dataset.csv"
FEATURE_COLUMNS = ["deviation_zscore_historical"]
TEST_SIZE = 0.30
VARIANCE_SEEDS = list(range(10))     # independent of the canonical Table 4.3 seed
CANONICAL_SEED = 42                  # matches evaluate_anomaly_detection_augmented.py


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
    """Runs the full split -> train-only feature/threshold selection -> test-only
    metrics pipeline once, for one seed. Returns per-model test predictions (for
    per-category breakdown) and their metrics (for variance aggregation)."""
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

    out = {"df_test": df_test, "y_test": y_test, "predictions": {}, "metrics": {}}

    # Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=true_rate_train, random_state=seed)
    iso.fit(X_train_scaled)
    iso_scores_train = -iso.decision_function(X_train_scaled)
    iso_scores_test = -iso.decision_function(X_test_scaled)

    iso_pred_default = (iso.predict(X_test_scaled) == -1).astype(int)
    thr, _ = best_f1_threshold(y_train, iso_scores_train)
    iso_pred_best = (iso_scores_test >= thr).astype(int)

    out["predictions"]["Isolation Forest | default"] = iso_pred_default
    out["predictions"]["Isolation Forest | best-F1"] = iso_pred_best

    # Autoencoder
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

    out["predictions"]["Autoencoder | default"] = ae_pred_default
    out["predictions"]["Autoencoder | best-F1"] = ae_pred_best

    for name, pred in out["predictions"].items():
        out["metrics"][name] = {
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        }
    return out


def variance_check(df):
    print("=" * 70)
    print(f"VARIANCE across {len(VARIANCE_SEEDS)} independent train/test splits (seeds {VARIANCE_SEEDS[0]}-{VARIANCE_SEEDS[-1]})")
    print("=" * 70)
    all_runs = {}
    for seed in VARIANCE_SEEDS:
        result = fit_and_predict(df, seed)
        for name, m in result["metrics"].items():
            all_runs.setdefault(name, []).append(m)

    rows = []
    for name, runs in all_runs.items():
        for metric in ["precision", "recall", "f1"]:
            values = np.array([r[metric] for r in runs])
            rows.append({
                "model_threshold": name, "metric": metric,
                "mean": round(values.mean(), 3), "std": round(values.std(), 3),
                "min": round(values.min(), 3), "max": round(values.max(), 3),
            })
    summary = pd.DataFrame(rows)
    for name in all_runs:
        sub = summary[summary["model_threshold"] == name]
        print(f"\n[{name}]")
        for _, r in sub.iterrows():
            print(f"  {r['metric']:9s}: {r['mean']:.3f} +/- {r['std']:.3f}  (range {r['min']:.3f}-{r['max']:.3f})")

    summary.to_csv("robustness_variance_results.csv", index=False)
    print("\nSaved robustness_variance_results.csv")
    return summary


def per_category_check(df):
    print("\n" + "=" * 70)
    print(f"PER-CATEGORY RECALL, canonical seed={CANONICAL_SEED} split (matches Table 4.3)")
    print("=" * 70)
    result = fit_and_predict(df, CANONICAL_SEED)
    df_test = result["df_test"].reset_index(drop=True)

    rows = []
    for name, pred in result["predictions"].items():
        pred = np.asarray(pred)
        for category in ["consumption", "recipe"]:
            mask = (df_test["anomaly_type"] == category).values
            n_cat = mask.sum()
            if n_cat == 0:
                continue
            recall_cat = pred[mask].mean()
            rows.append({
                "model_threshold": name, "category": category,
                "n_in_test": int(n_cat), "recall": round(recall_cat, 3),
            })
        # false positive rate on "none" rows, for context
        none_mask = (df_test["anomaly_type"] == "none").values
        fp_rate = pred[none_mask].mean()
        rows.append({
            "model_threshold": name, "category": "none (false-positive rate)",
            "n_in_test": int(none_mask.sum()), "recall": round(fp_rate, 3),
        })

    out = pd.DataFrame(rows)
    for name in result["predictions"]:
        sub = out[out["model_threshold"] == name]
        print(f"\n[{name}]")
        for _, r in sub.iterrows():
            print(f"  {r['category']:28s} n={r['n_in_test']:4d}  rate={r['recall']:.3f}")

    out.to_csv("robustness_per_category_recall.csv", index=False)
    print("\nSaved robustness_per_category_recall.csv")
    return out


def main():
    df = load_and_prepare()
    variance_check(df)
    per_category_check(df)


if __name__ == "__main__":
    main()
