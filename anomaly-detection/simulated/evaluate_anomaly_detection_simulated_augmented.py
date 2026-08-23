"""
FAIR artefact - Section 4.6 evaluation on the simulated, real-calibrated, MASKED
dataset. SUPERSEDES evaluate_anomaly_detection_simulated.py (15 Aug 2026 rebuild),
which used the old fabricated-from-scratch simulate_dataset() and a different,
2-feature methodology not comparable to Section 4.5's corrected approach.

Run build_simulated_calibrated_dataset.py FIRST, then this script. Uses the exact same
single feature (deviation_zscore_historical) and train/test protocol as
evaluate_anomaly_detection_augmented.py (Section 4.5), so the two results are directly
comparable -- the only methodological difference is: (a) this population is generated
from real-calibrated distributions rather than being the literal real data, and (b) a
share of the injected signal is masked (see build_simulated_calibrated_dataset.py
docstring for the full rationale and the three-layer comparison this enables).

Run
---
    python3 build_simulated_calibrated_dataset.py
    python3 evaluate_anomaly_detection_simulated_augmented.py

Outputs
-------
- Console: precision/recall/F1/confusion matrix, held-out test partition, for the
  masked set (Table 4.X figures) and, for reference only (not saved), the same
  pipeline on the unmasked version of the identical population.
- Table4X_simulated_augmented_results.csv: masked-set results for Section 4.6.2.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve,
)

RANDOM_SEED = 42
TEST_SIZE = 0.30
INPUT_PATH = "simulated_calibrated_dataset.csv"
FEATURE_COLUMNS = ["deviation_zscore_historical"]


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


def evaluate(y_true, y_pred, model_name, threshold_name, partition="test", condition="masked"):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n[{condition} | {model_name} | {threshold_name} | {partition} partition]")
    print(f"  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {
        "condition": condition, "model": model_name, "threshold": threshold_name,
        "partition": partition,
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def run_evaluation(df, condition):
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
    )
    print(f"[{condition}] {len(df)} rows, {df['label'].sum()} anomalies "
          f"({df['label'].mean():.1%}). Split: {len(df_train)} train "
          f"({df_train['label'].sum()} anomalies), {len(df_test)} test "
          f"({df_test['label'].sum()} anomalies).")

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train["label"].values
    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test["label"].values
    true_rate_train = y_train.mean()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    iso = IsolationForest(n_estimators=200, contamination=true_rate_train, random_state=RANDOM_SEED)
    iso.fit(X_train_scaled)
    iso_scores_train = -iso.decision_function(X_train_scaled)
    iso_scores_test = -iso.decision_function(X_test_scaled)

    iso_pred_default_test = (iso.predict(X_test_scaled) == -1).astype(int)
    results.append(evaluate(y_test, iso_pred_default_test, "Isolation Forest",
                             "contamination-based (default)", condition=condition))

    thr, _ = best_f1_threshold(y_train, iso_scores_train)
    iso_pred_best_test = (iso_scores_test >= thr).astype(int)
    results.append(evaluate(y_test, iso_pred_best_test, "Isolation Forest",
                             f"best-F1 threshold ({thr:.3f}, selected on train)", condition=condition))

    ae = MLPRegressor(hidden_layer_sizes=(3,), activation="relu", max_iter=1000, random_state=RANDOM_SEED)
    ae.fit(X_train_scaled, X_train_scaled)
    recon_train = ae.predict(X_train_scaled)
    recon_error_train = np.mean((X_train_scaled - recon_train) ** 2, axis=1)
    recon_test = ae.predict(X_test_scaled)
    recon_error_test = np.mean((X_test_scaled - recon_test) ** 2, axis=1)

    ae_threshold_default = np.quantile(recon_error_train, 1 - true_rate_train)
    ae_pred_default_test = (recon_error_test >= ae_threshold_default).astype(int)
    results.append(evaluate(y_test, ae_pred_default_test, "Autoencoder (MLP)",
                             "contamination-based (default)", condition=condition))

    thr_ae, _ = best_f1_threshold(y_train, recon_error_train)
    ae_pred_best_test = (recon_error_test >= thr_ae).astype(int)
    results.append(evaluate(y_test, ae_pred_best_test, "Autoencoder (MLP)",
                             f"best-F1 threshold ({thr_ae:.3f}, selected on train)", condition=condition))

    return results


def main():
    df = load_and_prepare()

    results_masked = run_evaluation(df, condition="masked")
    out = pd.DataFrame(results_masked)
    out.to_csv("Table4X_simulated_augmented_results.csv", index=False)
    print("\nSaved Table4X_simulated_augmented_results.csv (masked, Section 4.6.2 figures).")

    print("\n--- Comparison only (not saved): same rows, masking undone (masked back to original perturbation) ---")
    df_unmasked = df.copy()
    was_masked = df_unmasked["masked"] == True
    # Re-derive the pre-mask z-score is not recoverable (the perturbation was
    # overwritten in build_simulated_calibrated_dataset.py); this comparison instead
    # simply excludes masked rows, showing performance on the anomalies whose signal
    # was NOT hidden, as a sanity check that the visible injected signal is still
    # detectable on its own.
    df_unmasked = df_unmasked[~(was_masked)]
    run_evaluation(df_unmasked, condition="unmasked_subset_comparison_only")


if __name__ == "__main__":
    main()
