"""
FAIR artefact - RQ1 PRIMARY evaluation on the augmented real-data evaluation set
(Section 3.1.3 / Section 4.5). This is what feeds Table 4.3, and downstream, the
Resumo, Abstract, Section 5.1.1, Table 5.1, and Section 5.5 placeholders.

Run build_augmented_real_dataset.py FIRST to produce augmented_real_dataset.csv, then
run this script. Do not confuse this with:
  - evaluate_anomaly_detection_v2.py (superseded -- fully synthetic simulate_dataset(),
    not the real-data-grounded set the project text now describes).
  - evaluate_anomaly_detection_simulated.py (Section 4.6, other-hotel generalisation,
    fully simulated with masking, not real-data-grounded by design).

Train/test protocol
--------------------
Same corrected, leakage-free protocol used throughout: split first, feature-selection
ablation and threshold selection (contamination-based and best-F1) on TRAIN only,
final precision/recall/F1 reported on the held-out TEST partition only.

Feature
-------
deviation_zscore_historical: (Quant - qty_mean) / qty_std, from the article's own real
    historical consumption distribution (already computed in real_dataset_v2.csv from
    the untouched real data). This is the ONLY detection feature. An earlier version
    also used deviation_pct_theoretical (deviation from the Fichas Técnicas per-dose
    quantity) as a second feature; that was dropped after testing showed
    theoretical_qtd_ref is not at the same scale as Quant (a per-dose quantity vs a
    requisition-level quantity), which buried the injected signal in scale noise and
    produced near-random results (~0.30 precision/recall) that reflected the bug, not
    a real finding. See build_augmented_real_dataset.py's module docstring
    ("CORRECTION LOGGED 15 Aug 2026") for the full diagnosis -- the same risk applies
    to any future feature engineering for Section 4.6's simulated environment if a
    theoretical/per-dose reference is used without a dose-volume link.

"consumption" and "recipe" anomalies are both grounded in this same feature; they are
distinguished only by injection pattern in build_augmented_real_dataset.py (isolated
one-off vs systematic per-article bias), not by separate detection features.

KNOWN LIMITATION carried over from build_augmented_real_dataset.py: this set has NO
"inventory" category (no real stock-count substrate was available). See that script's
docstring for the full rationale and the two failed real-data attempts that preceded
this approach.

Run
---
    python3 build_augmented_real_dataset.py
    python3 evaluate_anomaly_detection_augmented.py

Outputs
-------
- Console: precision/recall/F1/confusion matrix for both models, at two thresholds,
  held-out test partition only.
- RQ1_augmented_results.csv: the figures for Table 4.3.
- RQ1_augmented_shap_summary.txt: sample SHAP explanations for flagged test-set rows.
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
INPUT_PATH = "augmented_real_dataset.csv"

FEATURE_COLUMNS = [
    "deviation_zscore_historical",
]


def load_and_prepare(path=INPUT_PATH):
    df = pd.read_csv(path)
    # A handful of rows can have qty_std == 0 (article seen only once in the period);
    # these produce inf/nan z-scores and must not reach the model.
    df["deviation_zscore_historical"] = df["deviation_zscore_historical"].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=["deviation_zscore_historical"]).reset_index(drop=True)
    return df


def feature_correlation_check(df):
    print("Feature correlation with label (ablation check, TRAIN only):")
    for col in FEATURE_COLUMNS:
        corr = np.corrcoef(df[col], df["label"])[0, 1]
        print(f"  {col:32s} corr={corr:+.3f}")
    print()


def best_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0,
    )
    best_idx = np.argmax(f1[:-1]) if len(thresholds) else 0
    return thresholds[best_idx] if len(thresholds) else 0.0, f1[best_idx] if len(f1) else 0.0


def evaluate(y_true, y_pred, model_name, threshold_name, partition="test"):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n[{model_name} | {threshold_name} | {partition} partition]")
    print(f"  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {
        "model": model_name, "threshold": threshold_name, "partition": partition,
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def main():
    df = load_and_prepare()

    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
    )
    print(f"Augmented real-data set: {len(df)} rows, {df['label'].sum()} anomalies "
          f"({df['label'].mean():.1%}). By type:")
    print(df["anomaly_type"].value_counts())
    print(f"\nSplit: {len(df_train)} train ({df_train['label'].sum()} anomalies), "
          f"{len(df_test)} test ({df_test['label'].sum()} anomalies).\n")

    feature_correlation_check(df_train)

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train["label"].values
    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test["label"].values
    true_rate_train = y_train.mean()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ---------------- Isolation Forest ----------------
    iso = IsolationForest(n_estimators=200, contamination=true_rate_train, random_state=RANDOM_SEED)
    iso.fit(X_train_scaled)
    iso_scores_train = -iso.decision_function(X_train_scaled)
    iso_scores_test = -iso.decision_function(X_test_scaled)

    iso_pred_default_test = (iso.predict(X_test_scaled) == -1).astype(int)
    results.append(evaluate(y_test, iso_pred_default_test, "Isolation Forest", "contamination-based (default)"))

    thr, _ = best_f1_threshold(y_train, iso_scores_train)
    iso_pred_best_test = (iso_scores_test >= thr).astype(int)
    results.append(evaluate(y_test, iso_pred_best_test, "Isolation Forest",
                             f"best-F1 threshold ({thr:.3f}, selected on train)"))

    # ---------------- Autoencoder-style baseline (MLP reconstruction error) ----------------
    # With a single input feature there is no meaningful "bottleneck" (compression
    # requires more input dimensions than the middle layer, which does not apply
    # here). This is a small nonlinear reconstruction model over the 1-D distribution,
    # not a true undercomplete autoencoder -- reported for comparability with
    # Isolation Forest per the project's Objective 1, not because compression is
    # architecturally meaningful at 1 feature.
    ae = MLPRegressor(hidden_layer_sizes=(3,), activation="relu", max_iter=1000, random_state=RANDOM_SEED)
    ae.fit(X_train_scaled, X_train_scaled)

    recon_train = ae.predict(X_train_scaled)
    recon_error_train = np.mean((X_train_scaled - recon_train) ** 2, axis=1)
    recon_test = ae.predict(X_test_scaled)
    recon_error_test = np.mean((X_test_scaled - recon_test) ** 2, axis=1)

    ae_threshold_default = np.quantile(recon_error_train, 1 - true_rate_train)
    ae_pred_default_test = (recon_error_test >= ae_threshold_default).astype(int)
    results.append(evaluate(y_test, ae_pred_default_test, "Autoencoder (MLP)", "contamination-based (default)"))

    thr_ae, _ = best_f1_threshold(y_train, recon_error_train)
    ae_pred_best_test = (recon_error_test >= thr_ae).astype(int)
    results.append(evaluate(y_test, ae_pred_best_test, "Autoencoder (MLP)",
                             f"best-F1 threshold ({thr_ae:.3f}, selected on train)"))

    out = pd.DataFrame(results)
    out.to_csv("RQ1_augmented_results.csv", index=False)
    print("\nSaved RQ1_augmented_results.csv (test-partition figures for Table 4.3).")

    try:
        import shap
        explainer = shap.TreeExplainer(iso)
        sample_idx = np.where(iso_pred_best_test == 1)[0][:5]
        shap_values = explainer.shap_values(X_test_scaled[sample_idx])
        with open("RQ1_augmented_shap_summary.txt", "w") as f:
            for i, idx in enumerate(sample_idx):
                row = df_test.iloc[idx]
                f.write(f"Test-set row {idx} (Ref={row['Ref']}, {row.get('Design','')}, "
                        f"true label={y_test[idx]}, anomaly_type={row['anomaly_type']}):\n")
                for feat, val in zip(FEATURE_COLUMNS, shap_values[i]):
                    f.write(f"    {feat}: SHAP={val:.3f}\n")
                f.write("\n")
        print("Saved RQ1_augmented_shap_summary.txt.")
    except Exception as e:
        print(f"SHAP explanation skipped ({e}).")


if __name__ == "__main__":
    main()
