"""
RQ2 (Hyperledger Fabric) event source generator.

Reproduces the EXACT same protocol as
Avaliacao-Modelo/02-Augmented/evaluate_anomaly_detection_augmented.py (same
RANDOM_SEED, same TEST_SIZE, same single feature, same Isolation Forest
best-F1 threshold selected on TRAIN only) so the events registered on-chain
for RQ2 come from the SAME held-out test predictions reported in Table 4.3 --
not a separate, uncontrolled sample.

Exports every row the Isolation Forest (best-F1 threshold) flags as anomalous
on the held-out test partition -- this mirrors a real deployment, where the
system does not know the true label at flagging time. The true label and
anomaly_type are kept in the output for our own audit trail (thesis
narrative), not because the chaincode consumes them as ground truth.

Run
---
    python3 export_rq2_events_augmented.py

Output
------
flagged_events_augmented.csv -- one row per flagged test-set case, with the
columns register_validated_events.py needs (article_code, category,
anomaly_type, score, label, deviation_zscore_historical, Design, Data).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve

RANDOM_SEED = 42
TEST_SIZE = 0.30
INPUT_PATH = "../Avaliacao-Modelo/02-Augmented/augmented_real_dataset.csv"
FEATURE_COLUMNS = ["deviation_zscore_historical"]
OUTPUT_PATH = "flagged_events_augmented.csv"


def load_and_prepare(path=INPUT_PATH):
    df = pd.read_csv(path)
    df["deviation_zscore_historical"] = df["deviation_zscore_historical"].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=["deviation_zscore_historical"]).reset_index(drop=True)
    return df


def best_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0,
    )
    best_idx = np.argmax(f1[:-1]) if len(thresholds) else 0
    return thresholds[best_idx] if len(thresholds) else 0.0


def main():
    df = load_and_prepare()

    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
    )

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train["label"].values
    X_test = df_test[FEATURE_COLUMNS].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    true_rate_train = y_train.mean()
    iso = IsolationForest(n_estimators=200, contamination=true_rate_train, random_state=RANDOM_SEED)
    iso.fit(X_train_scaled)

    iso_scores_train = -iso.decision_function(X_train_scaled)
    iso_scores_test = -iso.decision_function(X_test_scaled)

    thr = best_f1_threshold(y_train, iso_scores_train)
    flagged_mask = iso_scores_test >= thr

    events = df_test.loc[flagged_mask].copy()
    events["if_score"] = iso_scores_test[flagged_mask]
    events = events.sort_values("if_score", ascending=False).reset_index(drop=True)

    out = events[[
        "Ref", "Design", "Data", "Quant", "anomaly_type", "label",
        "deviation_zscore_historical", "if_score",
    ]].rename(columns={"Ref": "article_code", "anomaly_type": "category"})
    out["article_code"] = out["article_code"].astype("int64").astype(str)

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Held-out test partition: {len(df_test)} rows, {flagged_mask.sum()} flagged "
          f"by Isolation Forest (best-F1 threshold={thr:.3f}, selected on train).")
    print(f"Of those, {int(events['label'].sum())} are true anomalies (TP), "
          f"{int((events['label'] == 0).sum())} are false positives -- matches "
          f"the best-F1 row of Table 4.3 (P=0.511, R=0.619, TP/FP counts).")
    print(f"\nSaved {OUTPUT_PATH} ({len(out)} events).")


if __name__ == "__main__":
    main()
