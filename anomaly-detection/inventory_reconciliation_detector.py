"""
FAIR artefact - inventory-reconciliation anomaly detector (Section 4.5.3 of the
dissertation). Reconstructed on 26 Sep 2026 from the raw physical-count files so that
the Section 4.5.3 figures can be reproduced; the original exploratory run (24 Aug 2026)
was not saved as a script.

Data (not included in the repository - operational data of the host organisation)
------------------------------------------------------------------------------------
48 "Análise de Faltas e Excessos" exports (one per outlet and count date, May-June
2026), each listing per article: Referencia, Designacao, Armazem, Unid, Inv fisico
(physical count), Inv contabilistico (book count), Faltas/Valor f (shortages),
Excessos/Valor e (surpluses). Point DATA_DIR at the folder that holds them.

Design (same convention as build_augmented_real_dataset.py)
-----------------------------------------------------------
1. Load all 48 files -> 19,104 line-level observations, ~1,500 distinct articles;
   961 observations have a non-zero physical-versus-book difference.
2. Detection variable: relative discrepancy rel = (physical - book) / book, which
   requires a non-zero book count.
3. Eligibility: an article needs its own real baseline, so only articles with at least
   three counts (book count != 0) and a historical standard deviation of rel above
   0.001 are kept -> 311 articles, 1,382 observations.
4. Controlled injection, calibrated in units of each article's own real standard
   deviation of rel (pre-injection mean/std, so injected rows do not contaminate
   their own baseline):
     - isolated, one-off discrepancies: 5% of eligible observations, 3-6 sigma;
     - systematic, same-direction discrepancies: 5% of eligible articles, applied to
       60% of each selected article's remaining observations, 1.5-2.5 sigma.
   Labels mark injected observations only; real discrepancies stay in the substrate
   as normal observations.
5. Feature: z = (rel_after_injection - article mean) / article std.
6. Isolation Forest (200 trees), 10 stratified 70/30 train/test splits (seeds 0-9);
   threshold chosen as best-F1 on the TRAIN partition only (contamination-based
   threshold also reported). Metrics on the held-out partition.

Caveat reported in the dissertation: the injection is defined in units of the same
variable the detector scores, so recovery is close to circular by construction.

Run
---
    pip install --break-system-packages pandas numpy scikit-learn openpyxl
    python3 inventory_reconciliation_detector.py --data-dir <path to the 48 files>
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
ISOLATED_RATE = 0.05           # fraction of eligible observations
SYSTEMATIC_ARTICLE_RATE = 0.05  # fraction of eligible articles
SYSTEMATIC_ROW_COVERAGE = 0.60  # fraction of a selected article's observations
ISOLATED_SIGMA = (3.0, 6.0)
SYSTEMATIC_SIGMA = (1.5, 2.5)
MIN_COUNTS = 3
MIN_STD = 0.001
N_SPLITS = 10


def load_counts(data_dir):
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.xlsx"))):
        raw = pd.read_excel(path, header=None)
        header_row = raw.index[raw.apply(lambda r: r.astype(str).str.contains("Referencia").any(), axis=1)][0]
        df = raw.iloc[header_row + 1:].copy()
        df.columns = [str(c) for c in raw.iloc[header_row]]
        df = df.loc[:, [c for c in df.columns if c != "nan"]]
        df = df[df["Referencia"].notna()]
        df["file"] = os.path.basename(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    for col in ["Inv fisico", "Inv contabilistico"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    data["Referencia"] = data["Referencia"].astype(str)
    data["diff"] = data["Inv fisico"] - data["Inv contabilistico"]
    return data


def eligible_set(data):
    book = data[data["Inv contabilistico"] != 0].copy()
    book["rel"] = book["diff"] / book["Inv contabilistico"]
    grp = book.groupby("Referencia")["rel"]
    book["n"] = grp.transform("size")
    book["mean"] = grp.transform("mean")
    book["std"] = grp.transform("std")
    return book[(book["n"] >= MIN_COUNTS) & (book["std"] > MIN_STD)].reset_index(drop=True)


def inject(elig, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    df = elig.copy()
    df["label"] = 0
    df["pattern"] = "none"
    df["rel_injected"] = df["rel"]

    n_iso = int(len(df) * ISOLATED_RATE)
    iso_idx = rng.choice(df.index, size=n_iso, replace=False)
    direction = rng.choice([-1, 1], size=n_iso)
    mag = rng.uniform(*ISOLATED_SIGMA, size=n_iso)
    df.loc[iso_idx, "rel_injected"] = df.loc[iso_idx, "mean"].values + direction * mag * df.loc[iso_idx, "std"].values
    df.loc[iso_idx, ["label", "pattern"]] = [1, "isolated"]

    iso_set = set(iso_idx)
    articles = df["Referencia"].unique()
    n_art = int(len(articles) * SYSTEMATIC_ARTICLE_RATE)
    for art in rng.choice(articles, size=n_art, replace=False):
        rows = [i for i in df.index[df["Referencia"] == art] if i not in iso_set]
        if not rows:
            continue
        n_rows = max(1, int(len(rows) * SYSTEMATIC_ROW_COVERAGE))
        chosen = rng.choice(rows, size=min(n_rows, len(rows)), replace=False)
        d = rng.choice([-1, 1])
        m = rng.uniform(*SYSTEMATIC_SIGMA, size=len(chosen))
        df.loc[chosen, "rel_injected"] = df.loc[chosen, "mean"].values + d * m * df.loc[chosen, "std"].values
        df.loc[chosen, ["label", "pattern"]] = [1, "systematic"]

    df["z"] = (df["rel_injected"] - df["mean"]) / df["std"]
    return df


def evaluate(df):
    X = df[["z"]].values
    y = df["label"].values
    rows = []
    for split in range(N_SPLITS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split)
        model = IsolationForest(n_estimators=200, contamination=ytr.mean(), random_state=split).fit(Xtr)
        s_tr, s_te = -model.score_samples(Xtr), -model.score_samples(Xte)
        p, r, t = precision_recall_curve(ytr, s_tr)
        f = 2 * p * r / (p + r + 1e-12)
        thr = t[np.argmax(f[:-1])]
        for name, pred in (("best-F1 (train)", (s_te >= thr).astype(int)),
                           ("contamination (default)", (model.predict(Xte) == -1).astype(int))):
            rows.append({"split": split, "threshold": name,
                         "precision": precision_score(yte, pred, zero_division=0),
                         "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
                         "pr_auc": average_precision_score(yte, s_te)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="inventory_reconciliation_results.csv")
    args = ap.parse_args()

    data = load_counts(args.data_dir)
    print(f"Files: {data['file'].nunique()} | observations: {len(data)} | articles: {data['Referencia'].nunique()}")
    print(f"Non-zero physical-versus-book differences: {(data['diff'] != 0).sum()} ({(data['diff'] != 0).mean():.1%})")

    elig = eligible_set(data)
    print(f"Eligible (book != 0, >= {MIN_COUNTS} counts, std(rel) > {MIN_STD}): "
          f"{elig['Referencia'].nunique()} articles, {len(elig)} observations "
          f"({(elig['diff'] != 0).sum()} with a real non-zero difference, kept as normal)")

    df = inject(elig)
    print(f"Injected: {df['label'].sum()} ({(df['pattern'] == 'isolated').sum()} isolated 3-6 sigma, "
          f"{(df['pattern'] == 'systematic').sum()} systematic 1.5-2.5 sigma), "
          f"base rate {df['label'].mean():.3f}")

    res = evaluate(df)
    res.to_csv(args.out, index=False)
    summary = res.groupby("threshold")[["precision", "recall", "f1", "pr_auc"]].agg(["mean", "std"]).round(3)
    print(f"\nIsolation Forest, {N_SPLITS} stratified 70/30 splits (held-out partition):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
