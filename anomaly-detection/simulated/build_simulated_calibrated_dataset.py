"""
FAIR artefact - builds the FULLY SIMULATED, real-calibrated dataset for Section 4.6
("Simulation-Based Generalisation to Other Hotel Contexts"). This SUPERSEDES the
dataset used by the first version of evaluate_anomaly_detection_simulated.py (which
used the old, fabricated-from-scratch simulate_dataset() with a different, 2-feature
methodology). Rebuilt 15 Aug 2026 to use the SAME single feature and injection logic as
build_augmented_real_dataset.py (Section 4.5), so the two sections are directly
comparable -- same detection feature, same train/test protocol, only the population
and the masking differ.

Why "fully simulated" but "calibrated to the real discrepancy log" (Section 4.6.1)
------------------------------------------------------------------------------------
Section 4.5 evaluates on THIS hotel's real consumption records (real_dataset_v2.csv).
Section 4.6 asks a different question: would detection still work for ANOTHER hotel
group with a similar operational profile? It cannot use this hotel's literal real
values (that would not test transferability), but it should not be arbitrary either.
This script therefore:
  1. Learns real per-article statistics (mean, std, transaction count) from
     real_dataset_v2.csv -- the "calibration" step.
  2. GENERATES a fresh population by sampling new quantities from Normal(mean, std)
     per article, rather than reusing the literal real values. This is the
     "simulated" part: a hypothetical hotel with the same statistical shape.
  3. Injects anomalies with the SAME two mechanisms as Section 4.5 (isolated
     "consumption", systematic "recipe"), at the same rates.
  4. Applies MASKING: overwrites the perturbed signal back to a normal-range value for
     MASKING_RATE of injected anomalies, mirroring "the simulated environment
     deliberately withholds original anomaly signals after correction" (Section
     4.6.1) and the ~95% masking rate empirically found in Section 3.1.3's first
     real-data attempt. This is what Section 4.5's dataset deliberately does NOT do
     (Section 4.5 avoids the masked/circular real labels entirely rather than
     simulating the masking itself).
  5. Computes the SAME feature as Section 4.5: deviation_zscore_historical, using the
     PRE-injection, PRE-masking calibration mean/std as the baseline.

Three-layer comparison this enables (see chat log 15 Aug 2026)
------------------------------------------------------------------
  1. Old synthetic simulation, unrelated to real data (evaluate_anomaly_detection_v2.py,
     superseded): ~0.78-0.92 precision/recall.
  2. Augmented real-data set, real substrate, one real feature, no masking
     (evaluate_anomaly_detection_augmented.py, Section 4.5): ~0.50-0.62.
  3. Simulated, real-calibrated population, SAME feature, WITH masking (this script +
     its companion evaluate_anomaly_detection_simulated_augmented.py, Section 4.6):
     expected to sit at or below layer 2, isolating the specific effect of masking
     from the general effect of "real data is noisier than synthetic."

KNOWN LIMITATION carried over from Section 4.5: only "consumption" and "recipe"
categories are represented (no real inventory/stock-count substrate to calibrate an
"inventory" category against, in either section). Kept symmetric with Section 4.5 on
purpose, for a fair comparison -- do not add a third category here without also
solving it for Section 4.5, or the two will no longer be comparable.

Run
---
    python3 build_simulated_calibrated_dataset.py

Output
------
simulated_calibrated_dataset.csv -- one row per generated transaction, with the same
column shape (Quant, Ref_int, article mean/std, anomaly_type, label,
deviation_zscore_historical) as augmented_real_dataset.csv, plus `masked` (bool).
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
import os  # noqa: E402
# Real substrate used only to LEARN calibration statistics (per-article mean/std).
# Not versioned -- see .gitignore. Point DATA_DIR at the local copy.
REAL_DATASET_PATH = os.environ.get(
    "REAL_DATASET_CSV",
    os.path.join(os.environ.get("DATA_DIR", "./data"), "real_dataset_v2.csv"))
OUTPUT_PATH = "simulated_calibrated_dataset.csv"

CONSUMPTION_ANOMALY_RATE = 0.05
RECIPE_ARTICLE_RATE = 0.05
RECIPE_ROW_COVERAGE = 0.60
CONSUMPTION_PERTURBATION_STD_MULTIPLES = (3.0, 6.0)
RECIPE_PERTURBATION_STD_MULTIPLES = (1.5, 2.5)

# Share of injected anomalies whose perturbed signal is masked back to a normal-range
# value, matching the ~95% masking rate found in Section 3.1.3's first real-data
# attempt and the "deliberately withholds original anomaly signals" design of Section
# 4.6.1.
MASKING_RATE = 0.95


def learn_calibration(path=REAL_DATASET_PATH):
    """Learns real per-article (mean, std, n) from the untouched real data. This is
    the ONLY thing taken from the real log -- no real transaction values are reused."""
    real = pd.read_csv(path)
    stats = real.groupby("Ref")["Quant"].agg(["mean", "std", "count"]).reset_index()
    stats = stats.rename(columns={"mean": "qty_mean", "std": "qty_std", "count": "n"})
    stats = stats[(stats["qty_std"] > 0) & stats["qty_std"].notna() & (stats["n"] >= 2)]
    return stats


def generate_population(stats, rng):
    """Generates a fresh population: for each calibrated article, draws `n` new
    quantities from Normal(qty_mean, qty_std). Same size as the real population per
    article, but none of the individual values are real."""
    rows = []
    for _, row in stats.iterrows():
        n = int(row["n"])
        drawn = rng.normal(loc=row["qty_mean"], scale=row["qty_std"], size=n)
        drawn = np.clip(drawn, a_min=0.001, a_max=None)
        for q in drawn:
            rows.append({
                "Ref": row["Ref"], "Quant": q,
                "qty_mean": row["qty_mean"], "qty_std": row["qty_std"],
            })
    return pd.DataFrame(rows)


def inject_consumption_anomalies(df, rng, rate=CONSUMPTION_ANOMALY_RATE):
    candidates = df.index
    n_flag = int(len(candidates) * rate)
    flagged = rng.choice(candidates, size=n_flag, replace=False)
    direction = rng.choice([-1, 1], size=n_flag)
    magnitude = rng.uniform(*CONSUMPTION_PERTURBATION_STD_MULTIPLES, size=n_flag)
    new_qty = df.loc[flagged, "qty_mean"].values + direction * magnitude * df.loc[flagged, "qty_std"].values
    df.loc[flagged, "Quant"] = np.clip(new_qty, a_min=0.001, a_max=None)
    df.loc[flagged, "anomaly_type"] = "consumption"
    return set(flagged)


def inject_recipe_anomalies(df, rng, article_rate=RECIPE_ARTICLE_RATE,
                             row_coverage=RECIPE_ROW_COVERAGE, exclude=None):
    exclude = exclude or set()
    articles = df["Ref"].unique()
    n_articles = int(len(articles) * article_rate)
    flagged_articles = rng.choice(articles, size=n_articles, replace=False)

    flagged_rows = []
    for article in flagged_articles:
        article_rows = [i for i in df.index[df["Ref"] == article] if i not in exclude]
        if not article_rows:
            continue
        n_rows = max(1, int(len(article_rows) * row_coverage))
        chosen = rng.choice(article_rows, size=min(n_rows, len(article_rows)), replace=False)
        direction = rng.choice([-1, 1])
        magnitude = rng.uniform(*RECIPE_PERTURBATION_STD_MULTIPLES, size=len(chosen))
        new_qty = df.loc[chosen, "qty_mean"].values + direction * magnitude * df.loc[chosen, "qty_std"].values
        df.loc[chosen, "Quant"] = np.clip(new_qty, a_min=0.001, a_max=None)
        df.loc[chosen, "anomaly_type"] = "recipe"
        flagged_rows.extend(chosen)
    return set(flagged_rows)


def apply_masking(df, flagged, rng, masking_rate=MASKING_RATE):
    """Overwrites the perturbed Quant back to a fresh Normal(mean, std) draw for
    masking_rate of the flagged (still-anomalous) rows, hiding the injected signal --
    the label is untouched, exactly as in the real masking problem (Section 3.1.3)."""
    flagged = list(flagged)
    n_mask = int(len(flagged) * masking_rate)
    to_mask = rng.choice(flagged, size=n_mask, replace=False) if n_mask else []
    df["masked"] = False
    if len(to_mask):
        remasked = rng.normal(
            loc=df.loc[to_mask, "qty_mean"].values,
            scale=df.loc[to_mask, "qty_std"].values,
        )
        df.loc[to_mask, "Quant"] = np.clip(remasked, a_min=0.001, a_max=None)
        df.loc[to_mask, "masked"] = True
    return len(to_mask)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    stats = learn_calibration()
    print(f"Calibration learned from {len(stats)} real articles "
          f"(mean/std of real Quant per article; no real values reused).")

    df = generate_population(stats, rng)
    df["anomaly_type"] = "none"
    print(f"Generated {len(df)} simulated transactions.")

    consumption_flagged = inject_consumption_anomalies(df, rng)
    recipe_flagged = inject_recipe_anomalies(df, rng, exclude=consumption_flagged)
    all_flagged = consumption_flagged | recipe_flagged
    df["label"] = (df["anomaly_type"] != "none").astype(int)

    n_masked = apply_masking(df, all_flagged, rng)
    print(f"Injected {len(consumption_flagged)} consumption + {len(recipe_flagged)} recipe "
          f"anomalies ({df['label'].sum()} total, {df['label'].mean():.1%}).")
    print(f"Masked {n_masked} of {len(all_flagged)} injected anomalies "
          f"({MASKING_RATE:.0%} target rate) -- signal deliberately hidden, label kept.")
    print("\nNOTE: 'inventory' category not represented, kept symmetric with Section 4.5.")

    df["deviation_zscore_historical"] = (df["Quant"] - df["qty_mean"]) / df["qty_std"]
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH} ({len(df)} rows, {df['label'].sum()} labelled anomalies).")


if __name__ == "__main__":
    main()
