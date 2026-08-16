"""
FAIR artefact - builds the AUGMENTED REAL-DATA evaluation set described in Section
3.1.3 / Section 4.5 of the dissertation. This is the primary evaluation set for RQ1
(Table 4.3); do not confuse it with the fully synthetic simulate_dataset() used by
evaluate_anomaly_detection_v2.py (superseded) or the masked generalisation set used
by evaluate_anomaly_detection_simulated.py (Section 4.6).

Why this exists (see chat log 15 Aug 2026 for the full trail)
---------------------------------------------------------------
Two direct evaluation attempts on the raw real discrepancy log (real_dataset.csv /
real_dataset_v2.csv, in real-data-exploration/) both failed for diagnosed reasons:
  - v1: deviation_pct computed from the Diferencas report is circular (0 by
    construction for every unflagged row) -> precision=recall=1.000, invalid.
  - v2: single-source statistics (qty_zscore, ref_novelty) give near-base-rate
    performance (0.219) -> the correction is a cross-system disagreement, not a
    single-source statistical outlier (confirmed empirically, matches the argument
    in Section 2.3.1).
  - No per-recipe "last validated" date exists in either Fichas Técnicas export
    (98b...Detalhado.xlsx, FichasTecnicas_Consolidado.xlsx) -- only a single
    batch-export timestamp, identical across all recipes. recipe_age_days as
    originally used in the synthetic simulation cannot be computed from real data.
  - No POS sales-by-dish export is available for banquets, so a full
    theoretical-consumption-vs-actual-consumption reconciliation (which would need
    dishes-sold x ficha-técnica quantities) cannot be built for the whole population.

Consequently, this script follows the "controlled augmentation" methodology already
described in Section 3.1.3: it does NOT trust the old (masked) real labels, and does
NOT fabricate lots from scratch (unlike simulate_dataset()). Instead it takes the REAL
population of consumption lines (real_dataset_v2.csv, 15,173 real rows, real articles,
real quantities) as the substrate, and injects controlled field-level perturbations
into a subset of those real rows, calibrated against REAL per-article baselines
(qty_mean, qty_std -- already computed from the untouched real distribution in
real_dataset_v2.csv):
  - "consumption" anomalies: an ISOLATED, one-off perturbation of Quant on a randomly
    selected subset of individual transaction rows, beyond tolerance (3-6 real std
    devs of that article). Represents a single unusual consumption event.
  - "recipe" anomalies: a SYSTEMATIC perturbation applied consistently (same
    direction, smaller magnitude, 1.5-2.5 real std devs) across most of a flagged
    article's real transactions in the window. Represents a persistent
    recipe-configuration bias that affects every use of that article, rather than a
    one-off event.

CORRECTION LOGGED 15 Aug 2026 -- read before reusing this design elsewhere (e.g. for
Section 4.6's simulated environment, which has the same class of risk)
------------------------------------------------------------------------------------
The first version of this script perturbed "recipe" anomalies relative to
theoretical_qtd_ref (real per-dose Fichas Técnicas quantity, from
FichasTecnicas_Consolidado.xlsx, Detalhe sheet). That was WRONG: theoretical_qtd_ref is
a per-dose quantity, while Quant is a requisition-level quantity covering an unknown
number of doses. On real, untouched rows, Quant / theoretical_qtd_ref ranges from 0 to
468 (mean 8.8) -- the two are not at a comparable scale without a dish-level sales
count to convert doses to requisition volume (which is not available; see
build_augmented_real_dataset.py chat history, 15 Aug 2026, for the banquets gap that
ruled this out). Using theoretical_qtd_ref as a magnitude-comparison feature buried the
injected signal in this scale noise and produced near-random detection performance
(precision/recall ~0.30 with correlation 0.08) that reflected the bug, not a real
finding. theoretical_qtd_ref is retained in the output only as a documented,
non-detection reference column (dividing by a per-article constant before z-scoring is
mathematically identical to the historical z-score anyway, so it adds no independent
signal without dose-volume data). Both anomaly types are now grounded exclusively in
qty_mean/qty_std, distinguished only by injection pattern (isolated vs systematic), not
by a second detection feature.

KNOWN LIMITATION -- please read before treating Table 4.3 as final
----------------------------------------------------------------------
This covers "consumption-pattern" and "recipe-configuration" with a real substrate.
It does NOT cover "inventory" as a distinct category: the available real exports are
consumption/requisition records (Listagem de Consumos), not physical stock-count data,
so there is no real substrate for inventory-count discrepancies specifically. If
Section 4.5's text claims all three categories are grounded in this augmented set,
either source real inventory-count data, or adjust the text to state that the
inventory category is represented in the simulated generalisation environment
(Section 4.6) rather than in this primary augmented set. This script only produces
"consumption" and "recipe" labelled anomalies.

Run
---
    pip install --break-system-packages pandas numpy openpyxl
    python3 build_augmented_real_dataset.py

Output
------
augmented_real_dataset.csv -- one row per real consumption line, with:
  - all original real_dataset_v2.csv columns (Quant is OVERWRITTEN for injected rows;
    Quant_original preserves the untouched real value for audit)
  - theoretical_qtd_ref: real per-dose reference from Fichas Técnicas (NaN if the
    article is not used as an ingredient in any recipe). Documented reference only,
    NOT used as a detection feature (see correction note above).
  - deviation_zscore_historical: (Quant - qty_mean) / qty_std, using the PRE-injection
    qty_mean/qty_std (so injected anomalies do not contaminate their own baseline).
    This is the primary detection feature.
  - anomaly_type: "none" / "consumption" / "recipe" (fresh labels, independent of the
    old, masked real anomaly_type column, which is dropped)
  - label: 1 if anomaly_type != "none"
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
REAL_DATASET_PATH = "../01-Dados-Reais-Fonte/real-data-exploration/real_dataset_v2.csv"
FICHAS_PATH = "../01-Dados-Reais-Fonte/ficha-tecnica-source/FichasTecnicas_Consolidado.xlsx"
OUTPUT_PATH = "augmented_real_dataset.csv"

# Target injection rate, matching the "approximately five per cent of records per
# category" design already stated in Section 3.1.3.
CONSUMPTION_ANOMALY_RATE = 0.05   # fraction of individual ROWS flagged (isolated events)
RECIPE_ARTICLE_RATE = 0.05        # fraction of ARTICLES flagged (systematic bias)
RECIPE_ROW_COVERAGE = 0.60        # fraction of a flagged article's rows that inherit the bias

# Perturbation magnitude, in multiples of the article's own real historical std dev.
# "consumption" is isolated and large (clears tolerance easily on its own row).
# "recipe" is smaller per row but systematic across many rows of the same article,
# mirroring a persistent configuration bias rather than a one-off event.
CONSUMPTION_PERTURBATION_STD_MULTIPLES = (3.0, 6.0)
RECIPE_PERTURBATION_STD_MULTIPLES = (1.5, 2.5)


def load_real_substrate(path=REAL_DATASET_PATH):
    df = pd.read_csv(path)
    # Drop the old, masked/circular labels -- this script builds fresh ground truth.
    df = df.drop(columns=["label", "anomaly_type"], errors="ignore")
    df["Ref_int"] = df["Ref"].astype("Int64")
    # Preserve the untouched real quantity for audit before any perturbation.
    df["Quant_original"] = df["Quant"]
    return df


def load_theoretical_reference(path=FICHAS_PATH):
    """Real per-dose quantity reference per article: median Qtd across all recipes
    that use this Codigo_Artigo as an ingredient (Detalhe sheet). This is the closest
    real reference available without a dish-level sales export to weight by portions
    actually sold."""
    detalhe = pd.read_excel(path, sheet_name="Detalhe")
    detalhe["Codigo_Artigo"] = detalhe["Codigo_Artigo"].astype("Int64")
    ref = detalhe.groupby("Codigo_Artigo")["Qtd"].median()
    ref.name = "theoretical_qtd_ref"
    return ref


def inject_consumption_anomalies(df, rng, rate=CONSUMPTION_ANOMALY_RATE):
    """Perturbs Quant relative to the article's own REAL historical mean/std,
    beyond tolerance. Uses qty_mean/qty_std as already computed in
    real_dataset_v2.csv from the untouched real distribution -- these are NOT
    recomputed after injection, so injected rows do not contaminate their own
    baseline."""
    eligible = df["qty_std"].notna() & (df["qty_std"] > 0)
    candidates = df.index[eligible]
    n_flag = int(len(candidates) * rate)
    flagged = rng.choice(candidates, size=n_flag, replace=False)

    direction = rng.choice([-1, 1], size=n_flag)
    magnitude = rng.uniform(*CONSUMPTION_PERTURBATION_STD_MULTIPLES, size=n_flag)
    new_qty = df.loc[flagged, "qty_mean"].values + direction * magnitude * df.loc[flagged, "qty_std"].values
    new_qty = np.clip(new_qty, a_min=0.001, a_max=None)  # quantities cannot be negative

    df.loc[flagged, "Quant"] = new_qty
    df.loc[flagged, "anomaly_type"] = "consumption"
    return set(flagged)


def inject_recipe_anomalies(df, rng, article_rate=RECIPE_ARTICLE_RATE,
                             row_coverage=RECIPE_ROW_COVERAGE, exclude=None):
    """Systematic perturbation: flags a subset of ARTICLES (not individual rows) as
    having a recipe-configuration bias, then applies a consistent-direction, smaller
    perturbation to most of that article's real transactions in the window, relative
    to the article's OWN real historical mean/std (qty_mean/qty_std) -- the same
    well-scaled baseline used for "consumption" anomalies. theoretical_qtd_ref is
    intentionally NOT used here (see correction note in the module docstring)."""
    exclude = exclude or set()
    eligible_rows = df["qty_std"].notna() & (df["qty_std"] > 0)
    eligible_articles = df.loc[eligible_rows, "Ref_int"].dropna().unique()
    n_articles = int(len(eligible_articles) * article_rate)
    flagged_articles = rng.choice(eligible_articles, size=n_articles, replace=False)

    flagged_rows = []
    for article in flagged_articles:
        article_rows = df.index[eligible_rows & (df["Ref_int"] == article)]
        article_rows = [i for i in article_rows if i not in exclude]
        if not article_rows:
            continue
        n_rows = max(1, int(len(article_rows) * row_coverage))
        chosen = rng.choice(article_rows, size=min(n_rows, len(article_rows)), replace=False)
        # Same direction for every row of this article: a systematic bias, not noise.
        direction = rng.choice([-1, 1])
        magnitude = rng.uniform(*RECIPE_PERTURBATION_STD_MULTIPLES, size=len(chosen))
        new_qty = df.loc[chosen, "qty_mean"].values + direction * magnitude * df.loc[chosen, "qty_std"].values
        new_qty = np.clip(new_qty, a_min=0.001, a_max=None)
        df.loc[chosen, "Quant"] = new_qty
        df.loc[chosen, "anomaly_type"] = "recipe"
        flagged_rows.extend(chosen)

    return set(flagged_rows)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = load_real_substrate()
    theoretical_ref = load_theoretical_reference()
    df = df.merge(theoretical_ref, left_on="Ref_int", right_index=True, how="left")

    n_with_theoretical = df["theoretical_qtd_ref"].notna().sum()
    print(f"Real substrate: {len(df)} rows, {df['Ref_int'].nunique()} unique articles.")
    print(f"  {n_with_theoretical} rows ({n_with_theoretical/len(df):.1%}) have a real "
          f"theoretical Fichas Técnicas quantity reference.")

    df["anomaly_type"] = "none"

    consumption_flagged = inject_consumption_anomalies(df, rng)
    recipe_flagged = inject_recipe_anomalies(df, rng, exclude=consumption_flagged)

    df["label"] = (df["anomaly_type"] != "none").astype(int)

    print(f"\nInjected {len(consumption_flagged)} consumption anomalies "
          f"(isolated, target rate {CONSUMPTION_ANOMALY_RATE:.0%} of eligible rows).")
    print(f"Injected {len(recipe_flagged)} recipe anomalies "
          f"(systematic, target rate {RECIPE_ARTICLE_RATE:.0%} of eligible articles, "
          f"{RECIPE_ROW_COVERAGE:.0%} row coverage per article, excluding consumption-flagged).")
    print(f"Total: {df['label'].sum()} anomalies of {len(df)} rows ({df['label'].mean():.1%}).")
    print("\nNOTE: 'inventory' category is NOT represented in this augmented set -- no real "
          "stock-count substrate is available (see script docstring).")

    # ---------------- detection feature, computed AFTER injection, using PRE-injection baseline ----------------
    # theoretical_qtd_ref is kept as a documented reference column only (see correction
    # note in the module docstring for why it is not used as a detection feature).
    df["deviation_zscore_historical"] = (df["Quant"] - df["qty_mean"]) / df["qty_std"]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH} ({len(df)} rows, {df['label'].sum()} labelled anomalies).")


if __name__ == "__main__":
    main()
