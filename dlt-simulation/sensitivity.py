"""
Sensitivity analysis for the ICDLT 2026 paper / project Section 4.8.3,
Table 4.5 — "Sensitivity of investigation-volume reduction to stale-recipe
rate and trigger rate".

Companion to simulation.py (same core logic, same seed=42, same fixed
parameters for everything except the two varied ones), looped over a 3x3
grid: stale-recipe rate in {5%, 10%, 15%} and trigger rate in {50%, 70%,
90%}. The centre cell (10%, 70%) reproduces the base-case 45.8% reported
in Table 4.4 / Section 4.8.2.

Run: python3 sensitivity.py

Verified 15 Aug 2026 and 1 Oct 2026: reproduces Table 4.5 exactly (26.1%-62.0% range).
"""

import numpy as np

N_MONTHS = 6
N_ARTICLES = 300
LOTS_PER_MONTH = 250
CONSUMPTION_ANOMALY_RATE = 0.04
RANDOM_SEED = 42

GRID_STALE_RATE = [0.05, 0.10, 0.15]
GRID_TRIGGER_RATE = [0.50, 0.70, 0.90]


def run(stale_rate, trigger_rate, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed=seed)
    articles = np.arange(N_ARTICLES)
    stale_recipe = rng.random(N_ARTICLES) < stale_rate

    total_lots = LOTS_PER_MONTH * N_MONTHS
    lot_articles = rng.integers(0, N_ARTICLES, size=total_lots)
    lot_stale = stale_recipe[lot_articles]

    consumption_anomaly = rng.random(total_lots) < CONSUMPTION_ANOMALY_RATE
    recipe_driven_anomaly = lot_stale & (rng.random(total_lots) < trigger_rate)

    flagged = consumption_anomaly | recipe_driven_anomaly
    pure_consumption = flagged & consumption_anomaly & ~recipe_driven_anomaly
    pure_recipe = flagged & recipe_driven_anomaly & ~consumption_anomaly
    co_occurring = flagged & recipe_driven_anomaly & consumption_anomaly

    baseline_investigations = int(flagged.sum())

    recipe_driven_lots_mask = pure_recipe | co_occurring
    affected_articles = np.unique(lot_articles[recipe_driven_lots_mask])
    n_ftFlagged_reviews = len(affected_articles)

    with_governance_investigations = (
        int(pure_consumption.sum()) + int(co_occurring.sum()) + n_ftFlagged_reviews
    )
    reduction = 1 - (with_governance_investigations / baseline_investigations)
    return baseline_investigations, with_governance_investigations, reduction


def main():
    print("=" * 78)
    print("TABLE 4.2 - Sensitivity of investigation-volume reduction to")
    print("stale-recipe rate and trigger rate (3x3 grid, seed=42)")
    print("=" * 78)
    print(f"{'Stale rate':>12} {'Trigger rate':>14} {'Baseline':>10} {'Governance':>12} {'Reduction':>11}")
    print("-" * 78)
    for stale_rate in GRID_STALE_RATE:
        for trigger_rate in GRID_TRIGGER_RATE:
            baseline, governance, reduction = run(stale_rate, trigger_rate)
            base_case = " *" if (stale_rate == 0.10 and trigger_rate == 0.70) else ""
            print(f"{stale_rate*100:11.0f}% {trigger_rate*100:13.0f}% {baseline:10d} "
                  f"{governance:12d} {reduction*100:10.1f}%{base_case}")
    print("-" * 78)
    print("* Stale-recipe rate 10%, trigger rate 70%: the parameter combination")
    print("  assumed in Section 4.8.1 and used to produce Table 4.4.")


if __name__ == "__main__":
    main()
