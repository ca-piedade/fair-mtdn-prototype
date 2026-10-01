"""
Investigation-volume reduction under the MEASURED detector operating point
(project Section 4.8, Table 4.4 — added in response to supervisor review, 23 Sep 2026).

simulation.py assumes perfect detection: every one of the 144 anomalous lots is
flagged and no false positives occur (45.8% reduction). This companion script keeps
the same ground truth (same parameters, seed=42 -> 144 anomalous lots: 93 pure recipe,
49 pure consumption, 2 co-occurring, 27 stale articles) and passes it through a
detector with the primary operating point reported in Section 4.5.2
(temporal hold-out, Isolation Forest): recall = 0.663, precision = 0.454.

Detection layer (Monte Carlo, N_REPS draws, independent RNG so ground truth is unchanged):
  - each anomalous lot is flagged independently with probability RECALL;
  - false positives are drawn among the 1,356 normal lots with a per-lot rate set so that
    expected precision equals PRECISION (E[FP] = E[TP] * (1-P)/P).
Investigation counting (conservative):
  - baseline (no recipe governance): every flag (TP or FP) is one investigation;
  - with recipe governance: detected recipe-driven lots collapse to one ftFlagged review
    per distinct stale article that still has >=1 detected lot; detected pure-consumption
    and co-occurring lots stay individual; every FP stays an individual investigation
    (a false alarm is not attributable to a stale ficha).
Run: python3 simulation_measured_detector.py
"""
import numpy as np

N_MONTHS, N_ARTICLES, LOTS_PER_MONTH = 6, 300, 250
STALE_RECIPE_RATE, CONSUMPTION_ANOMALY_RATE, STALE_RECIPE_TRIGGER_RATE = 0.10, 0.04, 0.70
RECALL, PRECISION = 0.663, 0.454
N_REPS = 10_000

def ground_truth(stale_rate=STALE_RECIPE_RATE, trigger_rate=STALE_RECIPE_TRIGGER_RATE, seed=42):
    rng = np.random.default_rng(seed=seed)
    stale = rng.random(N_ARTICLES) < stale_rate
    n = LOTS_PER_MONTH * N_MONTHS
    art = rng.integers(0, N_ARTICLES, size=n)
    cons = rng.random(n) < CONSUMPTION_ANOMALY_RATE
    rec = stale[art] & (rng.random(n) < trigger_rate)
    return art, cons, rec

def reduction(art, cons, rec, recall, precision, reps, seed=2026):
    rng = np.random.default_rng(seed)
    anom = cons | rec
    n_anom, n_neg = int(anom.sum()), int((~anom).sum())
    fp_rate = (recall * n_anom * (1 - precision) / precision) / n_neg if precision < 1 else 0.0
    rec_mask = rec  # recipe-driven (pure + co-occurring)
    out = []
    for _ in range(reps):
        det = anom & (rng.random(anom.size) < recall)
        fp = int((rng.random(n_neg) < fp_rate).sum())
        tp = int(det.sum())
        base = tp + fp
        reviews = len(np.unique(art[det & rec_mask]))
        indiv = int((det & cons).sum())          # pure consumption + co-occurring
        gov = reviews + indiv + fp
        out.append((tp, fp, base, reviews, indiv, gov, 1 - gov / base))
    return np.array(out)

if __name__ == "__main__":
    art, cons, rec = ground_truth()
    anom = cons | rec
    print(f"Ground truth: {anom.sum()} anomalous lots; pure recipe {(rec & ~cons).sum()}, "
          f"pure consumption {(cons & ~rec).sum()}, co-occurring {(rec & cons).sum()}, "
          f"stale articles with anomalous lots {len(np.unique(art[rec]))}")
    for label, r, p in [("Perfect detection (Table 4.4 baseline)", 1.0, 1.0),
                        ("Measured recall only (no false positives)", RECALL, 1.0),
                        ("Measured operating point (R=0.663, P=0.454)", RECALL, PRECISION)]:
        a = reduction(art, cons, rec, r, p, 1 if (r == 1 and p == 1) else N_REPS)
        m = a.mean(0); lo, hi = np.percentile(a[:, 6], [2.5, 97.5])
        print(f"\n{label}")
        print(f"  flags TP {m[0]:.1f}  FP {m[1]:.1f}  -> baseline investigations {m[2]:.1f}")
        print(f"  with governance: {m[3]:.1f} ftFlagged reviews + {m[4]:.1f} individual + {m[1]:.1f} FP = {m[5]:.1f}")
        print(f"  reduction mean {m[6]*100:.1f}%  (95% MC interval {lo*100:.1f}%-{hi*100:.1f}%)")
    print("\nSensitivity grid at the measured operating point (mean reduction, %):")
    print("stale\\trigger   0.50   0.70   0.90")
    for s in (0.05, 0.10, 0.15):
        row = []
        for t in (0.50, 0.70, 0.90):
            a = reduction(*ground_truth(s, t), RECALL, PRECISION, 2000)
            row.append(a[:, 6].mean() * 100)
        print(f"  {s:.2f}         " + "  ".join(f"{v:5.1f}" for v in row))
