# Prototype Code — MTDN Dissertation "Enhancing Accuracy and Trust in Hospitality F&B Reporting: A Hybrid AI-Blockchain Artefact"

Supplementary code referenced in **Annex D** of the dissertation (Carla Alexandra Viveiros Piedade, ISCTE MTDN). This repository contains the two functional components of the artefact evaluated in Chapter 4 (Demonstration & Evaluation) and in the companion ICDLT 2026 conference paper.

## Repository structure

```
prototype/
├── anomaly-detection/          AI layer — RQ1 evaluation (Isolation Forest vs. Autoencoder)
├── chain-event-registration/   RQ2 — registers RQ1's flagged test-set events on Fabric
├── dlt-simulation/              DLT layer — throughput/volume simulation (ICDLT 2026 paper, Section 5)
├── chaincode/fnb-trust/         Hyperledger Fabric chaincode (StockLot + RecipeGovernance contracts)
└── replay-harness/              Node.js harness to replay simulated events as real Fabric transactions
```

### `anomaly-detection/`

The primary RQ1 evaluation (Table 4.3; Sections 3.1.3 / 4.5) runs on an **augmented real-data evaluation set**: the real substrate is 3,705 confirmed POS–ERP consumption discrepancies (real articles, real quantities), into which controlled, calibrated field-level perturbations are injected (isolated "consumption" anomalies and systematic per-article "recipe" anomalies), each grounded in that article's own real historical mean/std. Two earlier direct attempts to evaluate on the raw real discrepancy log were tried and discarded (circular labels; near-base-rate single-source features) — this augmented approach is what is actually reported in the dissertation.

Pipeline:
```bash
pip install scikit-learn pandas numpy shap
python3 build_augmented_real_dataset.py        # builds the real+injected evaluation set
python3 evaluate_anomaly_detection_augmented.py  # single seed=42 split, feeds Table 4.3
python3 evaluate_anomaly_detection_robustness.py # 10-split variance + per-category recall
```

Reported result (Isolation Forest, mean over 10 random splits): **precision 0.578 ± 0.029, recall 0.576 ± 0.029** — below the 0.80 target set in Chapter 1, but far above the near-random performance of the two discarded direct attempts. See `RQ1_augmented_results.csv` (canonical single-split figures), `robustness_variance_results.csv` (10-split mean/std), and `robustness_per_category_recall.csv` (recall by anomaly type).

**Note on inputs:** `build_augmented_real_dataset.py` requires two real source files (real consumption export + Fichas Técnicas export) that are **not included in this repository** — they contain row-level real hotel operational data and are excluded via `.gitignore`. Only the code and the resulting aggregate metric tables are published here. The "inventory" anomaly category is not represented in this set (no real stock-count substrate is available at this scope); it is instead represented in the simulated generalisation environment (Section 4.6).

### `chain-event-registration/`

Bridges RQ1 and RQ2: registers on Hyperledger Fabric the exact same held-out test-set events that Isolation Forest flags in `anomaly-detection/evaluate_anomaly_detection_augmented.py` (Table 4.3), so the on-chain audit trail comes from the same reported predictions rather than a separate, uncontrolled sample.

```bash
cd anomaly-detection && python3 evaluate_anomaly_detection_augmented.py   # if not already run
cd ../chain-event-registration
python3 export_rq2_events_augmented.py      # reads ../anomaly-detection outputs, writes flagged_events_augmented.csv
python3 register_events_augmented.py --max-events 15   # writes events_to_register.json + invoke_register_events.sh
```

**Note on outputs:** `flagged_events_augmented.csv`, `events_to_register.json`, and `invoke_register_events.sh` all carry real article codes, quantities, and dates per flagged event, and are excluded via `.gitignore` for the same reason as the raw evaluation data. Only the generating code is published here.

### `dlt-simulation/`

`simulation.py` is the domain-informed simulation behind the ICDLT 2026 paper's throughput/volume-reduction analysis (recipe-governance consolidation benefit). Explicitly documented in-file as a **simulation with assumed parameters**, not a benchmark of a deployed system.

`generate_events.py` reuses the same seed/scenario to emit the ordered sequence of real chaincode calls (`events.json`) needed to reproduce the scenario against a live Fabric network — this is the input consumed by `replay-harness/`.

Run:
```bash
python3 simulation.py
python3 generate_events.py
```

### `chaincode/fnb-trust/`

Hyperledger Fabric chaincode (Node.js, `fabric-contract-api`/`fabric-shim`) implementing the two smart contracts described in the dissertation's design (Chapter 3) and the ICDLT paper (Tables I & II):

- `lib/stockLotContract.js` — stock-lot registration, receipt validation, allocation.
- `lib/recipeGovernanceContract.js` — recipe (ficha técnica) governance lifecycle and flagging.

Deploy with the standard Hyperledger Fabric `test-network` (`fabric-samples`), channel `mychannel`, chaincode name `fnbtrust`.

### `replay-harness/`

`replay.js` reads `dlt-simulation/generate_events.py`'s `events.json` output and submits each event as a real transaction against the deployed `fnbtrust` chaincode, measuring actual latency/throughput with bounded concurrency.

```bash
cd replay-harness
npm install
node replay.js ../dlt-simulation/events.json --concurrency 10
```

## Requirements

| Component | Version |
|---|---|
| Python | >= 3.10 |
| Node.js | >= 18 |
| Docker + Docker Compose | >= 24.x (for Fabric test-network) |
| `fabric-samples` test-network | any recent release |

## Reproducibility

All scripts use fixed random seeds (`seed=42`). Running them in the order above (`anomaly-detection` -> `chain-event-registration` -> deploy `chaincode` -> `dlt-simulation` -> `replay-harness`) reproduces the figures reported in Chapter 4 and in the ICDLT 2026 paper.
