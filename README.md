# Prototype Code — MTDN Dissertation "Enhancing Accuracy and Trust in Hospitality F&B Reporting: A Hybrid AI-Blockchain Artefact"

Supplementary code referenced in **Annex D** of the dissertation (Carla Alexandra Viveiros Piedade, ISCTE MTDN). This repository contains the two functional components of the artefact evaluated in Chapter 4 (Demonstration & Evaluation) and in the companion ICDLT 2026 conference paper.

## Repository structure

```
prototype/
├── anomaly-detection/          AI layer — RQ1 evaluation (Isolation Forest vs. Autoencoder)
│   ├── nonaugmented/           RQ1 direct evaluation on the unaugmented operational record (Section 4.5.1)
│   └── simulated/              Generalisation check on a simulated, masked population (Section 4.6)
├── data-crosscheck/            POS × recipe-sheet × ERP reconciliation (Section 3.1.3)
├── chain-event-registration/   RQ2 — registers RQ1's flagged test-set events on Fabric
├── dlt-simulation/              DLT layer — throughput/volume simulation (ICDLT 2026 paper, Section 5)
├── chaincode/fnb-trust/         Hyperledger Fabric chaincode of the artefact design (StockLot + RecipeGovernance)
├── chaincode/anomaly-events/    Reduced PoC chaincode actually deployed for the RQ2 validation (Section 4.7)
└── replay-harness/              Node.js harness to replay simulated events as real Fabric transactions
```

### `anomaly-detection/`

The primary RQ1 evaluation (Table 4.2; Sections 3.1.3 / 4.5.2) runs on an **augmented real-data evaluation set**: the real substrate is 3,705 confirmed POS–ERP consumption discrepancies (real articles, real quantities), into which controlled, calibrated field-level perturbations are injected (isolated "consumption" anomalies and systematic per-article "recipe" anomalies), each grounded in that article's own real historical mean/std. Two earlier direct attempts to evaluate on the raw real discrepancy log were tried and discarded (circular labels; near-base-rate single-source features). A third, methodologically sound direct evaluation was subsequently built on the full operational record and **is** reported, in Section 4.5.1 — see [`anomaly-detection/nonaugmented/`](#anomaly-detectionnonaugmented) below. The augmented set remains the primary RQ1 evaluation (Section 4.5.2).

Pipeline:
```bash
pip install scikit-learn pandas numpy shap
python3 build_augmented_real_dataset.py        # builds the real+injected evaluation set
python3 evaluate_anomaly_detection_augmented.py  # single seed=42 split, feeds Table 4.2
python3 evaluate_anomaly_detection_robustness.py # 10-split variance + per-category recall
```

Reported result (Isolation Forest, mean over 10 random splits): **precision 0.578 ± 0.029, recall 0.576 ± 0.029** — below the 0.80 target set in Chapter 1, but far above the near-random performance of the two discarded direct attempts. See `RQ1_augmented_results.csv` (canonical single-split figures), `robustness_variance_results.csv` (10-split mean/std), and `robustness_per_category_recall.csv` (recall by anomaly type).

**Note on inputs:** `build_augmented_real_dataset.py` requires two real source files (real consumption export + Fichas Técnicas export) that are **not included in this repository** — they contain row-level real hotel operational data and are excluded via `.gitignore`. Only the code and the resulting aggregate metric tables are published here. The "inventory" anomaly category is not represented in this set (no real stock-count substrate is available at this scope); it is instead represented in the simulated generalisation environment (Section 4.6).

**Protocol robustness check.** `check_augmented_temporal.py` re-runs the augmented evaluation under a *temporal* hold-out instead of the stratified random split used for Table 4.2, to test whether the reported figures depend on the splitting protocol. Three variants: (A) the thesis protocol, stratified random 70/30; (B) temporal hold-out, train ≤ 14 Jun, test 15–28 Jun; (C) temporal hold-out with the z-score baseline recomputed on the training window only (no transductive leakage). F1 stays within 0.539–0.566 across all three, so the Table 4.2 result is not an artefact of random splitting. The strictest of the three variants is the one carried into Table 4.4. Results in `check_augmented_temporal.csv`.

```bash
AUGMENTED_CSV=./augmented_real_dataset.csv python3 check_augmented_temporal.py
```

### `anomaly-detection/nonaugmented/`

The **direct evaluation on the unaugmented operational record** reported in Section 4.5.1 (Table 4.4). No synthetic perturbation is involved: the unit of analysis is a consumption line exactly as the POS proposed it, and the label is whether the manual review process subsequently corrected that line (`Quantidade Alterada` / `Referência Alterada` in the ERP differences report). The population is the full operational window, 12,246 lines, of which 281 were corrected — a base rate of 2.29%.

The methodological point that distinguishes this from the two discarded attempts is that the **post-correction ERP quantity never enters as a feature**. For flagged lines the feature set is built from `Qtt host`, the pre-correction value; for unflagged lines host and ERP agree by definition. The evaluation script asserts this at runtime: no feature may correlate with the label above |r| = 0.5, and the script aborts if one does.

Pipeline:
```bash
pip install scikit-learn numpy openpyxl
export DATA_DIR=/path/to/local/exports    # not versioned — see "Note on inputs"
python3 build_dataset.py                  # writes lines.json + dayvol.json (12,246 lines)
python3 evaluate_nonaugmented_real.py     # Isolation Forest + autoencoder, temporal hold-out
python3 robustness.py                     # permutation test, 10 random splits, feature importance
```

Reported result (Isolation Forest, temporal hold-out, test window 15–28 Jun 2026): **PR-AUC 0.063 against a 2.41% base rate — a lift of 2.61×**, with precision 0.076 (95% CI [0.034, 0.124]) and recall 0.101 (95% CI [0.048, 0.165]). A permutation test over 2,000 label shuffles gives *p* = 0.0005, so the lift is distinguishable from chance; over ten random splits the mean lift rises to 4.4×. The autoencoder baseline reaches a lift of 1.62×. See `results_nonaugmented.csv`.

These figures are far below the augmented-set figures, and Section 4.5.1 reads them as such: detection on the raw operational record is well above chance but not yet operationally useful on its own. Permutation feature importance shows the signal is carried by line **value** rather than by the historical z-score — corrected lines have a median value of €6.84 against €0.70 for uncorrected ones — which the dissertation discusses as evidence of reviewer-attention bias in the labels rather than of a purely technical detection limit.

**Note on inputs:** `build_dataset.py` reads three real ERP exports (consumption listing, banquet consumption detail, POS–ERP differences report) from `$DATA_DIR/01_DATA_CONSUMOS/`, and the feature builders read `$DATA_DIR/fichas_tecnicas_flat.csv` (produced by `data-crosscheck/`). None of these are versioned here — they carry row-level real hotel operational data and are excluded via `.gitignore`. Only the code and the aggregate metric tables are published.

### `anomaly-detection/simulated/`

The generalisation check reported in Section 4.6 (Table 4.5). The population here is neither real nor augmented-real: it is **generated from distributions calibrated on the real log** — per-article means and standard deviations learned from the operational data — and a share of the injected signal is then **masked**, reproducing the retention behaviour diagnosed in Section 3.1.3.

The point of the exercise is comparability: the same single feature (`deviation_zscore_historical`) and the same train/test protocol as the augmented evaluation, so the three populations (real, augmented-real, simulated-masked) can be read side by side. Detection on the masked simulated population comes out close to random for both models (Isolation Forest precision 0.087, recall 0.085 at the contamination threshold), which is the point: it independently reproduces the signal-loss finding rather than demonstrating successful transfer.

```bash
export DATA_DIR=/path/to/local/exports        # holds real_dataset_v2.csv, not versioned
python3 build_simulated_calibrated_dataset.py # learns calibration, writes the simulated population
python3 evaluate_anomaly_detection_simulated_augmented.py
```

Results in `table4_5_simulated_results.csv`. The generated population itself, `simulated_calibrated_dataset.csv`, is excluded via `.gitignore` because it carries per-article statistics learned from the real log.

### `data-crosscheck/`

`build_real_crosscheck.py` reconstructs theoretical consumption from POS sales by exploding each sold menu item through its recipe technical sheet (*ficha técnica*), and compares the result against the ERP's recorded consumption for the same window. This is the reconciliation described in Section 3.1.3, and it is what establishes that the differences report retains both the POS-proposed and the ERP-stored quantity for every record.

```bash
python3 build_real_crosscheck.py --outdir ./out
```

It writes `fichas_tecnicas_flat.csv` (the flattened recipe sheets, consumed by `anomaly-detection/nonaugmented/`) and `cruzamento_teorico_vs_erp.csv` (the line-by-line comparison). Both contain real article codes and quantities and are excluded via `.gitignore`.

### `chain-event-registration/`

Bridges RQ1 and RQ2: registers on Hyperledger Fabric the exact same held-out test-set events that Isolation Forest flags in `anomaly-detection/evaluate_anomaly_detection_augmented.py` (Table 4.2), so the on-chain audit trail comes from the same reported predictions rather than a separate, uncontrolled sample.

```bash
cd anomaly-detection && python3 evaluate_anomaly_detection_augmented.py   # if not already run
cd ../chain-event-registration
python3 export_rq2_events_augmented.py      # reads ../anomaly-detection outputs, writes flagged_events_augmented.csv
python3 register_events_augmented.py --max-events 15   # writes events_to_register.json + invoke_register_events.sh
```

The generated `invoke_register_events.sh` targets the `anomalyevents` chaincode; deploy it first with `chaincode/anomaly-events/setup_test_network.sh`.

**Note on outputs:** `flagged_events_augmented.csv`, `events_to_register.json`, and `invoke_register_events.sh` all carry real article codes, quantities, and dates per flagged event, and are excluded via `.gitignore` for the same reason as the raw evaluation data. Only the generating code is published here.

### `dlt-simulation/`

`simulation.py` is the domain-informed simulation behind the ICDLT 2026 paper's throughput/volume-reduction analysis (recipe-governance consolidation benefit). Explicitly documented in-file as a **simulation with assumed parameters**, not a benchmark of a deployed system.

`simulation_measured_detector.py` re-runs the same scenario (same seed, same 144 anomalous lots) with the detector's measured operating point from Section 4.5.2 (recall 0.663, precision 0.454) instead of perfect detection, via 10,000 Monte Carlo draws of the detection step. It reproduces the 45.8% perfect-detection figure and reports 17.2% (95% interval 13.4%–21.0%) at the measured operating point, plus the sensitivity grid (9.4%–24.3%) — dissertation Table 4.4 (last row) and Section 4.8.3. Output saved in `simulation_measured_detector_output.txt`.

`generate_events.py` reuses the same seed/scenario to emit the ordered sequence of real chaincode calls (`events.json`) needed to reproduce the scenario against a live Fabric network — this is the input consumed by `replay-harness/`.

Run:
```bash
python3 simulation.py
python3 simulation_measured_detector.py
python3 generate_events.py
```

### `chaincode/fnb-trust/`

Hyperledger Fabric chaincode (Node.js, `fabric-contract-api`/`fabric-shim`) implementing the two smart contracts described in the dissertation's design (Chapter 3) and the ICDLT paper (Tables I & II):

- `lib/stockLotContract.js` — stock-lot registration, receipt validation, allocation.
- `lib/recipeGovernanceContract.js` — recipe (ficha técnica) governance lifecycle and flagging.

Deploy with the standard Hyperledger Fabric `test-network` (`fabric-samples`), channel `mychannel`, chaincode name `fnbtrust`.

### `chaincode/anomaly-events/`

The reduced proof-of-concept chaincode **actually deployed** for the blockchain validation reported in Section 4.7 and shown in Figures 4.1–4.4. Written in Go (`fabric-contract-api-go`), it implements a single `SmartContract` with four externally invoked functions — `RegisterEvent`, `GetEvent`, `GetEventsByArticle`, `GetAllEvents` — over an `AnomalyEvent` record (event id, article code, category, anomaly type, model source, score, timestamp, validator, payload hash, status, description).

This is deliberately a **subset** of the fuller lot- and recipe-governance lifecycle in `chaincode/fnb-trust/`. Its purpose is to validate the two trust mechanisms that RQ2 turns on — multi-organisation endorsement, and immutable queryable storage — not to reproduce every function of the target design. Section 4.7.1 of the dissertation states this explicitly.

`setup_test_network.sh` brings up the standard two-organisation Fabric test-network (downloading `fabric-samples` if absent), generates a minimal `go.mod` pinning `fabric-contract-api-go v1.2.2`, and deploys this chaincode to channel `mychannel` under the name `anomalyevents`.

```bash
cd chaincode/anomaly-events
chmod +x setup_test_network.sh
./setup_test_network.sh
```

The events submitted against it are the ones produced by `chain-event-registration/` — closing the chain from an Isolation Forest detection to a committed on-chain record.

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

All scripts use fixed random seeds (`seed=42`). Running them in the order `data-crosscheck` -> `anomaly-detection` (augmented, then `nonaugmented/`) -> `chain-event-registration` -> deploy `chaincode` -> `dlt-simulation` -> `replay-harness` reproduces the figures reported in Chapter 4 and in the ICDLT 2026 paper.

## Data access

Every script that touches real operational data reads it from a location given by an environment variable (`DATA_DIR`, `AUGMENTED_CSV`) rather than a hard-coded path, and no such file is versioned here. The row-level exports are held under the dissertation's own data management arrangements and are available to the jury on request; the aggregate metric tables published in this repository are sufficient to verify every number reported in Chapter 4.
