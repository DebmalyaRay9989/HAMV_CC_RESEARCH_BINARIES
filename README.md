# HAMV-CC Research / Execution Package

This is the executable research harness for the Heterogeneity-Aware Multi-View Contrastive Clustering (HAMV-CC) study described in the revised manuscript.

## What is included

- `run_experiments.py` — matched-seed HAMV-CC runner and frozen ablation matrix.
- `aggregate_results.py` — seed-level statistics, paired differences, 95% CIs, bootstrap CIs, Wilcoxon signed-rank tests, standardized paired effect sizes, and Benjamini–Hochberg correction.
- `generate_manifest.py` — deterministic archive manifest generator with SHA-256 hashes and source-level grouping.
- `benchmark_rvl_cdip.py` — frozen external-benchmark evaluator that requires a trained HAMV-CC checkpoint and precomputed RVL-CDIP test features; it does not train or tune on test labels.
- `src/hamvcc/` — HAMV-CC model, data loading, clustering, metrics, experiment utilities.
- `tests/` — unit and end-to-end tests.
- `configs/` — demo and real-archive configuration templates.
- `data_demo/` — small synthetic multi-view fixture for smoke testing.
- `docs/PROTOCOL.md` — frozen experiment protocol and anti-leakage rules.
- `docs/REAL_ARCHIVE_SETUP.md` — how to connect the 638-document archive.
- `docs/RVL_CDIP_SETUP.md` — external benchmark procedure.

## Environment

Python 3.9+ is supported (Python 3.10+ is recommended).

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Run the built-in smoke experiment

```bash
python run_experiments.py --config configs/demo.json --seeds 0-9 --output runs/demo
python aggregate_results.py --input runs/demo --output results/demo
python -m pytest -q
```

## 2. Run the real 638-document archive

First create a manifest from the archive root:

```bash
python generate_manifest.py --root /PATH/TO/ARCHIVE --output manifests/archive_a.csv --extensions png,jpg,jpeg,tif,tiff,bmp
```

The runner expects precomputed multi-view features in an `.npz` file. See `docs/REAL_ARCHIVE_SETUP.md` for the exact schema. Then edit `configs/real_archive.json` and run:

```bash
python run_experiments.py --config configs/real_archive.json --seeds 42,43,44,45,46,47,48,49,50,51 --output runs/archive_a
python aggregate_results.py --input runs/archive_a --output results/archive_a
```

The 10 seeds are matched: every condition receives exactly the same seed list and source manifest.

## 3. Reproducibility outputs

Each run writes:

- `run.json` — configuration, seed, software versions, timestamps, status, and metrics.
- `labels.npy` — final cluster assignment.
- `embeddings.npy` — final fused representation.
- `per_view_weights.npy` — learned reliability weights.
- `model_checkpoint.pt` — trained model state plus architecture/config metadata for frozen downstream evaluation.
- `loss_history.csv` — epoch-level objective terms.
- `failures.json` — retained failure information when a run fails.

The aggregator reads raw run records and never reconstructs statistics from screenshots or prose.

## Scientific note

This package does **not** claim that HAMV-CC is superior until the real-archive matrix, independent semantic audit, and external benchmark are actually executed. The synthetic fixture is only an implementation/smoke test. Internal clustering metrics and external ground-truth metrics are kept separate.

Clustering metrics in the implementation follow standard scikit-learn definitions; ARI/NMI require reference labels, while Silhouette/Calinski–Harabasz/Davies–Bouldin are internal measures. See the official scikit-learn clustering documentation: https://scikit-learn.org/stable/modules/clustering.html


## Windows / PowerShell path handling

All relative dataset paths in the runner are resolved relative to the HAMV-CC package root. From the package directory, use:

```powershell
python run_experiments.py --config configs/real_archive.json --seeds 49,50,51 --ablations fixed_k --conditions only_structural --output runs/archive_a
```

The real archive must be present at `DATA/archive_a_features.npz`. The package does not fabricate or silently replace the real research dataset.
