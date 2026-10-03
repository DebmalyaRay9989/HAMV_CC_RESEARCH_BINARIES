# Package validation report

## Protocol execution

Command:

```text
python run_experiments.py --config configs/demo.json --seeds 0-9 --output runs/demo
python aggregate_results.py --input runs/demo --output results/demo
```

Observed on the validation environment:

- 180 samples
- 10 matched seeds
- 10/10 successful runs
- 45 pairwise seed comparisons
- ARI mean +/- SD: 0.898392 +/- 0.031324
- NMI mean +/- SD: 0.862553 +/- 0.038073
- Pairwise seed ARI mean +/- SD: 0.924018 +/- 0.019959

These are the results of the reconstructed package fixture. They are not substituted for the manuscript's reported values. See `MANUSCRIPT_REFERENCE_RESULTS.md`.
