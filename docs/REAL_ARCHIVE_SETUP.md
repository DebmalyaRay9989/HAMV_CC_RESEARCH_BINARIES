# Real 638-document archive setup

The executable HAMV-CC runner operates on **precomputed views** so that feature extraction can be audited separately from the representation/clustering experiment.

Create `DATA/archive_a_features.npz` with:

- `structural`: float32 array `[N, D_structural]`
- `visual`: float32 array `[N, D_visual]`
- `text`: float32 array `[N, D_text]`
- `masks`: float32/bool array `[N,3]`, ordered structural/visual/text
- optional `ids`: string array `[N]` matching the manifest
- optional `labels`: integer/string array `[N]`; leave absent for the unlabeled deployment archive

For each row, the feature extractor must use only the corresponding source document. Do not compute archive-level normalization using future/test partitions when a benchmark split is being evaluated.

Then edit `configs/real_archive.json` and run:

```bash
python run_experiments.py --config configs/real_archive.json --seeds 42,43,44,45,46,47,48,49,50,51 --ablations full,no_reliability,no_cross_view,no_cluster_consistency,fixed_k --conditions full,no_text,no_visual,no_structural,only_text,only_visual,only_structural --output runs/archive_a
python aggregate_results.py --input runs/archive_a --output results/archive_a
```

For the initial 638-document confirmation, it is reasonable to run only `full` first. Do not launch the full factorial matrix until the pilot completes without data-integrity failures.
