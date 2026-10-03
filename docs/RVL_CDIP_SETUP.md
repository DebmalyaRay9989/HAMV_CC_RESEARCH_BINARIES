# RVL-CDIP external benchmark

Use the official RVL-CDIP source and preserve its train/validation/test separation. The author page describes 400,000 grayscale document images across 16 classes, with 320,000 training, 40,000 validation, and 40,000 test images.

Official source: https://adamharley.com/rvl-cdip/

For the manuscript protocol:

1. Build features from train/validation data for any representation choices.
2. Freeze preprocessing, representation, checkpoint, and cluster-count rule.
3. Generate test features without consulting test labels.
4. Run `benchmark_rvl_cdip.py` with the frozen checkpoint and frozen test features, for example:

```bash
python benchmark_rvl_cdip.py --features DATA/rvl_cdip_test_features.npz --checkpoint runs/archive_a/<frozen_run_id>/model_checkpoint.pt --k 16 --seed 42 --output results/rvl_cdip.json
```

5. Use labels only for final ARI/NMI (and any predeclared semantic measures).

Do not use test labels to select k or tune the encoder. The evaluator requires an already-trained checkpoint and never initializes or trains a fresh model.
