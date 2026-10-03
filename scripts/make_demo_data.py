"""Generate the controlled 180-sample synthetic validation fixture.
This is a reproducible reconstruction of the manuscript fixture specification,
not a byte-identical release of the original fixture.
"""
from pathlib import Path
import numpy as np
rng = np.random.default_rng(7)
n = 180
y = np.repeat(np.arange(3), n // 3)
views = {}
for name, d, scale in [("structural", 12, 1.0), ("visual", 16, 0.9), ("text", 10, 0.8)]:
    centers = rng.normal(size=(3, d)) * 0.425
    views[name] = np.vstack([centers[c] + rng.normal(scale=scale, size=d) for c in y]).astype("float32")
mask = np.ones((n, 3), dtype="float32")
# Exactly 65 of 540 view entries are masked: 12.04%, while every sample keeps at least one view.
flat = rng.choice(n * 3, size=65, replace=False)
mask.flat[flat] = 0
Path("data_demo").mkdir(exist_ok=True)
np.savez("data_demo/synthetic_multiview.npz", **views, masks=mask, labels=y, ids=np.array([f"doc_{i:04d}" for i in range(n)]))
print(f"created data_demo/synthetic_multiview.npz: n={n}, missing_fraction={1.0-mask.mean():.4f}")
