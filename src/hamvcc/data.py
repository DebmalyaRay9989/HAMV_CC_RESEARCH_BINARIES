from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np

from .paths import resolve_project_path

VIEWS = ("structural", "visual", "text")

@dataclass
class MultiViewData:
    views: dict
    masks: np.ndarray
    labels: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None

    @property
    def n(self):
        return self.masks.shape[0]

    def validate(self):
        if set(self.views) != set(VIEWS):
            raise ValueError(f"views must be exactly {VIEWS}")
        n = self.n
        for name, x in self.views.items():
            if x.ndim != 2 or x.shape[0] != n:
                raise ValueError(f"view {name} must be 2-D with {n} rows")
            if not np.isfinite(x).all():
                raise ValueError(f"view {name} contains non-finite values")
        if self.masks.shape != (n, len(VIEWS)):
            raise ValueError("masks must have shape (N,3)")
        if not np.all(self.masks.sum(axis=1) >= 1):
            raise ValueError("every sample must retain at least one available view")


def load_npz(path):
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {resolved}\n"
            f"Configured path: {path}\n"
            f"Expected package-relative location: DATA\\archive_a_features.npz"
        )
    z = np.load(resolved, allow_pickle=False)
    views = {v: np.asarray(z[v], dtype=np.float32) for v in VIEWS}
    masks = np.asarray(z["masks"], dtype=np.float32)
    labels = np.asarray(z["labels"]) if "labels" in z.files else None
    ids = np.asarray(z["ids"]) if "ids" in z.files else np.arange(masks.shape[0]).astype(str)
    d = MultiViewData(views, masks, labels, ids)
    d.validate()
    return d


def apply_condition(data, condition):
    masks = data.masks.copy()
    idx = {v:i for i,v in enumerate(VIEWS)}
    if condition == "full":
        pass
    elif condition in {"no_text", "no_visual", "no_structural"}:
        masks[:, idx[condition.replace("no_", "")]] = 0
    elif condition.startswith("only_"):
        keep = condition.replace("only_", "")
        masks[:] = 0
        masks[:, idx[keep]] = 1
    else:
        raise ValueError(f"unknown missing_view_condition={condition}")
    if np.any(masks.sum(axis=1) < 1):
        raise ValueError(f"condition {condition} leaves samples with zero available views")
    return MultiViewData(data.views, masks, data.labels, data.ids)
