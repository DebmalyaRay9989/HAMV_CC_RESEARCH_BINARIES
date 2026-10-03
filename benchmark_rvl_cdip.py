"""Evaluate a frozen HAMV-CC checkpoint on an external benchmark.

The test feature NPZ is expected to contain structural, visual, text, masks,
labels, and optionally ids. Labels are used only after representation and k
selection have been frozen. A trained checkpoint is mandatory; this script
never creates or trains a new encoder.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from src.hamvcc.data import load_npz
from src.hamvcc.metrics import internal_metrics, external_metrics


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "view_dims" not in ckpt:
        raise ValueError("Checkpoint must contain state_dict and view_dims metadata")
    return ckpt


def main():
    p = argparse.ArgumentParser(description="Evaluate a frozen HAMV-CC checkpoint on precomputed external test features")
    p.add_argument("--features", required=True)
    p.add_argument("--checkpoint", required=True, help="Trained HAMV-CC model checkpoint; must be frozen before test evaluation")
    p.add_argument("--k", type=int, required=True, help="Predeclared cluster count; test labels are never used to choose k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_npz(a.features)
    ckpt = load_checkpoint(a.checkpoint, device)
    dims = {k: v.shape[1] for k, v in d.views.items()}
    if dims != ckpt["view_dims"]:
        raise ValueError(f"Feature dimensions {dims} do not match checkpoint dimensions {ckpt['view_dims']}")

    from src.hamvcc.model import HAMVCC
    model = HAMVCC(dims, hidden=int(ckpt["hidden_dim"]), latent=int(ckpt["latent_dim"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        td = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in d.views.items()}
        mt = torch.tensor(d.masks, dtype=torch.float32, device=device)
        _, _, fused = model(td, mt)

    X = fused.cpu().numpy()
    labels = KMeans(n_clusters=a.k, n_init=20, random_state=a.seed).fit_predict(X)
    out = {
        "seed": a.seed,
        "k": a.k,
        "checkpoint": str(Path(a.checkpoint)),
        "status": "frozen_checkpoint_evaluation",
        **internal_metrics(X, labels),
        **external_metrics(d.labels, labels),
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
