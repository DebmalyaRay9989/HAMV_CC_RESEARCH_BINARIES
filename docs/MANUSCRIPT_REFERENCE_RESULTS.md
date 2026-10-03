# Manuscript reference results and reproducibility boundary

The revised manuscript reports the following observed results for its completed controlled synthetic fixture:

- ARI: 0.8882 +/- 0.0960
- NMI: 0.8769 +/- 0.1006
- Pairwise seed ARI: 0.8281 +/- 0.1037
- Silhouette: 0.1497 +/- 0.0095
- Calinski-Harabasz: 24.64 +/- 1.61
- Davies-Bouldin: 2.2470 +/- 0.0842

The manuscript specifies 180 samples, three latent groups, 12/16/10 dimensional views, approximately 12% independently masked view entries, ten matched seeds (0-9), 30 training epochs, and fixed k=3.

This package contains a reproducible fixture reconstructed from those structural specifications, not the original byte-identical fixture. The runner therefore computes and reports its own observed metrics; it does not hard-code the manuscript values. Exact numerical reproduction requires the original fixture/training state or a released byte-identical implementation/data artifact.
