HAMV-CC RESEARCH BINARIES RELEASE

This package contains the executable HAMV-CC research runner plus the required binary feature archive at DATA/archive_a_features.npz.

DATA/archive_a_features.npz was generated from the supplied derived feature table results/document_features.csv and aggregated to source-id level (683 feature records -> 659 source IDs). Arrays:
  structural: float32 [659,16]
  visual: float32 [659,81]
  text: float32 [659,128]
  masks: float32 [659,3]
  ids: string [659]

Important evidence boundary: the supplied text feature columns are zero-valued in the current derived feature table. Therefore the archive is a structurally valid three-view container, but the current empirical real-archive representation is effectively structural+visual. This is not a substitute for regenerating informative OCR/text embeddings.

The NPZ is a derived research artifact, not the private source-image archive. It is intended to remove the missing-file failure and permit execution of the packaged runner against the supplied derived features.
