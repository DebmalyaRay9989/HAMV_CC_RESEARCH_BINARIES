HAMV-CC REAL ARCHIVE DATA
==========================

Place the real precomputed feature archive here:

    DATA/archive_a_features.npz

Required arrays inside the NPZ:
    structural : float32 [N, D_structural]
    visual     : float32 [N, D_visual]
    text       : float32 [N, D_text]
    masks      : float32/bool [N, 3]  (structural, visual, text)

Optional arrays:
    ids        : string [N]
    labels     : integer/string [N]

The experiment package now resolves DATA/archive_a_features.npz relative to
this package root, so it can be launched from the package directory without
requiring an absolute Windows path.
