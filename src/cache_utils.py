"""
Helpers for cache path resolution and robust embedding loading.

tfidf cache was originally created before the seed-suffix system was added.
Files exist as e.g. 'lgbm.pkl' not 'lgbm_seed_42.pkl'.
Some .npy files were saved before the toarray() conversion was applied,
meaning they contain a csr_matrix wrapped in a numpy object array.
"""

import os
import numpy as np


def resolve_cache_path(cache_dir, name, suffix, ext):
    """
    Try  : <cache_dir>/<name><suffix>.<ext>
    Then : <cache_dir>/<name>.<ext>

    Returns the path that exists, or the suffixed path (for creation).
    """
    suffixed = os.path.join(cache_dir, f"{name}{suffix}.{ext}")
    legacy   = os.path.join(cache_dir, f"{name}.{ext}")
    if os.path.exists(suffixed):
        return suffixed
    if os.path.exists(legacy):
        return legacy
    return suffixed  # neither exists → return suffixed (new files will use suffix)


def load_embedding(path):
    """
    Load a cached embedding .npy file robustly.

    Handles three formats that may appear in the tfidf cache:
      1. Normal dense float32/float64 array  → load and cast
      2. Object array wrapping a csr_matrix  → call toarray(), then cast
      3. Object array wrapping a dense array → extract and cast
    """
    arr = np.load(path, allow_pickle=True)

    if arr.dtype != object:
        # Standard dense array — fast path
        return arr.astype(np.float32)

    # Object array: unwrap the contained object
    obj = arr.item() if arr.ndim == 0 else arr.flat[0]

    if hasattr(obj, "toarray"):
        # scipy sparse matrix (e.g. csr_matrix)
        return obj.toarray().astype(np.float32)

    # Plain python/numpy object — try direct conversion
    return np.array(obj, dtype=np.float32)
