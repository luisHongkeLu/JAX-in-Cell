# jaxincell/_external_fields.py
# Stand-alone, NPZ-only external field loader.

from __future__ import annotations
from typing import Optional
import numpy as np

def _load_npz_key(npz_path: str, key: str, G: int) -> np.ndarray:
    """Load key ('B_ext' or 'E_ext') from an NPZ and validate shape (G,3)."""
    with np.load(npz_path) as z:
        if key not in z:
            # If the file doesn't contain the requested key, just use zeros.
            return np.zeros((G, 3), dtype=np.float32)
        arr = np.asarray(z[key])
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{key} in {npz_path} must have shape (G,3), got {arr.shape}")
    if arr.shape[0] != G:
        raise ValueError(f"{key} rows G={arr.shape[0]} do not match solver G={G}")
    return arr.astype(np.float32)

def _resolve_one_section(section: Optional[dict], key: str, G: int) -> Optional[np.ndarray]:
    """
    Resolve one top-level section:
      [external_magnetic_field] -> key='B_ext'
      [external_electric_field] -> key='E_ext'
    Behavior:
      - If section is None: return None (absent → default zero later).
      - If present, must contain 'path' to a .npz.
      - Returns a NumPy array (G,3) or None.
    """
    if section is None:
        return None
    if not isinstance(section, dict) or "path" not in section or not str(section["path"]).strip():
        raise ValueError(
            "External field header present but missing 'path'. Expected e.g.:\n"
            "[external_magnetic_field]\npath = \"/path/to/file.npz\""
        )
    return _load_npz_key(str(section["path"]).strip(), key, G)

def resolve_external_fields_top_level_inplace(toml_dict: dict, G: int) -> None:
    """
    Read top-level TOML headers and replace them in-place with numeric arrays:
      [external_magnetic_field] -> {'B': (G,3) array}
      [external_electric_field] -> {'E': (G,3) array}
    If a header is absent, do nothing (the simulation will default to zeros).
    """
    secB = toml_dict.get("external_magnetic_field")
    secE = toml_dict.get("external_electric_field")

    B = _resolve_one_section(secB, "B_ext", G)
    E = _resolve_one_section(secE, "E_ext", G)

    if B is not None:
        toml_dict["external_magnetic_field"] = {"B": B}
    if E is not None:
        toml_dict["external_electric_field"] = {"E": E}
