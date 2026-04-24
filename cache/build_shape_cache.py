"""Derive shape_cache.jsonl from three_d_cache.jsonl.

Extracts PMI (npr1, npr2, shape_class) from the cached 3D shape strings so
downstream code can skip ETKDG embedding + MMFF minimization. Pure regex
transform — no RDKit needed.
"""
import json
import os
import re

_CACHE_DIR = os.path.dirname(__file__)
_SRC = os.path.join(_CACHE_DIR, "three_d_cache.jsonl")
_DST = os.path.join(_CACHE_DIR, "shape_cache.jsonl")

_PMI_RE = re.compile(
    r"PMI ratios \(npr1, npr2\): \(([^,]+),\s*([^)]+)\)\s*(?:\u2192|->)\s*(\S+)"
)


def _parse(result: str):
    m = _PMI_RE.search(result)
    if not m:
        return None
    try:
        npr1 = float(m.group(1))
        npr2 = float(m.group(2))
    except ValueError:
        return None
    shape_class = m.group(3).strip()
    return npr1, npr2, shape_class


def main() -> None:
    seen: set[str] = set()
    written = 0
    with open(_SRC) as f_in, open(_DST, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            entry = json.loads(line)
            smi = entry.get("smiles")
            result = entry.get("result")
            if not isinstance(smi, str) or not isinstance(result, str):
                continue
            if smi in seen:
                continue
            parsed = _parse(result)
            if parsed is None:
                continue
            npr1, npr2, shape_class = parsed
            f_out.write(json.dumps({
                "smiles": smi,
                "npr1": npr1,
                "npr2": npr2,
                "shape_class": shape_class,
            }) + "\n")
            seen.add(smi)
            written += 1
    print(f"wrote {written} entries to {_DST}")


if __name__ == "__main__":
    main()
