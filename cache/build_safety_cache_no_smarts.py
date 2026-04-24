"""Derive safety_cache_no_smarts.jsonl from safety_cache.jsonl.

The no-smarts variant drops the 'Matched alerts (...)' detail lines so the
output matches screen_safety(..., include_smarts=False). Pure string
transform — no RDKit needed.
"""
import json
import os
import re

_CACHE_DIR = os.path.dirname(__file__)
_SRC = os.path.join(_CACHE_DIR, "safety_cache.jsonl")
_DST = os.path.join(_CACHE_DIR, "safety_cache_no_smarts.jsonl")

_HEADER_RE = re.compile(r"^Structural Alerts \(\d+ categories\):$")
_NUMBERED_RE = re.compile(r"^(\d+)\.\s+(.+)$")


def _strip(text: str) -> str:
    lines = text.splitlines()
    categories: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _NUMBERED_RE.match(line)
        if m:
            name = m.group(2).strip()
            desc = lines[i + 1].strip() if i + 1 < len(lines) else ""
            categories.append((name, desc))
            j = i + 2
            while j < len(lines) and not _NUMBERED_RE.match(lines[j]) and not _HEADER_RE.match(lines[j]):
                j += 1
            i = j
            continue
        i += 1

    if not categories:
        return "Structural Alerts:\nNo structural alerts found."

    out = [f"Structural Alerts ({len(categories)} categories):", ""]
    for n, (name, desc) in enumerate(categories, 1):
        out.append(f"{n}. {name}")
        if desc:
            out.append(desc)
        out.append("")
    return "\n".join(out).rstrip()


def main() -> None:
    with open(_SRC) as f_in, open(_DST, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            entry = json.loads(line)
            result = entry.get("result")
            if isinstance(result, str):
                entry["result"] = _strip(result)
            f_out.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
