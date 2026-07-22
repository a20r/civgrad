"""
validation/export_site_data.py — export repo data for the static site.

The site (site/) is plain HTML/JS with no build step and browsers don't parse
yaml, so the single-source-of-truth yamls are exported to committed JSON that
the page fetches. CI fails if these exports are stale (same convention as
SCORECARD.md):

    python3 -m validation.export_site_data

writes, deterministically:
  site/data/events.json   <- validation/events/*.yaml (ordered, verbatim)
  site/data/oracles.json  <- map/_oracles/*.yaml (fog-node tooltips)
  site/data/essay.md      <- essay/essay.md (copied; skipped until it exists)

Infinities (yaml `.inf`) become the string "inf" — strict JSON has no
Infinity literal; the site renders it as "never".
"""

import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data"


def _sanitize(node):
    if isinstance(node, dict):
        return {k: _sanitize(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_sanitize(v) for v in node]
    if isinstance(node, float) and node == float("inf"):
        return "inf"
    return node


def _load_all(directory):
    docs = []
    for f in sorted(directory.glob("*.yaml")):
        with open(f) as fh:
            docs.append(yaml.safe_load(fh))
    return docs


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    events = sorted(_load_all(ROOT / "validation" / "events"),
                    key=lambda e: e["order"])
    (OUT / "events.json").write_text(
        json.dumps(_sanitize(events), indent=1, ensure_ascii=False) + "\n")

    oracles = _load_all(ROOT / "map" / "_oracles")
    (OUT / "oracles.json").write_text(
        json.dumps(_sanitize(oracles), indent=1, ensure_ascii=False) + "\n")

    essay = ROOT / "essay" / "essay.md"
    if essay.exists():
        shutil.copyfile(essay, OUT / "essay.md")
        copied = ", essay.md"
    else:
        copied = " (essay/essay.md not present yet — skipped)"
    print(f"wrote {OUT}/events.json, oracles.json{copied}")


if __name__ == "__main__":
    main()
