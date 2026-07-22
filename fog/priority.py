"""
fog/priority.py — which oracle's uncertainty costs us most?

v0.1 PLACEHOLDER. The real ranking is expected value of information (EVOI):
the contribution of each oracle's parameter uncertainty to variance in the
headline outputs — gradient rankings and replay predictions — so outreach
targets demonstrated need, and an expansion that moves no prediction gets
deprioritized automatically.

TODO(PLAN.md §5): replace this stub with true EVOI. Until then the ranking is
a proxy in two tiers:
  1. unexpanded oracles — pure fog: every parameter unknown, wrong in unknown
     directions (they have hidden both fragility and resilience in our own
     history). An oracle opened by a failing test outranks a declared one.
  2. expanded subnets, ranked by count of confidence-C parameters (session
     estimates that still need citations — see PLAN.md §2).

Run `python3 -m fog.priority` from the repo root.
"""

from pathlib import Path

import yaml

MAP_DIR = Path(__file__).resolve().parents[1] / "map"


def count_confidence_c(node):
    """Count provenance dicts with confidence C in a parsed yaml tree."""
    if isinstance(node, dict):
        own = 1 if node.get("confidence") == "C" else 0
        return own + sum(count_confidence_c(v) for v in node.values())
    if isinstance(node, list):
        return sum(count_confidence_c(v) for v in node)
    return 0


def load(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def oracle_rank_key(o):
    """Failing-test-opened oracles first, then declared next slices, then name."""
    return (0 if o.get("opened_by") else (1 if o.get("next_slice") else 2), o["name"])


def main():
    oracles = sorted((load(f) for f in sorted((MAP_DIR / "_oracles").glob("*.yaml"))),
                     key=oracle_rank_key)
    subnets = sorted(d.name for d in MAP_DIR.iterdir()
                     if d.is_dir() and not d.name.startswith("_"))

    print("== fog priority — placeholder EVOI ranking (TODO: PLAN.md §5) ==\n")
    print("Tier 1 — unexpanded oracles (pure fog: wrong in unknown directions):")
    for o in oracles:
        why = (f"opened by failing test: {o['opened_by']}" if o.get("opened_by")
               else "declared next slice" if o.get("next_slice")
               else "declared interface")
        print(f"   {o['name']:<16} {why}")

    print("\nTier 2 — expanded subnets by confidence-C parameter count")
    print("(session estimates needing citations; a data PR upgrades one):")
    counts = {s: count_confidence_c(load(MAP_DIR / s / "net.yaml")) for s in subnets}
    for s, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {s:<16} {n} confidence-C parameters")

    print("\n(placeholder proxy — not EVOI; see this module's docstring)")


if __name__ == "__main__":
    main()
