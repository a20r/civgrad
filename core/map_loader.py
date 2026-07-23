"""
core/map_loader.py — the map is the source of truth for parameter values.

map/semiconductors/net.yaml supplies every parameter VALUE the engine uses:
initial tokens, rebuild years, regions, arc weights, flow capacities, oracle
contracts (via the interface yamls), and the frozen globals (K_SAT, ALPHA,
HORIZON, utilization). The ordered NAME lists — the engine's index layout for
marking tuples and capacity vectors — stay in core/net.py and
core/continuous.py, and are asserted against the yaml here so structural
drift fails loudly at import time instead of silently.

Provenance dicts ({value, units, source, confidence, date}) unwrap to their
`value`; the provenance itself is contribution metadata (PLAN.md §2) and is
consumed by fog/priority.py, not the engine.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}


def _val(x):
    """Unwrap a provenance dict to its value; pass plain scalars through."""
    return x["value"] if isinstance(x, dict) and "value" in x else x


def load_map(subnet="semiconductors"):
    if subnet not in _CACHE:
        with open(_ROOT / "map" / subnet / "net.yaml") as fh:
            _CACHE[subnet] = yaml.safe_load(fh)
    return _CACHE[subnet]


def frozen_globals(subnet="semiconductors"):
    """K_SAT / ALPHA / HORIZON / utilization — change only via methodology PR."""
    return {k: _val(v) for k, v in load_map(subnet)["frozen_globals"].items()}


def _contract(t):
    """An oracle's contract string, read from its interface yaml."""
    iface = t.get("interface")
    if not iface:
        return ""
    with open(_ROOT / iface) as fh:
        return yaml.safe_load(fh)["contract"]


def net_spec(place_order, transition_order, subnet="semiconductors"):
    """Build spec for the discrete net (core/net.py), ordered per the engine
    layout lists. Returns (net_name, place_specs, transition_specs)."""
    d = load_map(subnet)
    places = {p["name"]: p for p in d["places"]}
    trans = {t["name"]: t for t in d["transitions"]}
    assert set(places) == set(place_order), (
        f"map/{subnet}/net.yaml places drifted from the core layout: "
        f"{set(places) ^ set(place_order)}")
    assert set(trans) == set(transition_order), (
        f"map/{subnet}/net.yaml transitions drifted from the core layout: "
        f"{set(trans) ^ set(transition_order)}")
    pspec = [SimpleNamespace(name=n,
                             tokens=_val(places[n]["initial_tokens"]),
                             rebuild_years=_val(places[n]["rebuild_years"]),
                             region=places[n]["region"])
             for n in place_order]
    tspec = []
    for n in transition_order:
        t = trans[n]
        tspec.append(SimpleNamespace(
            name=n, region=t["region"], oracle=bool(t.get("oracle")),
            inputs={p: _val(w) for p, w in (t.get("inputs") or {}).items()},
            outputs={p: _val(w) for p, w in (t.get("outputs") or {}).items()},
            read_arcs={p: _val(w) for p, w in (t.get("read_arcs") or {}).items()},
            inhibitors=list(t.get("inhibitors") or []),
            contract=_contract(t) if t.get("oracle") else ""))
    return d["net"], pspec, tspec


def continuous_spec(place_order, transition_order, subnet="semiconductors"):
    """Build spec for the continuous relaxation (core/continuous.py), in the
    engine's index layout, with names mapped through `continuous.aliases`.
    Whatever the relaxation deliberately does not represent (the policy place,
    the fertilizer oracle) is asserted explicitly, so new structure cannot
    slip out of the relaxation unnoticed."""
    d = load_map(subnet)
    alias = d["continuous"]["aliases"]
    a = lambda n: alias.get(n, n)
    trans = {a(t["name"]): t for t in d["transitions"]}
    missing = set(transition_order) - set(trans)
    assert not missing, f"continuous layout names absent from the map: {missing}"
    extra_t = {t["name"] for t in d["transitions"] if a(t["name"]) not in set(transition_order)}
    assert extra_t == {"ORACLE_Fertilizer"}, (
        f"transitions unrepresented in the relaxation changed: {extra_t} — "
        f"update core/continuous.py's layout consciously (subnet PR)")
    extra_p = {p["name"] for p in d["places"] if a(p["name"]) not in set(place_order)}
    assert extra_p == {"ExportBan_CN_US"}, (
        f"places unrepresented in the relaxation changed: {extra_p} — "
        f"update core/continuous.py's layout consciously (subnet PR)")
    tuples, caps = [], []
    for cname in transition_order:
        t = trans[cname]
        tuples.append((cname,
                       {a(p): _val(w) for p, w in (t.get("inputs") or {}).items()},
                       {a(p): _val(w) for p, w in (t.get("outputs") or {}).items()},
                       [a(p) for p in (t.get("read_arcs") or {})]))
        caps.append(float(_val(t.get("capacity", 1.0))))
    fg = frozen_globals(subnet)
    return SimpleNamespace(transitions=tuples, capacities=caps,
                           k_sat=fg["K_SAT"], utilization=fg["utilization"])
