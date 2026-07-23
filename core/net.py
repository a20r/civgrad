"""
GSC Petri Net — hierarchical model of global supply chain brittleness.

Concepts:
  Place       = stock (tokens = units of material/capacity), with rebuild_years
                and region attributes for brittleness scoring.
  Transition  = process. Supports read arcs (capital equipment: required,
                not consumed) and inhibitor arcs (policy blocks, e.g. export bans).
  Oracle      = substitution transition: black-box stand-in for a subnet we
                haven't expanded yet. Declares its interface (port places) and
                a contract string. Can later be refined_by a full Net.

Analyses:
  - structural SPOFs: transitions whose removal makes a goal place unreachable
  - bounded reachability + deadlock detection
  - brittleness score per place:
      concentration (1 / #independent minting transitions)
      x rebuild_years
      x correlated-risk multiplier (shared-region exposure)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from collections import defaultdict


@dataclass(eq=False)  # identity-hashed so Places can key arc dicts
class Place:
    name: str
    tokens: int = 0
    rebuild_years: float = 0.5   # time to re-mint capacity if wiped out
    region: str = "global"       # for correlated-risk grouping

    def __repr__(self):
        return f"({self.name}:{self.tokens})"


@dataclass
class Transition:
    name: str
    inputs: dict = field(default_factory=dict)     # Place -> weight (consumed)
    outputs: dict = field(default_factory=dict)    # Place -> weight (produced)
    read_arcs: dict = field(default_factory=dict)  # Place -> weight (required, not consumed)
    inhibitors: list = field(default_factory=list) # Places that BLOCK firing if tokens > 0
    region: str = "global"
    disabled: bool = False                          # for what-if removal

    def enabled(self) -> bool:
        if self.disabled:
            return False
        if any(p.tokens > 0 for p in self.inhibitors):
            return False
        if any(p.tokens < w for p, w in self.read_arcs.items()):
            return False
        return all(p.tokens >= w for p, w in self.inputs.items())

    def fire(self):
        assert self.enabled(), f"{self.name} not enabled"
        for p, w in self.inputs.items():
            p.tokens -= w
        for p, w in self.outputs.items():
            p.tokens += w


@dataclass
class Oracle(Transition):
    """Substitution transition: unexpanded subnet. Fires per its declared
    contract until refined_by a real Net (then the subnet's dynamics apply)."""
    contract: str = ""
    refined_by: "Net | None" = None


class Net:
    def __init__(self, name: str):
        self.name = name
        self.places: dict[str, Place] = {}
        self.transitions: dict[str, Transition] = {}

    def place(self, name, **kw) -> Place:
        self.places[name] = Place(name, **kw)
        return self.places[name]

    def add(self, t: Transition) -> Transition:
        self.transitions[t.name] = t
        return t

    # ---------- analyses ----------

    def marking(self) -> tuple:
        return tuple(p.tokens for p in self.places.values())

    def set_marking(self, m: tuple):
        for p, v in zip(self.places.values(), m):
            p.tokens = v

    def enabled_transitions(self):
        return [t for t in self.transitions.values() if t.enabled()]

    def reachable(self, cap=3, max_states=200_000):
        """Bounded reachability: token counts capped at `cap` to keep the
        state space finite. Returns (states, deadlocks)."""
        start = self.marking()
        seen, stack, deadlocks = {start}, [start], []
        while stack and len(seen) < max_states:
            m = stack.pop()
            self.set_marking(m)
            en = self.enabled_transitions()
            if not en:
                deadlocks.append(m)
                continue
            for t in en:
                self.set_marking(m)
                t.fire()
                nxt = tuple(min(v, cap) for v in self.marking())
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        self.set_marking(start)
        return seen, deadlocks

    def can_reach_tokens_in(self, goal: str, cap=3) -> bool:
        gi = list(self.places).index(goal)
        states, _ = self.reachable(cap=cap)
        return any(s[gi] > 0 for s in states)

    def spofs(self, goal: str) -> list[str]:
        """Transitions whose removal makes `goal` unreachable from the
        current marking (with goal place zeroed to force production)."""
        saved = self.marking()
        out = []
        for t in self.transitions.values():
            self.set_marking(saved)
            self.places[goal].tokens = 0
            t.disabled = True
            if not self.can_reach_tokens_in(goal):
                out.append(t.name)
            t.disabled = False
        self.set_marking(saved)
        return out

    def minting_transitions(self, p: Place):
        return [t for t in self.transitions.values() if p in t.outputs]

    def brittleness(self) -> dict[str, float]:
        """concentration x rebuild x correlated-risk, per place."""
        region_load = defaultdict(int)  # how many critical places share a region
        for p in self.places.values():
            region_load[p.region] += 1
        scores = {}
        for p in self.places.values():
            mint = self.minting_transitions(p)
            if not mint:
                continue  # source places scored elsewhere
            concentration = 1.0 / len({t.region for t in mint})
            corr = region_load[p.region]  # co-located critical stocks amplify
            scores[p.name] = round(concentration * p.rebuild_years * corr, 2)
        return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


# =====================================================================
# TOP-LEVEL MAP: semiconductor slice, with oracles at the frontier
# =====================================================================
# Parameter values (tokens, rebuild years, regions, arcs, contracts) come
# from map/semiconductors/net.yaml via core.map_loader — the map is the
# source of truth. The ordered name lists below are the ENGINE INDEX LAYOUT
# (declaration order defines marking-tuple indices) and are asserted against
# the yaml at build time; changing structure means changing both, on purpose
# (that's a subnet PR).

from core.map_loader import net_spec

NET_PLACE_ORDER = [
    "Ga_byproduct", "Ga_refined", "Ne_crude", "Ne_purified", "EUV_tools",
    "Wafers_advanced", "Chips_fabbed", "Chips_packaged", "Goods", "E_waste",
    "ExportBan_CN_US",
    "EUV_optics", "EUV_worn",   # EquipMfg subnet (REFINED from ORACLE_EquipMfg)
]
NET_TRANSITION_ORDER = [
    "Refine_Ga", "Purify_Ne", "Fab", "Package", "Ship_TW_Strait",
    "Consume", "Recycle",
    "ORACLE_Mining", "ORACLE_WaferSupply",
    "Build_EUV", "Wear_EUV", "Refurb",          # EquipMfg subnet
    "ORACLE_Optics", "ORACLE_Fertilizer",
]


def build_semiconductor_slice() -> Net:
    net_name, pspec, tspec = net_spec(NET_PLACE_ORDER, NET_TRANSITION_ORDER)
    n = Net(net_name)
    for p in pspec:
        n.place(p.name, tokens=p.tokens, rebuild_years=p.rebuild_years,
                region=p.region)
    for t in tspec:
        cls = Oracle if t.oracle else Transition
        kw = {"contract": t.contract} if t.oracle else {}
        n.add(cls(t.name,
                  {n.places[p]: w for p, w in t.inputs.items()},
                  {n.places[p]: w for p, w in t.outputs.items()},
                  read_arcs={n.places[p]: w for p, w in t.read_arcs.items()},
                  inhibitors=[n.places[p] for p in t.inhibitors],
                  region=t.region, **kw))
    return n


if __name__ == "__main__":
    net = build_semiconductor_slice()

    print(f"== {net.name} ==")
    print(f"{len(net.places)} places, {len(net.transitions)} transitions "
          f"({sum(isinstance(t, Oracle) for t in net.transitions.values())} oracles)\n")

    print("-- Structural SPOFs for Goods --")
    for t in net.spofs("Goods"):
        print(f"   {t}")

    print("\n-- Brittleness (concentration x rebuild_years x co-located risk) --")
    for name, s in net.brittleness().items():
        print(f"   {s:6.1f}  {name}")

    print("\n-- What-if: export ban fires (policy token placed) --")
    net.places["ExportBan_CN_US"].tokens = 1
    net.places["Ga_refined"].tokens = 0  # stockpile exhausted
    ok = net.can_reach_tokens_in("Chips_fabbed")
    print(f"   Chips reachable with ban active + no Ga stockpile: {ok}")
    net.places["ExportBan_CN_US"].tokens = 0

    print("\n-- Deadlock check (bounded, cap=2) --")
    states, dls = net.reachable(cap=2)
    print(f"   {len(states)} reachable states, {len(dls)} deadlock states")

    print("\n-- Doomed-state check: the fab<->equipment trap --")
    # Firing sequence to the trap: tools wear out while no chip stock exists.
    # Wear_EUV fires (tool dies) before any chips were banked.
    net.places["EUV_tools"].tokens = 1
    net.places["Chips_fabbed"].tokens = 0
    net.places["Chips_packaged"].tokens = 0
    net.transitions["Wear_EUV"].fire()   # legal single firing from initial marking
    print(f"   marking after premature tool wear: EUV={net.places['EUV_tools'].tokens}, "
          f"chips={net.places['Chips_fabbed'].tokens}, pkg={net.places['Chips_packaged'].tokens}")
    ok = net.can_reach_tokens_in("Chips_fabbed")
    print(f"   Chips_fabbed ever reachable again: {ok}")
    print("   (raw materials still flow forever -- livelock, not deadlock: "
          "the net keeps 'working' but can never make a chip again)")
