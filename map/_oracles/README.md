# Unexpanded oracles — the map's frontier

Each yaml here declares a subnet we have *not* mapped: its port places, a
contract string, and nothing else. That is the point — the frontier of our
ignorance is explicit and machine-readable, and fog is wrong in unknown
directions (in this repo's own history it hid extra fragility *and* fake
resilience). Expanding one is a subnet PR: match the ports exactly, ship at
least one historical event, pass the validation gate (PLAN.md §2–3).

| oracle | ports | contract | what expanding it would test |
|---|---|---|---|
| **packaging_resin** ⚠ | `Packaging_resin` → `Package` (proposed) | *proposed*: epoxy molding compound / packaging-materials supply feeding `Package` | **The one with a failing test attached**: Sumitomo 1993 is xfail twice — packaging-material shocks are currently unrepresentable (v0 literally cannot see the event). Expansion makes the miss addressable. |
| mining | → `Ga_byproduct`, `Ne_crude` | bauxite/zinc mining + steel-mill air separation -> byproduct feeds | Byproduct economics: whether Ga/Ne supply can really scale independently of its host industries, or whether the source transitions are infinite faucets flattering the model. |
| wafer_supply | → `Wafers_advanced` | polysilicon -> ingot -> wafer (Shin-Etsu/SUMCO ~50%) | Whether the Tohoku replay survives a real two-supplier structure with its own inventories, instead of one aggregate source. |
| optics | → `EUV_optics` | Zeiss SMT: 30yr optical know-how, mirrors polished to <1nm. Deeper SPOF than ASML itself. | Whether the tool-fleet livelock deepens when optics has its own decade-scale rebuild chain behind it. |
| fertilizer | (none wired yet) | phosphate/potash/Haber-Bosch subnet — food system. NEXT SLICE. | Whether the same brittleness structure shows up outside chips — the first food-system slice, starting from an interface proposal. |

The engine treats these as substitution transitions (`core/net.py`); the
continuous relaxation stands them in as source transitions. Contracts here
are the tooltips the demo's fog nodes show.
