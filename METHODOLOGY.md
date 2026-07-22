# civgrad methodology

`version: 0.1 — skeleton`

This document is versioned like a spec (PLAN.md §4): methodology changes bump
the version here and re-run **all** validation from scratch — every event,
training events included, re-fit and re-frozen — so methodology drift cannot
quietly overfit event by event. Licensed CC BY 4.0 (see LICENSE-DOCS).

## 1. Why Petri nets

A supply chain is stocks and concurrent processes, not a graph of nodes that
are "up" or "down." Petri nets natively express consumption (input arcs),
capital equipment that enables without being consumed (read arcs), policy
blocks like export bans (inhibitor arcs), and conservation of material.
They also expose failure modes richer than disconnection: the doomed-state
demo in `core/net.py` is a livelock, not a deadlock — raw materials keep
flowing forever, yet no chip can ever be produced again once the last tool
wears out before any chips were banked. Flow-network models cannot even
state that condition.

## 2. Hierarchy: oracles and the fog of war

Regions we have not mapped are declared as **oracles** — substitution
transitions carrying only an interface (port places) and a contract string —
so the frontier of our ignorance is explicit and machine-readable
(`map/_oracles/`). The epistemic stance: unexpanded regions are wrong in
*unknown directions* — in our own design history the fog hid extra fragility
(the Zeiss optics monopoly sits a level deeper than ASML) and, symmetrically,
sources with no upstream constraint acted as infinite faucets that made
everything look rebuildable. Expanding an oracle is the highest-value
contribution (PLAN.md §2); `map/_oracles/README.md` lists each frontier and
what expanding it would test.

## 3. Continuous relaxation and differentiability

The discrete net is relaxed to a fluid model (David & Alla): markings become
real-valued stocks, firings become flow rates limited by Michaelis-Menten
saturations of their inputs, so the dynamics are piecewise-smooth and JAX can
differentiate total throughput with respect to every capacity and every
initial stockpile. The ranked gradient vector *is* the project thesis:
d(resilience)/d($) — where a marginal unit of capacity or inventory buys the
most marginal resilience. Averaging over a disruption prior
(`core/gradients.py::expected_throughput`) keeps the ranking from being
hostage to one hand-picked catastrophe; tail objectives (CVaR over scenarios)
are declared future work and currently have no test.

## 4. Adaptation

Recovery is not imposed; capacity responds to two scarcity signals — runway
(months of stock cover against net drain, against a HORIZON of 6 months) and
unfilled demand relative to the pre-crisis anchor — through a control law
whose single gain ALPHA was fitted once on the neon training event and then
frozen (`core/adaptation.py`). What is endogenous: recovery time. What is
not yet endogenous: **price-mediated allocation** — who gets the scarce
intermediate while it is scarce. The Sumitomo hysteresis (57.6% dip, no
recovery, vs. "brief pain" in history) is the open failing test that marks
this boundary, kept red on the scorecard until demand/allocation dynamics
(PLAN.md §5) resolve it honestly.

## 5. Validation protocol

Events are frozen data (`validation/events/*.yaml`): documented facts with
sources required, never tuned. Neon 2022 is the training event (ALPHA and the
90%-utilization calibration rule were set there); Tohoku 2011, Sumitomo 1993,
and photoresist 2019 are holdouts replayed with everything frozen; demand-side
events (the 2020-21 chip crunch) are explicitly out of scope until the model
has demand dynamics. New events must ship with predictions registered in the
PR description *before* the maintainer runs them, making post-hoc fitting
visible. The standing **pessimism-bias caveat**: with no demand reallocation
or substitution the model systematically overstates dips and recovery times,
so *where fragility concentrates* is more trustworthy than *how bad it gets*.

## 6. Known limits, ranked

1. **No demand/allocation dynamics** — the biggest gap; tied to the failing
   Sumitomo xfail (both protocols) and the out-of-scope chip-crunch row.
2. **Packaging materials are invisible** — `Package` has no consumable
   inputs; tied to the same miss; opens `map/_oracles/packaging_resin.yaml`.
3. **Oracle fog at every frontier** — mining, wafer supply, optics,
   fertilizer are interface-only; each hides unknown-direction error
   (`map/_oracles/README.md` lists them; no replay currently crosses them).
4. **Mean-only objective** — no tail/CVaR objective and no test exercising
   one (§3).
5. **Provenance debt** — every parameter in `map/semiconductors/net.yaml` is
   confidence C ("session estimate — needs citation"), 43 of them. Data PRs
   burn this down one citation at a time.
