# civgrad — Repo Plan, Contribution Model, and Gating Strategy

A differentiable, hierarchical Petri net of the global supply chain. Goal: locate where marginal investment buys the most marginal resilience — and make the map improvable by the people who actually know each territory.

## 1. Repo structure

    civgrad/
      METHODOLOGY.md          # the living methodology (versioned, see §4)
      SCORECARD.md            # auto-generated: every historical replay, hits AND misses
      core/
        net.py                # discrete PN: places, transitions, oracles, reachability
        continuous.py         # JAX fluid relaxation, flows, saturation kinetics
        adaptation.py         # scarcity signals + capacity control law (alpha, HORIZON)
        gradients.py          # capacity/stockpile gradients, expected & tail objectives
      map/
        semiconductors/       # one directory per expanded subnet
          net.yaml            # places, transitions, arcs — WITH provenance (see §2)
          events.yaml         # historical replays this subnet participates in
        _oracles/             # declared-but-unexpanded subnets: interface + contract only
          fertilizer.yaml
          packaging_resin.yaml   <- opened by the Sumitomo miss
      validation/
        events/               # frozen event definitions: documented facts only
        runner.py             # CI: replays all events, regenerates SCORECARD.md
      fog/
        priority.py           # EVOI ranking: which oracle's uncertainty costs us most

## 2. Contribution guide (the short version)

Two ways in, deliberately asymmetric in effort:

**Data PRs** (low friction): improve a number. Every parameter in `net.yaml` carries
`{value, units, source: <citation/URL>, confidence: A|B|C, date}`. A data PR must cite
a source; confidence grades are assigned by reviewers, not authors. No source, no merge —
this single rule is what keeps the parameter file from becoming a vibes registry.

**Subnet PRs** (high value): expand an oracle. A contributed subnet must:
1. Match the oracle's declared interface (port places) exactly.
2. Ship with at least one historical event definition it participates in.
3. Pass the full gate (§3).

Domain experts own their subnets after two merged PRs (CODEOWNERS). The core team owns
`core/` and `METHODOLOGY.md`; nobody owns the scorecard — it's generated.

## 3. Gating strategy — CI for models

A PR merges only if `validation/runner.py` shows:

- **No regression**: every previously-passing historical replay still passes
  (dip and recovery within its stated tolerance band).
- **Frozen-parameter discipline**: global parameters (K_SAT, ALPHA, HORIZON,
  utilization rule) may only change in a dedicated "methodology PR" (§4), never
  inside a data or subnet PR. Event inputs must be documented facts with citations.
- **Registered predictions**: a subnet PR adding a new historical event must include
  the predicted dip/recovery in the PR description *before* the maintainer runs it.
  (Enforced socially + by CI comparing PR text to output; imperfect, but it makes
  post-hoc fitting visible.)
- **Provenance lint**: every changed parameter has source + confidence; confidence-C
  parameters trigger a fog-priority entry rather than blocking merge.
- **Honesty artifact**: SCORECARD.md regenerates on merge and includes failures.
  A model whose public scorecard shows its misses (Sumitomo, twice) is the product.

## 4. The living methodology

`METHODOLOGY.md` is versioned like a spec (v0.1, v0.2 …) and changed by RFC:
an issue template stating the proposed change, what historical evidence motivates it,
and which scorecard entries it's expected to move. Methodology changes re-run the
*entire* validation suite from scratch — all events, including training events,
re-fit and re-frozen — so methodology drift can't quietly overfit event by event.

Contents (write-up order):
1. Why Petri nets: stocks, concurrency, deadlock/livelock, conservation.
2. Hierarchy: oracles as substitution transitions; fog-of-war epistemics —
   unexpanded regions are wrong in unknown directions (they've hidden both
   fragility and resilience in our own history; examples included).
3. Continuous relaxation and differentiability: why gradients, what
   d(resilience)/d($) means, mean vs tail objectives.
4. Adaptation: scarcity signals (runway + unfilled demand), what's endogenous
   (recovery time) and what isn't yet (price-mediated allocation — with the
   Sumitomo hysteresis as the open failing test).
5. Validation protocol: training vs holdout events, frozen parameters,
   registered predictions, the pessimism-bias caveat, scope boundaries
   (demand-side events currently out of scope).
6. Known limits, ranked, each tied to a failing or missing test.

## 5. Fog priority (expert outreach)

`fog/priority.py` ranks oracles by expected value of information: contribution of
each oracle's parameter uncertainty to variance in the headline outputs (gradient
rankings, replay predictions). Output is a public leaderboard: "expanding
packaging_resin is currently worth the most." Outreach then targets the top of the
list — recruitment by demonstrated need, not enthusiasm. An oracle expansion that
doesn't move any prediction gets deprioritized automatically, which protects
contributors' time as much as ours.

## 6. Sequencing

1. v0.1: port current code into the structure above; scorecard auto-generation.
2. Methodology write-up v0.1 (the six sections; the debugging history is source material).
3. Open two oracle bounties: packaging_resin (has a failing test) and fertilizer
   (the food-system slice; scarier and unexplored).
4. Demand/allocation RFC — the Sumitomo hysteresis fix; biggest known model gap.
5. Only then: the explorable site, reading everything from this repo.
