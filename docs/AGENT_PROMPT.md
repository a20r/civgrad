# AGENT_PROMPT — civgrad repo bootstrap

You are setting up a public research repo from a handoff zip. The zip contains
working code, a plan, and design history. Your job is structure, tooling, and
documentation — NOT modeling. Follow PLAN.md exactly; where this prompt and
PLAN.md conflict, PLAN.md wins.

## Step 0 — Baseline before touching anything
Run, in order, and save complete stdout to `validation/baseline_outputs.txt`:
    python3 gsc_net.py
    python3 gsc_grad.py
    python3 gsc_validate.py
    python3 gsc_adapt.py
(deps: `pip install jax`). These outputs are the regression oracle. Every number
in them must be reproducible after your refactor. If any refactored run differs
numerically, you introduced a bug — fix the refactor, never the number.

## Step 1 — Restructure per PLAN.md §1
Port the four scripts into the layout in PLAN.md §1:
  - gsc_net.py      -> core/net.py            (discrete PN + reachability)
  - gsc_grad.py     -> core/continuous.py + core/gradients.py
  - gsc_adapt.py    -> core/adaptation.py
  - gsc_validate.py -> validation/ (event definitions -> validation/events/*.yaml,
                       replay logic -> validation/runner.py)
Move hard-coded event facts (kill fractions, buffer months, historical notes)
into `validation/events/*.yaml` with fields:
  {name, year, transition, capacity_lost, buffers: {place: months},
   observable: fab|delivered, history: <one-line record>, sources: [urls],
   role: training|holdout, expected: {dip_pct, recovery_months, tolerance}}
Sumitomo's expected entry is marked `status: xfail` — it is a DOCUMENTED MISS
(see PLAN.md §3 "Honesty artifact" and §4 item 4). Do not tune anything to make
it pass. It is the failing test that motivates the demand/allocation RFC.

## Step 2 — Scorecard generation
`validation/runner.py` runs every event yaml and regenerates SCORECARD.md:
a table of event | model dip/recovery | historical record | status
(pass / xfail / out-of-scope). Include the 2021 chip crunch as a row with
status `out-of-scope (no demand dynamics)`. Misses are displayed as
prominently as hits.

## Step 3 — Map + fog stubs
- map/semiconductors/net.yaml: transcribe places/transitions/arcs from
  core/net.py with the provenance schema from PLAN.md §2. For every numeric
  parameter set `confidence: C, source: "session estimate — needs citation"`.
  Do not invent citations.
- map/_oracles/: one yaml per unexpanded oracle (fertilizer, packaging_resin,
  optics, mining, wafer_supply) with interface places + contract strings from
  the code.
- fog/priority.py: stub that lists confidence-C parameter counts per subnet as
  a placeholder EVOI ranking, with a TODO pointing at PLAN.md §5.

## Step 4 — Licenses & docs
- LICENSE: Apache-2.0 (all code).
- LICENSE-DOCS: CC-BY-4.0, applying to METHODOLOGY.md and docs/.
- METHODOLOGY.md: create v0.1 as the six-section skeleton from PLAN.md §4,
  each section 2-5 sentences drawn from PLAN.md and docs/TRANSCRIPT.md. Mark
  it `version: 0.1 — skeleton; expanded by RFC`.
- AGENTS.md at repo root: instructions for future AI agents working here —
  (1) read PLAN.md and METHODOLOGY.md first; (2) frozen-parameter discipline:
  never change K_SAT/ALPHA/HORIZON/utilization outside a methodology PR;
  (3) never tune to a historical event; registered predictions go in the PR
  description before running; (4) provenance required on every parameter;
  (5) docs/TRANSCRIPT.md is the design history — consult it for why decisions
  were made before proposing to reverse them.
- docs/TRANSCRIPT.md: if the placeholder is still present, leave it and add a
  README note that the design-history transcript is pending upload.
- README.md: project summary (from PLAN.md preamble), the scorecard table
  inlined, the pessimism-bias caveat stated plainly, contribution pointer,
  license badges.

## Step 5 — CI
.github/workflows/validate.yml: on PR, install jax, run validation/runner.py,
fail on any regression vs expected values in the event yamls (xfail entries
must fail; if Sumitomo starts *passing*, that's a methodology-level event —
flag it loudly rather than silently accepting).

## Step 6 — Commits
Granular history: baseline capture / restructure / events-as-data / scorecard /
map+fog / docs+licenses / CI. Verify `python3 -m validation.runner` reproduces
baseline numbers as the final check before finishing.

## Hard guardrails
- Do not modify model equations, parameters, or event facts.
- Do not fix xfail tests.
- Do not fabricate citations, historical values, or confidence grades.
- If something in the code looks wrong, file an issue in a `KNOWN_ISSUES.md`
  instead of changing behavior.

## Naming
The project is named **civgrad** ("civilization gradient"). Use civgrad in all
docs, README, and repo-facing text. Legacy internal identifiers in code output
(e.g. the net name "GSC:semiconductors", filenames gsc_*.py before Step 1's
restructure) may remain as-is — renaming them is cosmetic and risks breaking
the baseline regression check. Never trade numerical reproducibility for naming.
