# Instructions for AI agents working in this repo

civgrad is a research repo with a validation discipline. Read this before
touching anything; it is short on purpose.

1. **Read `PLAN.md` and `METHODOLOGY.md` first.** PLAN.md is the
   authoritative plan (structure, contribution model, gating); METHODOLOGY.md
   is the versioned spec of what the model claims. Where guidance conflicts,
   PLAN.md wins.

2. **Frozen-parameter discipline.** `K_SAT`, `ALPHA`, `HORIZON`, and the
   90%-utilization calibration rule are frozen. Never change them — or any
   model equation — outside a dedicated methodology PR (PLAN.md §4), which
   re-runs and re-freezes the entire validation suite. Data PRs and subnet
   PRs must not touch them.

3. **Never tune to a historical event.** Event yamls in `validation/events/`
   are documented facts, not fitting targets. `sumitomo_1993` is `xfail` on
   purpose — a documented miss. Do not make it pass; if your change makes it
   pass, that is a methodology-level event and CI will fail loudly so it gets
   reviewed as one. Registered predictions go in the PR description *before*
   running the replay.

4. **Provenance on every parameter.** Every numeric parameter carries
   `{value, units, source, confidence: A|B|C, date}`
   (`map/semiconductors/net.yaml`). No source, no merge. Never fabricate
   citations, historical values, or confidence grades; confidence is assigned
   by reviewers, not authors. Unknown means `confidence: C` with
   `source: "session estimate — needs citation"`.

5. **Consult the design history before reversing decisions.**
   `docs/TRANSCRIPT.md` records why the model is shaped this way — the Petri
   net framing, the oracle/fog hierarchy, the relaxation, the adaptation law,
   and five instructive failures with their fixes. If something looks wrong,
   check there, then file it in `KNOWN_ISSUES.md` rather than silently
   changing behavior. `validation/baseline_outputs.txt` is the regression
   oracle: refactors must reproduce every number in it
   (`python3 -m validation.runner` must exit 0).
