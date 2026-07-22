# AGENT_PROMPT_2 — civgrad: de-scope community infra, add essay + demo site

> **Archival note (2026-07-22).** Second bootstrap prompt, executed on this
> date (convention set by docs/AGENT_PROMPT.md). For ongoing work the
> operative guidance is AGENTS.md and PLAN.md.

Context: repo a20r/civgrad, bootstrapped per docs/AGENT_PROMPT.md. Direction
change: civgrad is a solo exploratory project — a lens, not a movement.
Remove coordination machinery; add a static essay + interactive demo.
All "Hard guardrails" from docs/AGENT_PROMPT.md still apply. The regression
oracle is validation/baseline_outputs.txt (jax 0.10.2, CPU, x64): those
numbers must still reproduce when you're done.

## Part A — Remove community infrastructure
1. Delete fog/ entirely (including fog/priority.py and its EVOI framing).
   Replace with map/_oracles/README.md: list each unexpanded oracle with its
   port places, contract, and one line on what expanding it would test.
   Note packaging_resin as the one with a failing test attached (Sumitomo).
2. README.md updates:
   - Reproduce block: drop the `python3 -m fog.priority` line.
   - Contributing: keep the two-door structure (data PR with citation /
     subnet PR passing the gate) but delete the fog.priority sentence and
     the "which expansion is worth the most" leaderboard framing. Add one
     line: "This is an exploratory solo project. Issues and PRs welcome; no
     roadmap promises."
3. PLAN.md: remove the fog-priority/expert-outreach section and any bounty
   language in the sequencing; replace the RFC process with one sentence:
   "Methodology changes bump the version in METHODOLOGY.md and re-run all
   validation from scratch." Renumber references (check README links to
   PLAN.md sections still point at the right ones).
4. AGENTS.md: trim references to removed machinery; everything else stands.
5. KEEP unchanged: SCORECARD.md generation (both v0 and v1 protocol rows,
   misses included), provenance schema, frozen-parameter discipline, the
   Sumitomo xfail, KNOWN_ISSUES.md, docs/ archive convention.

## Part B — Interactive demo (GitHub Pages, no build step)
Create site/ (plain HTML/JS/CSS, single page, no framework, no bundler);
configure Pages to serve from it (or gh-pages branch if repo settings
require — prefer /site on main).
1. Port the v1 adaptive continuous simulator to JS from core/continuous.py
   + core/adaptation.py: flow equations, K_SAT saturation, ALPHA/HORIZON,
   runway + unfilled-demand signals, restoration gating. ~12 ODEs, Euler
   steps — trivially client-side. Transcribe constants exactly from core/.
2. Regression-check the JS port: site/test/regression.mjs (node, no deps)
   runs the four historical events and asserts dip/recovery match the v1
   rows of SCORECARD.md within 1%. Add to .github/workflows/validate.yml.
   If JS and Python disagree, fix JS.
3. UI:
   - net diagram: places/transitions as the map; unexpanded oracles drawn
     as literal fog (blurred/hatched), tooltip = contract from
     map/_oracles/README.md
   - controls: disruption target dropdown, severity + stockpile-months +
     alpha sliders; preset buttons for the four historical events
   - live charts: fab flow + delivered flow over 72 months; dip % and
     recovery-months readout; a small v0-vs-v1 note linking the scorecard
   - scorecard table rendered from validation/events/*.yaml (both protocol
     rows, xfail and out-of-scope shown plainly)
4. Design restraint: readable typography, one accent color, fast load.
   Fog is the one visual flourish that matters.

## Part C — Essay scaffold (the living essay)
essay/essay.md, rendered as the site's main page with the demo embedded
between sections 5 and 6. Write 2-4 sentence STUBS per section from PLAN.md,
METHODOLOGY.md, SCORECARD.md, and the design-history conversation linked in
README ("Design history" section) — the author writes the prose; do NOT
ghostwrite the essay. Sections:
  1. An island with no off-island — reading Collapse; the metaphor stated
     honestly alongside the Hunt/Lipo challenge to the ecocide narrative
  2. Brittleness is not depletion — the chokepoint tour
  3. Why Petri nets — stocks, concurrency, livelock; busy futility
  4. The map and the fog — oracles hide fragility AND resilience (both
     demonstrated in this repo's own history)
  5. The gradient of collapse — investment as gradient ascent; the negative
     shipping gradient (rationing emerges); the boneyard result
  6. Making recovery emergent — the adaptation law and the five instructive
     failures, told straight
  7. The scorecard — three hits, one double miss (in opposite directions
     under the two protocols), the pessimism bias, and hysteresis:
     collapse-as-attractor appearing in the math uninvited
  8. What this is and isn't — "a lens for perspective, not a fix"; play
     with the demo
Add essay/VERSION (v0.1) and a changelog stub; the essay is versioned like
METHODOLOGY.md. When docs/TRANSCRIPT.md stops being a placeholder, the
essay's footer links it as the design history.

## Part D — Publication prep (light touch)
1. CITATION.cff (author: Alex Wallar; fill software metadata; repo URL).
2. Minimal pytest coverage for core/ invariants (conservation where
   expected, discrete-net reachability, one adaptation regression against
   baseline_outputs.txt values) so the repo is JOSS-submittable later.
   Do not write a paper.

## Finishing
- Archive this prompt at docs/AGENT_PROMPT_2.md (convention set by the
  first bootstrap).
- Granular commits per part. Final check: python3 -m validation.runner
  reproduces baseline; node site/test/regression.mjs passes; Pages builds.
