# Known issues

Oddities found while porting, filed instead of fixed (AGENTS.md: never trade
behavior for tidiness silently). None of these changes numbers; the removals
listed were proven side-effect-free against `validation/baseline_outputs.txt`.

1. **Dead code removed in port** — `gsc_validate.dip_and_recovery` assigned
   `rec = jnp.argmax(...)` and immediately overwrote it; `gsc_validate.burn_in`
   ran a full `simulate()` whose result was discarded (pure, so no state
   effect); `frozen_replay` had an unused `cap = dict(cal=cap0)`. All three
   were dropped in the port to `validation/runner.py` / `core/continuous.py`;
   outputs verified byte-identical to baseline.

2. **Misplaced docstring kept** — `core/continuous.py::simulate` has its
   "docstring" after the first statement, so it is a no-op string expression,
   not a docstring. Kept as-is to keep the port literal.

3. **Redundant initialization kept** — `x0 = jnp.full(NP_, 1.0).at[P["EUV_tools"]].set(1.0)`
   sets a value that is already 1.0. Kept as-is.

4. **Doc/code mismatch in the original** — gsc_validate.py's module docstring
   said the neon shock hits at t=12mo; the code uses `t0=6.0`. The port's
   docs and `validation/events/neon_2022.yaml` follow the code
   (`shock_month: 6.0`).

5. **Recovery metric degenerates on shallow dips** — when the dip never
   breaches 95% of pre-shock, "months to 95% recovery" returns the time of
   the minimum instead (e.g. the neon v0 stockpiled row's "16 mo"). Noted in
   SCORECARD.md; a metric fix is a methodology PR because it changes scored
   numbers.

6. **`brittleness()` skips source places** — the comment says they are
   "scored elsewhere", but no elsewhere exists yet. Source-place brittleness
   (mines, oracles) is currently unscored.

7. **RESOLVED — the map is now the source of truth for values.**
   `core/map_loader.py` builds parameter values (tokens, rebuild years,
   regions, arc weights, capacities, oracle contracts, frozen globals) from
   `map/semiconductors/net.yaml`, so a data PR that edits the yaml changes
   the model directly and must pass the validation gate; nothing is mirrored
   by hand anymore. Residual, by design: the ordered name lists (the
   engine's marking/capacity index layout) remain in `core/net.py` and
   `core/continuous.py` and are asserted against the yaml at import —
   adding or removing places/transitions touches both files consciously
   (that's a subnet PR).

8. **`Package` consumes nothing but chips** — packaging materials (resin,
   substrates) are not modeled, which is exactly why Sumitomo 1993 is
   invisible to protocol v0 through the fab observable. Tracked by the xfail
   and `map/_oracles/packaging_resin.yaml`, not fixable by a data PR.

9. **Documented-value discrepancies recorded, not applied** — the 2026-07-27
   provenance pass found two event inputs whose documented values disagree
   with the frozen session estimates: `tohoku_2011` buffers.Wafers 2.0 mo
   (documented: TSMC ~30 days, "mere weeks" industry-wide) and `neon_2022`
   buffers.Ne_purified 6.0 mo (documented as the UPPER end — largest
   producers only; smaller makers 4-6 weeks). Both are recorded in the
   yamls' `value_notes` instead of changed, because changing them moves
   scored numbers: each is a candidate data PR that must register predicted
   dip/recovery in the PR description BEFORE the replay runs (AGENTS.md
   rule 3).

10. **The v1 hysteresis narration corrected by measurement** — the design
    session narrated the Sumitomo never-recovery as "the tool fleet wears
    down unreplaced." The traced anatomy (PRICE_EXPERIMENT.md) is different:
    the bloated Chips buffer masks Package's destroyed capacity inside the
    flow-based restoration signal (fat input saturation substitutes for
    missing capacity), restoration stalls ~8% short, and deliveries settle
    ~10% low with every alarm silent. Same conclusion (missing reallocation
    mechanism), different mechanism — essay text updated to match the
    measurement.

11. **Tail analysis is horizon-truncated** — the CVaR(30%) analysis
    (POLICY.md §2) prices catastrophes within the 72-month replay horizon,
    so the Zeiss-knockout scenario (tau ~120 mo) barely registers: the
    refurb loop carries the system inside the window and the real pain sits
    beyond it. Extending the horizon for slow catastrophes is named future
    work; do not read the tail table as "optics loss is fine."
