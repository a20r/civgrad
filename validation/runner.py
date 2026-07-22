"""
validation/runner.py — historical replay engine (port of gsc_validate.py).

Event definitions are DATA: validation/events/*.yaml hold the documented
facts of each historical event (transition hit, capacity lost, buffers,
observable, one-line record, acceptance band). This module holds the frozen
replay protocols and never hardcodes event facts.

1. HISTORICAL VALIDATION: replay the 2022 neon shock (training event).
   Kill capacity_lost of the event's transition at shock_month; capacity
   recovers with the event's documented ramp tau. Compare WITH vs WITHOUT
   the historical stockpile.

2. EXPECTED-THROUGHPUT GRADIENT: loss = sum_s p_s * throughput(scenario_s)
   over a distribution of disruptions (core/gradients.py).

3. FROZEN-PROTOCOL HOLDOUTS (protocol v0, imposed-tau): structure, K_SAT,
   and the 90%-utilization rule are FROZEN (set during the neon training
   event). Per-event inputs come from the event yamls — documented
   historical facts, not tuned. Protocol v0 observes fab flow; where a
   yaml carries a `tau_replay` block, those inputs override (e.g. the
   photoresist analog replays the feared fraction).

4. ADAPTIVE-PROTOCOL REPLAYS (protocol v1): impulse disruption, recovery
   emergent from the frozen adaptation law (core/adaptation.py, ALPHA
   frozen). Observes each event's yaml observable (fab | delivered).

5. SCORECARD + GATES: regenerates SCORECARD.md from the replay results and
   enforces the acceptance bands:
     - a `status: pass` row outside its band is a REGRESSION -> exit 1
     - a `status: xfail` row must FAIL its band; if it ever passes, that is
       a methodology-level event -> loud banner + exit 1 (never silently
       accept — see PLAN.md §4 and the event yaml)
     - `status: out-of-scope` events are listed, never simulated
   A band is expected ± tolerance on dip_pct and recovery_months; an
   infinite recovery fails any band.

Time units: 1.0 = one month. DT = 0.1 month.
`python3 -m validation.runner` reproduces the gsc_validate.py baseline
numbers, then the gsc_adapt.py replay numbers (see
validation/baseline_outputs.txt), then writes SCORECARD.md.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from core.continuous import (Pre, Post, TRANSITIONS, PLACES, P, T_IDX, NT, NP_,
                             flows, cap0, burn_in, simulate_recovery, UTILIZATION)
from core.gradients import expected_throughput

DT = 0.1
MONTHS = 72
FAB = T_IDX["Fab"]

EVENTS_DIR = Path(__file__).resolve().parent / "events"


def load_events():
    """All event yamls, sorted by their `order` field."""
    events = []
    for f in sorted(EVENTS_DIR.glob("*.yaml")):
        with open(f) as fh:
            events.append(yaml.safe_load(fh))
    return sorted(events, key=lambda e: e["order"])


def simulatable(events):
    return [e for e in events if e["status"] != "out-of-scope"]


def dip_and_recovery(traj, t0):
    """Depth of throughput dip vs pre-shock level, and months to 95% recovery."""
    i0 = int(t0 / DT)
    pre = jnp.mean(traj[i0 - int(6 / DT):i0])
    post = traj[i0:]
    dip = float(1.0 - jnp.min(post) / pre)
    # first index AFTER the minimum that crosses back over 95%
    imin = int(jnp.argmin(post))
    after = post[imin:]
    crossed = jnp.argmax(after > 0.95 * pre)
    rec_months = float((imin + crossed) * DT) if float(jnp.max(after)) > 0.95 * float(pre) else float("inf")
    return dip, rec_months


# ---------------- 1. neon shock replay (training event) ----------------

def calibrate_neon_v0(ev):
    """Purification runs at ~90% utilization pre-shock (just-in-time);
    stocks don't accumulate — impose them. Lean = ~2wks buffer."""
    lean_months = ev["counterfactual_buffer_months"]
    stock_months = ev["buffers"]["Ne_purified"]
    x_ss, F_star = burn_in(cap0, jnp.full(NP_, 1.0))
    cap_cal = cap0.at[T_IDX[ev["transition"]]].set(F_star / UTILIZATION)
    x_ss, F_star = burn_in(cap_cal, x_ss)   # re-equilibrate under calibrated cap
    x_ss         = x_ss.at[P["Ne_purified"]].set(lean_months * F_star)
    x_stockpiled = x_ss.at[P["Ne_purified"]].set(stock_months * F_star)  # post-2014 lesson
    x_lean       = x_ss
    return cap_cal, x_stockpiled, x_lean, F_star


def neon_section(ev, cap_cal, x_stockpiled, x_lean, F_star):
    shock = dict(tidx=T_IDX[ev["transition"]], kill=ev["capacity_lost"],
                 tau=ev["ramp_tau_months"], t0=ev["shock_month"])
    print(f"== {ev['year']} neon shock replay ==")
    print(f"(steady-state fab flow {F_star:.3f}, purify utilization {round(100*UTILIZATION)}%;")
    print(f" {round(100*shock['kill'])}% purification capacity lost, "
          f"alt-supply tau={shock['tau']:.0f}mo)\n")
    results = {}
    stock_months = ev["buffers"]["Ne_purified"]
    for label, key, x0 in [(f"WITH ~{stock_months:.0f}mo stockpile (history)", "stockpiled", x_stockpiled),
                           ("WITHOUT stockpile (counterfactual)", "lean", x_lean)]:
        _, traj = simulate_recovery(cap_cal, x0, months=MONTHS, **shock)
        dip, rec = dip_and_recovery(traj, shock["t0"])
        results[key] = (dip, rec)
        print(f"   {label}:")
        print(f"      throughput dip: {100*dip:5.1f}%   recovery to 95%: {rec:.0f} months")
    print("\n   historical record: no fab stoppages; supply normalized ~12-18mo.")
    return results


# ---------------- 2. expected-throughput gradients ----------------

def expected_gradient_section(cap_cal, x_stockpiled):
    g = jax.grad(expected_throughput, argnums=0)(cap_cal, x_stockpiled)
    gs = jax.grad(expected_throughput, argnums=1)(cap_cal, x_stockpiled)

    print("\n== E[throughput] gradients over disruption distribution ==")
    print("CAPACITY:")
    for i in jnp.argsort(-g):
        print(f"   {float(g[i]):9.3f}  {TRANSITIONS[int(i)][0]}")
    print("STOCKPILES:")
    for i in jnp.argsort(-gs)[:6]:
        print(f"   {float(gs[i]):9.3f}  {PLACES[int(i)]}")


# ---------------- 3. FROZEN-PROTOCOL HOLDOUTS ----------------
# Structure, K_SAT, and the 90%-utilization rule are FROZEN (set during the
# neon training event). Per-event inputs below are documented historical
# facts, not tuned: (transition hit, fraction lost, ramp tau, buffer months).

def calibrate_frozen():
    """Uniform calibration rule: every direct fab-input source runs at 90% util;
    JIT default: thin (0.5mo) buffers on intermediate stocks."""
    c = cap0
    x = jnp.full(NP_, 1.0)
    for _ in range(2):                      # calibrate -> re-equilibrate -> recal
        xs, F = burn_in(c, x)
        for tname in ["Purify_Ne", "WaferSupply", "Refine_Ga"]:
            c = c.at[T_IDX[tname]].set(F / UTILIZATION)
        x = xs
    xs, F = burn_in(c, x)
    for pname in ["Ne_purified", "Wafers", "Ga_refined", "Chips", "Pkg"]:
        xs = xs.at[P[pname]].set(0.5 * F)
    return c, xs, F


def v0_inputs(ev):
    """Protocol-v0 replay inputs for an event: yaml facts, with any
    `tau_replay` overrides applied (e.g. photoresist replays the feared
    fraction). Returns (kill, tau, buffers, t0)."""
    over = ev.get("tau_replay") or {}
    kill = over.get("capacity_lost", ev["capacity_lost"])
    return kill, ev["ramp_tau_months"], ev["buffers"], ev["shock_month"]


def frozen_replay(events):
    c, xs, F = calibrate_frozen()
    print("\n== FROZEN-PROTOCOL HOLDOUT REPLAYS ==")
    print(f"(calibrated steady-state fab flow {F:.3f}; no per-event tuning)\n")
    results = {}
    for ev in events:
        kill, tau, buffers, t0 = v0_inputs(ev)
        x0 = xs
        for bplace, bmonths in buffers.items():
            x0 = x0.at[P[bplace]].set(bmonths * F)
        _, traj = simulate_recovery(c, x0, T_IDX[ev["transition"]], kill, tau, t0=t0,
                                    months=MONTHS)
        dip, rec = dip_and_recovery(traj, t0)
        results[ev["name"]] = (dip, rec)
        rec_s = f"{rec:.0f}mo" if rec != float("inf") else "n/a"
        print(f"   {ev['short']}")
        print(f"      model: dip {100*dip:5.1f}%, recovery {rec_s:>5}   | history: {ev['history']}")
    return results


# ---------------- 4. adaptive-protocol replays ----------------

OBSERVE = {"fab": "fab", "delivered": "ship"}


def adaptive_replay(adaptation, ev, kill=None, buffers=None):
    """Replay an event under the frozen adaptation law (impulse disruption,
    emergent recovery). Equivalent to adaptation.run() for single-buffer
    events; supports the yaml's {place: months} buffers dict and the event's
    own shock_month/horizon rather than the module defaults."""
    kill = ev["capacity_lost"] if kill is None else kill
    buffers = ev["buffers"] if buffers is None else buffers
    t0 = ev["shock_month"]
    x0 = adaptation.X_JIT
    for place, months in buffers.items():
        x0 = x0.at[P[place]].set(months * adaptation.F)
    fab, ship = adaptation.simulate(adaptation.c, x0, adaptation.X_REF,
                                    T_IDX[ev["transition"]], kill, adaptation.ALPHA,
                                    t0=t0, months=MONTHS)
    return adaptation.dip_recovery(fab if OBSERVE[ev["observable"]] == "fab" else ship, t0)


def adaptive_section(neon, holdouts):
    import core.adaptation as adaptation  # deferred: import runs its frozen calibration

    print(f"\n== ADAPTIVE-PROTOCOL REPLAYS (ALPHA frozen at {adaptation.ALPHA}, tau emergent) ==")
    results, counterfactuals = {}, []
    for ev in [neon] + holdouts:
        d, r = adaptive_replay(adaptation, ev)
        results[ev["name"]] = (100 * d, r)
        label = f"{ev['short']}:"
        print(f"   {label:<28} dip {100*d:5.1f}%, rec {r:5.1f}mo | history: {ev['history']}")
        if "counterfactual_buffer_months" in ev:
            d2, r2 = adaptive_replay(adaptation, ev,
                                     buffers={p: ev["counterfactual_buffer_months"]
                                              for p in ev["buffers"]})
            counterfactuals.append((f"{ev['short']}, WITHOUT stockpile", "v1 adaptive",
                                    100 * d2, r2, "same counterfactual, under the adaptation law"))
        if "feared_capacity_lost" in ev:
            f = ev["feared_capacity_lost"]
            d2, r2 = adaptive_replay(adaptation, ev, kill=f)
            print(f"     feared ({round(100*f)}%) counterfactual: dip {100*d2:5.1f}%, rec {r2:5.1f}mo")
            counterfactuals.append((f"{ev['short']}, feared ({round(100*f)}%)", "v1 adaptive",
                                    100 * d2, r2, "what the fear implied, had the curbs bitten fully"))
    return results, counterfactuals


# ---------------- 5. scorecard + gates ----------------

PROTO_V0 = "v0 imposed-τ"
PROTO_V1 = "v1 adaptive"


def in_band(dip_pct, rec_months, expected):
    """True iff both metrics sit inside expected ± tolerance. An infinite
    recovery fails every band."""
    tol = expected["tolerance"]
    ok_dip = abs(dip_pct - expected["dip_pct"]) <= tol["dip_pct"]
    ok_rec = (rec_months != float("inf")
              and abs(rec_months - expected["recovery_months"]) <= tol["recovery_months"])
    return ok_dip and ok_rec


def row_status(ev, dip_pct, rec_months):
    """Returns (display_status, gate_failure_or_None)."""
    ok = in_band(dip_pct, rec_months, ev["expected"])
    if ev["status"] == "pass":
        if ok:
            return "✅ pass", None
        return "🚨 REGRESSION (was pass)", (
            f"REGRESSION: '{ev['name']}' left its acceptance band "
            f"(dip {dip_pct:.1f}%, recovery {rec_months}, "
            f"band {ev['expected']['dip_pct']}±{ev['expected']['tolerance']['dip_pct']}% / "
            f"{ev['expected']['recovery_months']}±{ev['expected']['tolerance']['recovery_months']}mo)")
    else:  # xfail — the documented miss MUST keep failing
        if not ok:
            return "❌ xfail (documented miss)", None
        return "🚨 UNEXPECTED PASS", (
            f"METHODOLOGY-LEVEL EVENT: xfail event '{ev['name']}' now PASSES its "
            f"historical band. Either a bug or a genuine modeling advance; both "
            f"require a methodology PR (PLAN.md §4) — re-run the full suite from "
            f"scratch, re-freeze, and update the yaml consciously. Failing CI so "
            f"this cannot merge silently.")


def fmt_dip(dip_pct):
    return f"{dip_pct:.1f}%"


def fmt_rec(rec_months):
    return f"never ({MONTHS} mo horizon)" if rec_months == float("inf") else f"{rec_months:.1f} mo"


def build_scorecard(events, scored, counterfactuals):
    """scored: {(event_name, protocol) -> (dip_pct, rec_months)}.
    Returns (markdown, gate_failures)."""
    failures, rows = [], []
    for ev in events:
        if ev["status"] == "out-of-scope":
            rows.append((ev, "—", None, None,
                         f"⛔ out-of-scope ({ev['scope_reason']})"))
            continue
        for proto in (PROTO_V0, PROTO_V1):
            dip_pct, rec = scored[(ev["name"], proto)]
            status, failure = row_status(ev, dip_pct, rec)
            if failure:
                failures.append(failure)
            rows.append((ev, proto, dip_pct, rec, status))

    md = []
    md.append("# civgrad — historical replay scorecard\n")
    md.append("<!-- AUTO-GENERATED by `python3 -m validation.runner` — do not edit by hand. -->\n")
    md.append("Misses are displayed as prominently as hits: a model whose public scorecard")
    md.append("shows its misses is the product (PLAN.md §3). Nothing here is tuned to an")
    md.append("event; per-event inputs are documented facts from `validation/events/*.yaml`.\n")
    md.append(f"**Protocols.** `{PROTO_V0}`: capacity loss with an imposed exponential recovery")
    md.append("at the event's documented ramp τ; observes fab flow. "
              f"`{PROTO_V1}`: capacity loss")
    md.append("is a pure impulse; recovery is emergent from the frozen adaptation law")
    md.append("(ALPHA=0.06, HORIZON=6 mo); observes the event's own observable (fab or")
    md.append("delivered). *dip* = deepest drop of the observed flow vs its pre-shock mean")
    md.append("(negative dip: the flow never fell — it rose). *recovery* = months from shock")
    md.append("until the flow re-crosses 95% of pre-shock; with a dip inside 5% the metric")
    md.append("degenerates to time-of-minimum.\n")
    md.append("| event | protocol | model dip | model recovery | historical record | status |")
    md.append("|---|---|---:|---:|---|---|")
    for ev, proto, dip_pct, rec, status in rows:
        dip_s = "—" if dip_pct is None else fmt_dip(dip_pct)
        rec_s = "—" if rec is None else fmt_rec(rec)
        md.append(f"| {ev['title']} | {proto} | {dip_s} | {rec_s} | {ev['history']} | {status} |")
    md.append("")
    md.append("**Acceptance bands** (expected ± tolerance, encoded from the qualitative record")
    md.append("*before* scoring — see each yaml's `band_note`):\n")
    for ev in events:
        if ev["status"] == "out-of-scope":
            continue
        e, t = ev["expected"], ev["expected"]["tolerance"]
        md.append(f"- `{ev['name']}` ({ev['status']}): dip {e['dip_pct']}±{t['dip_pct']}%, "
                  f"recovery {e['recovery_months']}±{t['recovery_months']} mo")
    md.append("")
    md.append("## The documented miss: Sumitomo 1993 — xfail, twice\n")
    sumitomo = next(e for e in events if e["status"] == "xfail")
    md.append(sumitomo["xfail_reason"].strip() + "\n")
    md.append("## Out of scope\n")
    for ev in events:
        if ev["status"] == "out-of-scope":
            md.append(f"- **{ev['title']}** — {ev['history']}. Out of scope: "
                      f"{ev['scope_reason']}; see the demand/allocation work (PLAN.md §5).")
    md.append("")
    md.append("## Counterfactuals (informational, never scored)\n")
    md.append("| replay | protocol | model dip | model recovery | note |")
    md.append("|---|---|---:|---:|---|")
    for label, proto, dip_pct, rec, note in counterfactuals:
        md.append(f"| {label} | {proto} | {fmt_dip(dip_pct)} | {fmt_rec(rec)} | {note} |")
    md.append("")
    md.append("**Pessimism-bias caveat.** The model has no price-mediated demand allocation,")
    md.append("substitution, or design-around response, so it systematically overstates how")
    md.append("bad supply shocks get and how long they last (Sumitomo v1 is the exhibit).")
    md.append("Rankings of *where* fragility concentrates are more trustworthy than absolute")
    md.append("dip depths. See METHODOLOGY.md §5-6.")
    md.append("")
    return "\n".join(md), failures


def main():
    events = load_events()
    neon = next(e for e in events if e["role"] == "training")
    holdouts = [e for e in simulatable(events) if e["role"] == "holdout"]
    scored, counterfactuals = {}, []

    # protocol v0: neon replay + expected gradients + frozen holdouts
    cap_cal, x_stockpiled, x_lean, F_star = calibrate_neon_v0(neon)
    neon_res = neon_section(neon, cap_cal, x_stockpiled, x_lean, F_star)
    scored[(neon["name"], PROTO_V0)] = tuple([100 * neon_res["stockpiled"][0],
                                              neon_res["stockpiled"][1]])
    counterfactuals.append((f"{neon['short']}, WITHOUT stockpile", PROTO_V0,
                            100 * neon_res["lean"][0], neon_res["lean"][1],
                            "the 2014 lesson: what lean inventories would have cost"))
    expected_gradient_section(cap_cal, x_stockpiled)
    for name, (dip, rec) in frozen_replay(holdouts).items():
        scored[(name, PROTO_V0)] = (100 * dip, rec)

    # protocol v1: adaptive replays (emergent recovery)
    v1_results, v1_counter = adaptive_section(neon, holdouts)
    for name, dr in v1_results.items():
        scored[(name, PROTO_V1)] = dr
    counterfactuals.extend(v1_counter)

    # scorecard + gates
    md, failures = build_scorecard(events, scored, counterfactuals)
    out = Path(__file__).resolve().parents[1] / "SCORECARD.md"
    out.write_text(md)
    by_name = {e["name"]: e for e in events}
    statuses = [row_status(by_name[n], *dr)[0] for (n, _), dr in scored.items()]
    n_pass = sum(1 for s in statuses if s.startswith("✅"))
    n_xfail = sum(1 for s in statuses if s.startswith("❌"))
    n_oos = sum(1 for e in events if e["status"] == "out-of-scope")
    print(f"\n== SCORECARD ==")
    print(f"   wrote {out}")
    print(f"   scored rows: {n_pass} pass, {n_xfail} xfail (documented misses), "
          f"{len(statuses) - n_pass - n_xfail} gate failure(s); {n_oos} out-of-scope event(s) listed")
    if failures:
        print("\n" + "🚨" * 3)
        for f in failures:
            print(f)
        print("🚨" * 3)
        sys.exit(1)
    print("   gates: OK — no regressions; xfail entries still fail (as they must)")


if __name__ == "__main__":
    main()
