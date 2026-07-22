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

Time units: 1.0 = one month. DT = 0.1 month.
`python3 -m validation.runner` reproduces the gsc_validate.py baseline
(see validation/baseline_outputs.txt).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from core.continuous import (Pre, Post, TRANSITIONS, PLACES, P, T_IDX, NT, NP_,
                             flows, cap0, burn_in, simulate_recovery)
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
    cap_cal = cap0.at[T_IDX[ev["transition"]]].set(F_star / 0.90)
    x_ss, F_star = burn_in(cap_cal, x_ss)   # re-equilibrate under calibrated cap
    x_ss         = x_ss.at[P["Ne_purified"]].set(lean_months * F_star)
    x_stockpiled = x_ss.at[P["Ne_purified"]].set(stock_months * F_star)  # post-2014 lesson
    x_lean       = x_ss
    return cap_cal, x_stockpiled, x_lean, F_star


def neon_section(ev, cap_cal, x_stockpiled, x_lean, F_star):
    shock = dict(tidx=T_IDX[ev["transition"]], kill=ev["capacity_lost"],
                 tau=ev["ramp_tau_months"], t0=ev["shock_month"])
    print(f"== {ev['year']} neon shock replay ==")
    print(f"(steady-state fab flow {F_star:.3f}, purify utilization 90%;")
    print(f" {round(100*shock['kill'])}% purification capacity lost, "
          f"alt-supply tau={shock['tau']:.0f}mo)\n")
    results = {}
    stock_months = ev["buffers"]["Ne_purified"]
    for label, key, x0 in [(f"WITH ~{stock_months:.0f}mo stockpile (history)", "stockpiled", x_stockpiled),
                           ("WITHOUT stockpile (counterfactual)", "lean", x_lean)]:
        _, traj = simulate_recovery(cap_cal, x0, **shock)
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
            c = c.at[T_IDX[tname]].set(F / 0.90)
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
        _, traj = simulate_recovery(c, x0, T_IDX[ev["transition"]], kill, tau, t0=t0)
        dip, rec = dip_and_recovery(traj, t0)
        results[ev["name"]] = (dip, rec)
        rec_s = f"{rec:.0f}mo" if rec != float("inf") else "n/a"
        print(f"   {ev['short']}")
        print(f"      model: dip {100*dip:5.1f}%, recovery {rec_s:>5}   | history: {ev['history']}")
    return results


def main():
    events = load_events()
    neon = next(e for e in events if e["role"] == "training")
    holdouts = [e for e in simulatable(events) if e["role"] == "holdout"]

    cap_cal, x_stockpiled, x_lean, F_star = calibrate_neon_v0(neon)
    neon_section(neon, cap_cal, x_stockpiled, x_lean, F_star)
    expected_gradient_section(cap_cal, x_stockpiled)
    frozen_replay(holdouts)


if __name__ == "__main__":
    main()
