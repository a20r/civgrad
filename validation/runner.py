"""
validation/runner.py — historical replay engine (port of gsc_validate.py).

1. HISTORICAL VALIDATION: replay the 2022 neon shock.
   Feb 2022: Ingas (Mariupol) + Cryoin (Odesa) halt — roughly half of the
   world's semiconductor-grade neon purification CAPACITY (not stock).
   Historical record to match:
     - no major fab stoppages attributed to neon (chipmakers held ~6mo
       stockpiles, a lesson learned from the 2014 Crimea price spike)
     - alternative purification (China, Linde, POSCO) ramped over ~12-18mo
   Model: kill 50% of Purify_Ne capacity at t=6mo, capacity recovers with
   time constant tau=9mo. Compare WITH vs WITHOUT the 6-month stockpile.
   Validation = (a) dip is shallow with stockpile, (b) deep without,
   (c) recovery inside ~18 months.

2. EXPECTED-THROUGHPUT GRADIENT: loss = sum_s p_s * throughput(scenario_s)
   over a distribution of disruptions (core/gradients.py).

3. FROZEN-PROTOCOL HOLDOUTS: structure, K_SAT, and the 90%-utilization rule
   are FROZEN (set during the neon training event). Per-event inputs are
   documented historical facts, not tuned.

Time units: 1.0 = one month. DT = 0.1 month.
`python3 -m validation.runner` reproduces the gsc_validate.py baseline
(see validation/baseline_outputs.txt).
"""

import jax
import jax.numpy as jnp

from core.continuous import (Pre, Post, TRANSITIONS, PLACES, P, T_IDX, NT, NP_,
                             flows, cap0, burn_in, simulate_recovery)
from core.gradients import expected_throughput

DT = 0.1
MONTHS = 72
FAB = T_IDX["Fab"]


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

def calibrate_neon_v0():
    """Purification runs at ~90% utilization pre-shock (just-in-time);
    stocks don't accumulate — impose them. Lean = ~2wks buffer."""
    x_ss, F_star = burn_in(cap0, jnp.full(NP_, 1.0))
    cap_cal = cap0.at[T_IDX["Purify_Ne"]].set(F_star / 0.90)
    x_ss, F_star = burn_in(cap_cal, x_ss)   # re-equilibrate under calibrated cap
    x_ss         = x_ss.at[P["Ne_purified"]].set(0.5 * F_star)
    x_stockpiled = x_ss.at[P["Ne_purified"]].set(6.0 * F_star)   # post-2014 lesson
    x_lean       = x_ss
    return cap_cal, x_stockpiled, x_lean, F_star


NEON = dict(tidx=T_IDX["Purify_Ne"], kill=0.5, tau=9.0, t0=6.0)


def neon_section(cap_cal, x_stockpiled, x_lean, F_star):
    print(f"== 2022 neon shock replay ==")
    print(f"(steady-state fab flow {F_star:.3f}, purify utilization 90%;")
    print(f" 50% purification capacity lost, alt-supply tau=9mo)\n")
    results = {}
    for label, key, x0 in [("WITH ~6mo stockpile (history)", "stockpiled", x_stockpiled),
                           ("WITHOUT stockpile (counterfactual)", "lean", x_lean)]:
        _, traj = simulate_recovery(cap_cal, x0, **NEON)
        dip, rec = dip_and_recovery(traj, NEON["t0"])
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


EVENTS = [
    ("2011 Tohoku (wafers)",        "WaferSupply", 0.25, 4.0,  "Wafers",      2.0,
     "history: minor global impact"),
    ("1993 Sumitomo (packaging)",   "Package",     0.60, 6.0,  "Chips",       1.5,
     "history: price spike, brief pain, no catastrophe"),
    ("2019 photoresist (analog)",   "Purify_Ne",   0.90, 1.0,  "Ne_purified", 2.0,
     "history: non-event"),
]


def frozen_replay():
    c, xs, F = calibrate_frozen()
    print("\n== FROZEN-PROTOCOL HOLDOUT REPLAYS ==")
    print(f"(calibrated steady-state fab flow {F:.3f}; no per-event tuning)\n")
    results = {}
    for name, tname, kill, tau, bplace, bmonths, hist in EVENTS:
        x0 = xs.at[P[bplace]].set(bmonths * F)
        _, traj = simulate_recovery(c, x0, T_IDX[tname], kill, tau, t0=6.0)
        dip, rec = dip_and_recovery(traj, 6.0)
        results[name] = (dip, rec)
        rec_s = f"{rec:.0f}mo" if rec != float("inf") else "n/a"
        print(f"   {name}")
        print(f"      model: dip {100*dip:5.1f}%, recovery {rec_s:>5}   | {hist}")
    return results


def main():
    cap_cal, x_stockpiled, x_lean, F_star = calibrate_neon_v0()
    neon_section(cap_cal, x_stockpiled, x_lean, F_star)
    expected_gradient_section(cap_cal, x_stockpiled)
    frozen_replay()


if __name__ == "__main__":
    main()
