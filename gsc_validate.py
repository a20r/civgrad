"""
gsc_validate.py — two upgrades to gsc_grad:

1. HISTORICAL VALIDATION: replay the 2022 neon shock.
   Feb 2022: Ingas (Mariupol) + Cryoin (Odesa) halt — roughly half of the
   world's semiconductor-grade neon purification CAPACITY (not stock).
   Historical record to match:
     - no major fab stoppages attributed to neon (chipmakers held ~6mo
       stockpiles, a lesson learned from the 2014 Crimea price spike)
     - alternative purification (China, Linde, POSCO) ramped over ~12-18mo
   Model: kill 50% of Purify_Ne capacity at t=12mo, capacity recovers with
   time constant tau=9mo. Compare WITH vs WITHOUT the 6-month stockpile.
   Validation = (a) dip is shallow with stockpile, (b) deep without,
   (c) recovery inside ~18 months.

2. EXPECTED-THROUGHPUT GRADIENT: loss = sum_s p_s * throughput(scenario_s)
   over a distribution of disruptions, so investment rankings stop being
   hostage to one hand-picked catastrophe.

Time units: 1.0 = one month. DT = 0.1 month.
"""

import jax
import jax.numpy as jnp
from gsc_grad import (Pre, Post, Read, TRANSITIONS, PLACES, P, T_IDX,
                      NT, NP_, flows, cap0)

DT = 0.1
MONTHS = 72
STEPS = int(MONTHS / DT)
FAB = T_IDX["Fab"]


def simulate(cap_base, x0, tidx, kill, tau, t0):
    """Disruption kills `kill` fraction of transition `tidx`'s capacity at
    month t0; capacity recovers exponentially with time constant tau
    (alternative suppliers ramping). Returns (total, fab_flow_trajectory)."""
    def step(carry, k):
        x, tot = carry
        t = k * DT
        since = t - t0
        recov = 1.0 - kill * jnp.exp(-jnp.maximum(since, 0.0) / tau)
        mult = jnp.where(since >= 0.0, recov, 1.0)
        cap = cap_base.at[tidx].mul(mult)
        v = flows(x, cap)
        x = jnp.clip(x + DT * ((Post - Pre).T @ v), 0.0)
        return (x, tot + DT * v[FAB]), v[FAB]
    (_, tot), traj = jax.lax.scan(step, (x0, 0.0), jnp.arange(STEPS))
    return tot, traj


def dip_and_recovery(traj, t0):
    """Depth of throughput dip vs pre-shock level, and months to 95% recovery."""
    i0 = int(t0 / DT)
    pre = jnp.mean(traj[i0 - int(6 / DT):i0])
    post = traj[i0:]
    dip = float(1.0 - jnp.min(post) / pre)
    rec = jnp.argmax(post > 0.95 * pre) if float(jnp.min(post)) < 0.95 * float(pre) else 0
    # first index AFTER the minimum that crosses back over 95%
    imin = int(jnp.argmin(post))
    after = post[imin:]
    crossed = jnp.argmax(after > 0.95 * pre)
    rec_months = float((imin + crossed) * DT) if float(jnp.max(after)) > 0.95 * float(pre) else float("inf")
    return dip, rec_months


# ---------------- 1. neon shock replay ----------------

def burn_in(cap, x0, months=48):
    """Run undisturbed to steady state; return final stocks + fab flow."""
    _, traj = simulate(cap, x0, tidx=0, kill=0.0, tau=1.0, t0=1e9)
    steps = int(months / DT)
    # re-run capturing state: cheap trick — integrate manually
    x = x0
    for _ in range(steps):
        v = flows(x, cap)
        x = jnp.clip(x + DT * ((Post - Pre).T @ v), 0.0)
    return x, float(flows(x, cap)[FAB])

# calibrate: purification runs at ~90% utilization pre-shock (just-in-time)
x_ss, F_star = burn_in(cap0, jnp.full(NP_, 1.0))
cap_cal = cap0.at[T_IDX["Purify_Ne"]].set(F_star / 0.90)
x_ss, F_star = burn_in(cap_cal, x_ss)   # re-equilibrate under calibrated cap

# JIT reality: stocks don't accumulate — impose them. Lean = ~2wks buffer.
x_ss         = x_ss.at[P["Ne_purified"]].set(0.5 * F_star)
x_stockpiled = x_ss.at[P["Ne_purified"]].set(6.0 * F_star)   # post-2014 lesson
x_lean       = x_ss

NEON = dict(tidx=T_IDX["Purify_Ne"], kill=0.5, tau=9.0, t0=6.0)

if __name__ == "__main__":
    print(f"== 2022 neon shock replay ==")
    print(f"(steady-state fab flow {F_star:.3f}, purify utilization 90%;")
    print(f" 50% purification capacity lost, alt-supply tau=9mo)\n")
    for label, x0 in [("WITH ~6mo stockpile (history)", x_stockpiled),
                      ("WITHOUT stockpile (counterfactual)", x_lean)]:
        _, traj = simulate(cap_cal, x0, **NEON)
        dip, rec = dip_and_recovery(traj, NEON["t0"])
        print(f"   {label}:")
        print(f"      throughput dip: {100*dip:5.1f}%   recovery to 95%: {rec:.0f} months")
    print("\n   historical record: no fab stoppages; supply normalized ~12-18mo.")

    # ---------------- 2. expected-throughput gradients ----------------
    # crude prior over disruptions (prob, transition hit, severity, rebuild tau)
    SCENARIOS = [
        (0.30, T_IDX["Purify_Ne"],  0.5, 9.0),    # regional war hits inert gas
        (0.25, T_IDX["Ship_Strait"],0.8, 12.0),   # strait blockade
        (0.20, T_IDX["Fab"],        0.9, 60.0),   # Taiwan fab loss (slow rebuild)
        (0.15, T_IDX["Refine_Ga"],  0.9, 30.0),   # export ban hardens
        (0.10, T_IDX["OpticsMfg"],  0.9, 120.0),  # Zeiss knocked out (near-unrebuildable)
    ]

    def expected_throughput(cap, x0):
        tot = 0.0
        for prob, tidx, kill, tau in SCENARIOS:
            t, _ = simulate(cap, x0, tidx, kill, tau, t0=6.0)
            tot = tot + prob * t
        return tot

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

def frozen_replay():
    # uniform calibration rule: every direct fab-input source runs at 90% util
    cap = dict(cal=cap0)
    c = cap0
    x = jnp.full(NP_, 1.0)
    for _ in range(2):                      # calibrate -> re-equilibrate -> recal
        xs, F = burn_in(c, x)
        for tname in ["Purify_Ne", "WaferSupply", "Refine_Ga"]:
            c = c.at[T_IDX[tname]].set(F / 0.90)
        x = xs
    xs, F = burn_in(c, x)
    # JIT default: thin (0.5mo) buffers on intermediate stocks
    for pname in ["Ne_purified", "Wafers", "Ga_refined", "Chips", "Pkg"]:
        xs = xs.at[P[pname]].set(0.5 * F)

    EVENTS = [
        ("2011 Tohoku (wafers)",        "WaferSupply", 0.25, 4.0,  "Wafers",      2.0,
         "history: minor global impact"),
        ("1993 Sumitomo (packaging)",   "Package",     0.60, 6.0,  "Chips",       1.5,
         "history: price spike, brief pain, no catastrophe"),
        ("2019 photoresist (analog)",   "Purify_Ne",   0.90, 1.0,  "Ne_purified", 2.0,
         "history: non-event"),
    ]
    print("\n== FROZEN-PROTOCOL HOLDOUT REPLAYS ==")
    print(f"(calibrated steady-state fab flow {F:.3f}; no per-event tuning)\n")
    for name, tname, kill, tau, bplace, bmonths, hist in EVENTS:
        x0 = xs.at[P[bplace]].set(bmonths * F)
        _, traj = simulate(c, x0, T_IDX[tname], kill, tau, t0=6.0)
        dip, rec = dip_and_recovery(traj, 6.0)
        rec_s = f"{rec:.0f}mo" if rec != float("inf") else "n/a"
        print(f"   {name}")
        print(f"      model: dip {100*dip:5.1f}%, recovery {rec_s:>5}   | {hist}")

frozen_replay()
