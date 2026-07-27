"""
analysis/engine.py — parameterized twin of the frozen engine.

The frozen model (core/, validation/) bakes its confidence-C parameter values
in at import time; a sensitivity sweep needs the same dynamics with the
parameters exposed as arguments so JAX can vmap over perturbation draws.
This module re-states the frozen equations with (cap, k_sat, alpha, horizon,
utilization) as inputs — the SAME operations in the SAME order, so the twin
reproduces the frozen baseline exactly at the unperturbed point.

That equivalence is not assumed: `selfcheck()` asserts the twin against the
frozen modules on every headline number (throughput, gradients, all four v1
replays) and is run by tests/test_analysis_engine.py and by each analysis
entry point before any sweep. If the twin drifts from core/, everything here
fails loudly rather than auditing the wrong model.

Nothing in core/ or validation/ imports this module.
"""

import jax
import jax.numpy as jnp

from core.continuous import (Pre, Post, Read, P, T_IDX, NP_, NT, PLACES,
                             TRANSITIONS, cap0)
from core.continuous import K_SAT as K_SAT0, UTILIZATION as UTILIZATION0
from core.map_loader import frozen_globals

_FG = frozen_globals()
ALPHA0 = _FG["ALPHA"]
HORIZON0 = _FG["HORIZON"]

OUT = (Post > 0).astype(jnp.float64)

# The engine's frozen integrator constants (not confidence-C map parameters):
# core/continuous.py: DT=0.05, STEPS=4000, T_DISRUPT=1000 for the throughput
# objective; monthly-time integrators use dt=0.1, 72-month replay horizon.
THRU_DT, THRU_STEPS, THRU_T_DISRUPT = 0.05, 4000, 1000
DT, MONTHS = 0.1, 72


def flows(x, cap, k_sat):
    """Twin of core.continuous.flows with k_sat as an argument."""
    sat = x / (x + k_sat)
    inp = jnp.where(Pre > 0, sat[None, :], jnp.inf)
    lim = jnp.min(inp, axis=1)
    lim = jnp.where(jnp.isinf(lim), 1.0, lim)
    rd = jnp.prod(jnp.where(Read > 0, sat[None, :], 1.0), axis=1)
    return cap * lim * rd


def simulate_throughput(cap, x0, k_sat, kill_frac):
    """Twin of core.continuous.simulate: total fab throughput with kill_frac
    of EUV_tools destroyed at step T_DISRUPT."""
    kill_frac = jnp.asarray(kill_frac)
    fab = T_IDX["Fab"]

    def step(carry, t):
        x, tot = carry
        x = jax.lax.cond(
            t == THRU_T_DISRUPT,
            lambda x: x.at[P["EUV_tools"]].mul(1.0 - kill_frac),
            lambda x: x, x)
        v = flows(x, cap, k_sat)
        x = jnp.clip(x + THRU_DT * ((Post - Pre).T @ v), 0.0)
        return (x, tot + THRU_DT * v[fab]), None

    (xf, tot), _ = jax.lax.scan(step, (x0, 0.0), jnp.arange(THRU_STEPS))
    return tot


def burn_in(cap, x0, k_sat, months=48, dt=0.1):
    """Twin of core.continuous.burn_in (python loop restated as a scan; the
    per-step update is identical). Returns (final stocks, fab flow)."""
    def step(x, _):
        v = flows(x, cap, k_sat)
        x = jnp.clip(x + dt * ((Post - Pre).T @ v), 0.0)
        return x, None

    x, _ = jax.lax.scan(step, x0, None, length=int(months / dt))
    return x, flows(x, cap, k_sat)[T_IDX["Fab"]]


def simulate_recovery(cap_base, x0, k_sat, tidx, kill, tau, t0,
                      months=MONTHS, dt=DT):
    """Twin of core.continuous.simulate_recovery (imposed-tau protocol).
    Returns the fab flow trajectory."""
    fab = T_IDX["Fab"]
    hit = jnp.zeros(NT).at[tidx].set(1.0)

    def step(carry, k):
        x, tot = carry
        t = k * dt
        since = t - t0
        recov = 1.0 - kill * jnp.exp(-jnp.maximum(since, 0.0) / tau)
        mult = jnp.where(since >= 0.0, recov, 1.0)
        cap = cap_base * (1.0 + hit * (mult - 1.0))
        v = flows(x, cap, k_sat)
        x = jnp.clip(x + dt * ((Post - Pre).T @ v), 0.0)
        return (x, tot + dt * v[fab]), v[fab]

    (_, _), traj = jax.lax.scan(step, (x0, 0.0), jnp.arange(int(months / dt)))
    return traj


def adapt_simulate(cap_base, x0, x_ref, k_sat, tidx, kill, alpha, horizon,
                   t0=6.0, months=MONTHS):
    """Twin of core.adaptation.simulate (protocol v1). Returns (fab, ship)
    flow trajectories and the capacity trajectory."""
    hit = jnp.zeros(NT).at[tidx].set(1.0)
    v_ref = flows(x_ref, cap_base, k_sat)
    fab, ship = T_IDX["Fab"], T_IDX["Ship_Strait"]

    def step(carry, k):
        x, c = carry
        t = k * DT
        c = jnp.where(jnp.abs(t - t0) < DT / 2, c * (1.0 - kill * hit), c)
        v0 = flows(x, c, k_sat)
        inflow = Post.T @ v0
        outflow = Pre.T @ v0
        net_drain = jnp.maximum(outflow - inflow, 1e-9)
        runway = x / net_drain
        scar = jnp.clip(1.0 - runway / horizon, 0.0, 1.0)
        scar = jnp.where(outflow - inflow > 1e-6, scar, 0.0)
        runway_scar = jnp.max(OUT * scar[None, :], axis=1)
        flow_scar = jnp.clip((v_ref - v0) / (v_ref + 1e-9) / 0.10, 0.0, 1.0)
        restore_gap = jnp.clip((cap_base - c) / (1e-6 * cap_base + 1e-9), 0.0, 1.0)
        prod_scar = jnp.maximum(runway_scar, flow_scar * restore_gap)
        c = c + DT * alpha * cap_base * prod_scar
        v = flows(x, c, k_sat)
        x = jnp.clip(x + DT * ((Post - Pre).T @ v), 0.0)
        return (x, c), (v[fab], v[ship], c)

    (_, _), (fab_t, ship_t, c_t) = jax.lax.scan(
        step, (x0, cap_base), jnp.arange(int(months / DT)))
    return fab_t, ship_t, c_t


# ---------------------------------------------------------- calibrations ----

FAB_INPUT_SOURCES = ("Purify_Ne", "WaferSupply", "Refine_Ga")
JIT_PLACES = ("Ne_purified", "Wafers", "Ga_refined", "Chips", "Pkg")
SRC_MASK = jnp.zeros(NT).at[jnp.array([T_IDX[t] for t in FAB_INPUT_SOURCES])].set(1.0)
JIT_MASK = jnp.zeros(NP_).at[jnp.array([P[p] for p in JIT_PLACES])].set(1.0)


def calibrate_v1(cap_raw, k_sat, utilization):
    """Twin of the frozen calibration in core/adaptation.py: two rounds of
    (burn in; set fab-input sources to F/utilization), a final burn-in, then
    JIT stocks at 0.5*F. Returns (c, X_JIT, F). X_REF == X_JIT."""
    c, x = cap_raw, jnp.full(NP_, 1.0)
    for _ in range(2):
        xs, F = burn_in(c, x, k_sat)
        c = c * (1.0 - SRC_MASK) + SRC_MASK * (F / utilization)
        x = xs
    x_ref, F = burn_in(c, x, k_sat)
    x_jit = x_ref * (1.0 - JIT_MASK) + JIT_MASK * (0.5 * F)
    return c, x_jit, F


def calibrate_neon_v0(cap_raw, k_sat, utilization, lean_months, stock_months):
    """Twin of validation.runner.calibrate_neon_v0. Returns
    (cap_cal, x_stockpiled, x_lean, F_star)."""
    x_ss, F_star = burn_in(cap_raw, jnp.full(NP_, 1.0), k_sat)
    cap_cal = cap_raw.at[T_IDX["Purify_Ne"]].set(F_star / utilization)
    x_ss, F_star = burn_in(cap_cal, x_ss, k_sat)
    x_stockpiled = x_ss.at[P["Ne_purified"]].set(stock_months * F_star)
    x_lean = x_ss.at[P["Ne_purified"]].set(lean_months * F_star)
    return cap_cal, x_stockpiled, x_lean, F_star


# ------------------------------------------------------------- objectives ----

# The disruption prior, restated as arrays (twin of core.gradients.SCENARIOS).
SCEN_PROB = jnp.array([0.30, 0.25, 0.20, 0.15, 0.10])
SCEN_TIDX = jnp.array([T_IDX["Purify_Ne"], T_IDX["Ship_Strait"], T_IDX["Fab"],
                       T_IDX["Refine_Ga"], T_IDX["OpticsMfg"]])
SCEN_KILL = jnp.array([0.5, 0.8, 0.9, 0.9, 0.9])
SCEN_TAU = jnp.array([9.0, 12.0, 60.0, 30.0, 120.0])


def scenario_throughputs(cap, x0, k_sat, prob=None, tidx=None, kill=None,
                         tau=None, t0=6.0):
    """Total fab throughput under each scenario of the disruption prior."""
    tidx = SCEN_TIDX if tidx is None else tidx
    kill = SCEN_KILL if kill is None else kill
    tau = SCEN_TAU if tau is None else tau
    def one(ti, ki,ta):
        traj = simulate_recovery(cap, x0, k_sat, ti, ki, ta, t0=t0)
        return jnp.sum(traj) * DT
    return jax.vmap(one)(tidx, kill, tau)


def expected_throughput(cap, x0, k_sat, prob=None, tidx=None, kill=None,
                        tau=None, t0=6.0):
    """Twin of core.gradients.expected_throughput (same integrals, summed with
    the prior's probabilities)."""
    prob = SCEN_PROB if prob is None else prob
    return jnp.sum(prob * scenario_throughputs(cap, x0, k_sat, prob, tidx,
                                               kill, tau, t0=t0))


# -------------------------------------------------------------- metrics ----

def dip_recovery_adapt(traj, t0=6.0):
    """Pure-JAX twin of core.adaptation.dip_recovery (3-month pre window;
    recovery = time from shock to first re-cross of 95% after the minimum;
    jnp.inf when it never re-crosses)."""
    i0 = int(t0 / DT)
    pre = jnp.mean(traj[i0 - int(3 / DT):i0])
    post = traj[i0:]
    imin = jnp.argmin(post)
    dip = 1.0 - post[imin] / pre
    n = post.shape[0]
    idx = jnp.arange(n)
    above = (post > 0.95 * pre) & (idx >= imin)
    crossed = jnp.argmax(above)  # 0 if never
    rec = jnp.where(jnp.any(above), crossed * DT, jnp.inf)
    return dip, rec


def dip_recovery_v0(traj, t0):
    """Pure-JAX twin of validation.runner.dip_and_recovery (6-month pre
    window; same recovery semantics)."""
    i0 = int(t0 / DT)
    pre = jnp.mean(traj[i0 - int(6 / DT):i0])
    post = traj[i0:]
    dip = 1.0 - jnp.min(post) / pre
    imin = jnp.argmin(post)
    n = post.shape[0]
    idx = jnp.arange(n)
    above = (post > 0.95 * pre) & (idx >= imin)
    crossed = jnp.argmax(above)
    rec = jnp.where(jnp.any(above), (crossed) * DT, jnp.inf)
    # runner counts recovery from the start of `post`? No: (imin + crossed_rel).
    # Here `above` is absolute-indexed, so crossed already includes imin.
    return dip, rec


# ------------------------------------------------------------- replays ----

def v1_replay(cap_raw, k_sat, alpha, horizon, utilization,
              transition, kill, buffers, t0=6.0, observe="fab"):
    """Twin of validation.runner.adaptive_replay: frozen calibration, buffers
    from the event yaml, impulse kill, emergent recovery. Returns (dip, rec)."""
    c, x_jit, F = calibrate_v1(cap_raw, k_sat, utilization)
    x0 = x_jit
    for place, months in buffers.items():
        x0 = x0.at[P[place]].set(months * F)
    fab, ship, _ = adapt_simulate(c, x0, x_jit, k_sat, T_IDX[transition],
                                  kill, alpha, horizon, t0=t0)
    return dip_recovery_adapt(fab if observe == "fab" else ship, t0)


# The four scored v1 events, facts restated from validation/events/*.yaml
# (documented history, never tuned — see the yamls for provenance).
V1_EVENTS = {
    "neon_2022": dict(transition="Purify_Ne", kill=0.5,
                      buffers={"Ne_purified": 6.0}, observe="fab"),
    "tohoku_2011": dict(transition="WaferSupply", kill=0.25,
                        buffers={"Wafers": 2.0}, observe="fab"),
    "sumitomo_1993": dict(transition="Package", kill=0.60,
                          buffers={"Chips": 1.5}, observe="ship"),
    "photoresist_2019": dict(transition="Purify_Ne", kill=0.15,
                             buffers={"Ne_purified": 2.0}, observe="fab"),
}

# Baseline v1 scorecard values (dip %, recovery months) for the selfcheck.
V1_BASELINE = {
    "neon_2022": (-0.7, 0.0),
    "tohoku_2011": (-0.6, 0.0),
    "sumitomo_1993": (57.6, float("inf")),
    "photoresist_2019": (-0.7, 0.0),
}


# ------------------------------------------------------------ selfcheck ----

def selfcheck(verbose=False):
    """Assert the twin reproduces the frozen baseline at the null point.
    Raises AssertionError on any drift; returns a dict of checked values."""
    import core.continuous as cc
    import core.adaptation as ad

    x0 = jnp.full(NP_, 1.0)
    checked = {}

    # throughput + gradients (core/gradients.py context: raw cap0, x0 = 1)
    base = simulate_throughput(cap0, x0, K_SAT0, 0.0)
    hit = simulate_throughput(cap0, x0, K_SAT0, 0.9)
    base_c, _ = cc.simulate(cap0, x0, 0.0)
    hit_c, _ = cc.simulate(cap0, x0, 0.9)
    assert abs(float(base) - float(base_c)) < 1e-9, (base, base_c)
    assert abs(float(hit) - float(hit_c)) < 1e-9, (hit, hit_c)
    checked["throughput"] = (float(base), float(hit))

    thru = lambda c, x: simulate_throughput(c, x, K_SAT0, 0.9)
    g_cap = jax.grad(thru, argnums=0)(cap0, x0)
    g_stk = jax.grad(thru, argnums=1)(cap0, x0)
    # spot-anchor the headline numbers to validation/baseline_outputs.txt
    assert abs(float(g_cap[T_IDX["Ship_Strait"]]) - (-2.678)) < 5e-3
    assert abs(float(g_stk[P["EUV_worn"]]) - 0.848) < 5e-3
    assert abs(float(g_stk[P["EUV_tools"]]) - 0.695) < 5e-3
    checked["g_cap"], checked["g_stk"] = g_cap, g_stk

    # burn-in twin vs the frozen python-loop version
    xs_t, F_t = burn_in(cap0, x0, K_SAT0)
    xs_c, F_c = cc.burn_in(cap0, x0)
    assert float(jnp.max(jnp.abs(xs_t - xs_c))) < 1e-9
    assert abs(float(F_t) - F_c) < 1e-9

    # v1 calibration twin vs core/adaptation.py module state
    c_t, x_jit_t, F1 = calibrate_v1(cap0, K_SAT0, UTILIZATION0)
    assert float(jnp.max(jnp.abs(c_t - ad.c))) < 1e-9
    assert float(jnp.max(jnp.abs(x_jit_t - ad.X_JIT))) < 1e-9
    assert abs(float(F1) - ad.F) < 1e-9

    # all four v1 replays against the frozen scorecard numbers
    for name, ev in V1_EVENTS.items():
        d, r = v1_replay(cap0, K_SAT0, ALPHA0, HORIZON0, UTILIZATION0, **ev)
        d_pct, r_f = 100 * float(d), float(r)
        want_d, want_r = V1_BASELINE[name]
        assert abs(d_pct - want_d) < 0.05, (name, d_pct, want_d)
        if want_r == float("inf"):
            assert r_f == float("inf"), (name, r_f)
        else:
            assert abs(r_f - want_r) < 0.05, (name, r_f, want_r)
        checked[name] = (d_pct, r_f)
        if verbose:
            print(f"   selfcheck {name}: dip {d_pct:5.1f}%, rec {r_f:4.1f}mo  ok")

    # expected-throughput twin vs core.gradients in the runner's context
    from core.gradients import expected_throughput as exp_c
    cap_cal, x_stock, _, _ = calibrate_neon_v0(cap0, K_SAT0, UTILIZATION0,
                                               lean_months=0.5, stock_months=6.0)
    e_t = expected_throughput(cap_cal, x_stock, K_SAT0)
    e_c = exp_c(cap_cal, x_stock)
    assert abs(float(e_t) - float(e_c)) < 1e-6, (float(e_t), float(e_c))
    checked["expected"] = float(e_t)

    return checked


if __name__ == "__main__":
    selfcheck(verbose=True)
    print("analysis/engine.py selfcheck: twin reproduces the frozen baseline.")
