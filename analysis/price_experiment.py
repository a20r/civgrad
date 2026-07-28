"""
analysis/price_experiment.py — the Sumitomo hysteresis, probed.

The scorecard's standing miss: Sumitomo 1993 under protocol v1 predicts a
57.6% delivery collapse that never recovers, against a historical record of
"price spike, brief pain, no catastrophe." The diagnosis on the scorecard is
"no price-mediated allocation." This experiment tests that diagnosis from
both sides — it does NOT change the model, and the xfail stays red until a
mechanism ships as a methodology PR (PLAN.md §4-5).

WHAT THE TRAP ACTUALLY IS (measured here, not narrated): after the impulse,
the adaptation law rebuilds Package capacity only until Package's FLOW
returns to its pre-crisis reference — but the outage bloats the upstream
Chips buffer, and fat input saturation substitutes for the missing capacity.
Restoration stalls with ~8% of Package capacity still gone. Downstream, the
Pkg stock stays thin, so Ship/Build_EUV/Refurb all run depressed; nothing
drains, so the runway alarm is silent; Package's flow matches reference, so
the restoration alarm is silent. A self-consistent depressed equilibrium —
collapse as attractor. The wrong-observable failure, this time inside the
control law.

ARM 1 — processing-margin restoration (the escape reality has). In 1993 the
signal that ended the pain was a price spread: packaged-chip/EMC prices
spiked while upstream chips piled up, and that margin financed repair and
rival qualification within months. The formal shadow of that spread needs no
price variable: a transition sees a fat margin when its OUTPUT stock sits
below its reference saturation while its INPUTS are available. The variant
law adds that term, gated by restore_gap exactly like the flow signal — it
can only restore destroyed capacity toward baseline, never expand past it
(the restore-vs-expand distinction from design failure five):

    out_deficit_p = clip((sat_ref_p - sat_p) / sat_ref_p / 0.10, 0, 1)
    margin_i      = max_{p in out(i)} out_deficit_p * input_avail_i
    prod_scar     = max(runway_scar, flow_scar*restore_gap, margin_i*restore_gap)

ARM 2 — the cautionary control: allocate the contested stock by gradient.
The single-scenario capacity gradient (which prices holding chips back for
the tool-building loop, because that run's catastrophe destroys tools) is
applied as an allocation rule over the three consumers of Pkg. This is what
naive "invest/allocate along the gradient" advice would do in a crisis the
gradient was not computed for.

    python3 -m analysis.price_experiment            # prints + writes PRICE_EXPERIMENT.md
    python3 -m analysis.price_experiment --smoke    # prints only (CI-sized)
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from core.continuous import (Pre, Post, P, T_IDX, NP_, NT, PLACES,
                             TRANSITIONS, cap0)
from analysis import engine
from analysis.engine import (K_SAT0, UTILIZATION0, ALPHA0, HORIZON0, OUT,
                             flows, DT, MONTHS)

ROOT = Path(__file__).resolve().parents[1]
TNAMES = [t[0] for t in TRANSITIONS]

# The three consumers of the contested stock (Pkg), for arm 2.
PKG_CONSUMERS = ["Ship_Strait", "Build_EUV", "Refurb"]


def adapt_simulate_margin(cap_base, x0, x_ref, k_sat, tidx, kill, alpha,
                          horizon, t0=6.0, months=MONTHS):
    """The frozen v1 law + the processing-margin restoration term."""
    hit = jnp.zeros(NT).at[tidx].set(1.0)
    v_ref = flows(x_ref, cap_base, k_sat)
    sat_ref = x_ref / (x_ref + k_sat)
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

        # -- the margin term (the only addition) --
        sat = x / (x + k_sat)
        out_deficit = jnp.clip((sat_ref - sat) / (sat_ref + 1e-9) / 0.10,
                               0.0, 1.0)
        inp = jnp.where(Pre > 0, sat[None, :], jnp.inf)
        lim = jnp.min(inp, axis=1)
        lim = jnp.where(jnp.isinf(lim), 1.0, lim)   # input availability
        margin = jnp.max(OUT * out_deficit[None, :], axis=1) * lim

        prod_scar = jnp.maximum(runway_scar,
                                jnp.maximum(flow_scar, margin) * restore_gap)
        c = c + DT * alpha * cap_base * prod_scar
        v = flows(x, c, k_sat)
        x = jnp.clip(x + DT * ((Post - Pre).T @ v), 0.0)
        return (x, c), (v[fab], v[ship], c)

    (_, _), (fab_t, ship_t, c_t) = jax.lax.scan(
        step, (x0, cap_base), jnp.arange(int(months / DT)))
    return fab_t, ship_t, c_t


def flows_rationed(x, cap, k_sat, weights):
    """flows() with the Pkg stock allocated across its consumers by
    `weights` (sums to 1). Consumer i sees an x_pkg share of n*w_i*x_pkg,
    so equal weights reproduce the baseline exactly."""
    n = len(PKG_CONSUMERS)
    sat = x / (x + k_sat)
    sat_row = jnp.tile(sat[None, :], (NT, 1))
    for j, tname in enumerate(PKG_CONSUMERS):
        i = T_IDX[tname]
        xa = n * weights[j] * x[P["Pkg"]]
        sat_row = sat_row.at[i, P["Pkg"]].set(xa / (xa + k_sat))
    inp = jnp.where(Pre > 0, sat_row, jnp.inf)
    lim = jnp.min(inp, axis=1)
    lim = jnp.where(jnp.isinf(lim), 1.0, lim)
    from core.continuous import Read
    rd = jnp.prod(jnp.where(Read > 0, sat[None, :], 1.0), axis=1)
    return cap * lim * rd


def adapt_simulate_rationed(cap_base, x0, x_ref, k_sat, tidx, kill, alpha,
                            horizon, weights, t0=6.0, months=MONTHS):
    """The frozen v1 law, with Pkg allocated by `weights` from the shock
    onward (a CRISIS rationing rule): before t0 the competition is the
    baseline proportional one (equal weights reproduce it exactly)."""
    hit = jnp.zeros(NT).at[tidx].set(1.0)
    w_equal = jnp.full(len(PKG_CONSUMERS), 1.0 / len(PKG_CONSUMERS))
    v_ref = flows_rationed(x_ref, cap_base, k_sat, w_equal)
    fab, ship = T_IDX["Fab"], T_IDX["Ship_Strait"]

    def step(carry, k):
        x, c = carry
        t = k * DT
        c = jnp.where(jnp.abs(t - t0) < DT / 2, c * (1.0 - kill * hit), c)
        w_t = jnp.where(t >= t0, weights, w_equal)
        v0 = flows_rationed(x, c, k_sat, w_t)
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
        v = flows_rationed(x, c, k_sat, w_t)
        x = jnp.clip(x + DT * ((Post - Pre).T @ v), 0.0)
        return (x, c), (v[fab], v[ship])

    (_, _), (fab_t, ship_t) = jax.lax.scan(
        step, (x0, cap_base), jnp.arange(int(months / DT)))
    return fab_t, ship_t


def run_event(name, law="frozen", weights=None):
    """Replay one scored v1 event under a law variant. Returns (dip%, rec)."""
    ev = engine.V1_EVENTS[name]
    c, x_jit, F = engine.calibrate_v1(cap0, K_SAT0, UTILIZATION0)
    x0 = x_jit
    for place, months in ev["buffers"].items():
        x0 = x0.at[P[place]].set(months * F)
    args = (c, x0, x_jit, K_SAT0, T_IDX[ev["transition"]], ev["kill"],
            ALPHA0, HORIZON0)
    if law == "frozen":
        fab, ship, _ = engine.adapt_simulate(*args)
    elif law == "margin":
        fab, ship, _ = adapt_simulate_margin(*args)
    elif law == "rationed":
        if weights is None:
            raise ValueError("law='rationed' needs explicit weights over "
                             f"{PKG_CONSUMERS} (see gradient_weights())")
        weights = jnp.asarray(weights, dtype=jnp.float64)
        if weights.shape != (len(PKG_CONSUMERS),) or not bool(
                jnp.all(jnp.isfinite(weights)) & (jnp.sum(weights) > 0)):
            raise ValueError(f"weights must be {len(PKG_CONSUMERS)} finite "
                             "values with a positive sum, one per "
                             f"{PKG_CONSUMERS}")
        fab, ship = adapt_simulate_rationed(*args, weights / jnp.sum(weights))
    traj = fab if ev["observe"] == "fab" else ship
    d, r = engine.dip_recovery_adapt(traj)
    return 100 * float(d), float(r)


def gradient_weights():
    """Arm 2's price list: the single-scenario capacity gradient over the
    Pkg consumers, floored at zero and normalized."""
    x0 = jnp.full(NP_, 1.0)
    thru = lambda c, x: engine.simulate_throughput(c, x, K_SAT0, 0.9)
    g = jax.grad(thru, argnums=0)(cap0, x0)
    raw = jnp.array([jnp.maximum(g[T_IDX[t]], 0.0) for t in PKG_CONSUMERS])
    return g, raw / jnp.sum(raw)


def diagnose_trap():
    """Measured anatomy of the depressed equilibrium (frozen law, Sumitomo)."""
    ev = engine.V1_EVENTS["sumitomo_1993"]
    c, x_jit, F = engine.calibrate_v1(cap0, K_SAT0, UTILIZATION0)
    x0 = x_jit.at[P["Chips"]].set(ev["buffers"]["Chips"] * F)
    fab, ship, c_t = engine.adapt_simulate(
        c, x0, x_jit, K_SAT0, T_IDX["Package"], ev["kill"], ALPHA0, HORIZON0)
    pre = float(jnp.mean(ship[int(3 / DT):int(6 / DT)]))
    end = float(jnp.mean(ship[-int(6 / DT):]))
    pkg_cap_end = float(c_t[-1, T_IDX["Package"]] / c[T_IDX["Package"]])
    return {
        "pkg_cap_end_frac": pkg_cap_end,
        "delivered_end_frac": end / pre,
        "dip_pct": 100 * float(engine.dip_recovery_adapt(ship)[0]),
    }


def build_markdown(diag, table, wtab, g_ship):
    md = []
    md.append("# civgrad — the price experiment (Sumitomo probed from both sides)\n")
    md.append("<!-- AUTO-GENERATED by `python3 -m analysis.price_experiment` — do not edit by hand. -->\n")
    md.append("The scorecard's standing miss (SCORECARD.md): Sumitomo 1993 under the")
    md.append("frozen adaptive protocol predicts a ~58% delivery collapse that never")
    md.append("recovers; history records a price spike and brief pain. The diagnosis on")
    md.append("the scorecard is *no price-mediated allocation*. This experiment tests")
    md.append("that diagnosis. **Nothing here changes the model**: the law variants live")
    md.append("in `analysis/`, the frozen protocol and its xfail are untouched, and any")
    md.append("promotion of a mechanism into `core/` is a methodology PR")
    md.append("(PLAN.md §4-5; the design discussion lives in issue #2, closed with the")
    md.append("solo-project de-scope but still the source material for this work).\n")

    md.append("## The trap, measured\n")
    md.append("The narrated version of the hysteresis (\"the tool fleet wears down")
    md.append("unreplaced\") is not what the trace shows. The measured anatomy: the")
    md.append("impulse bloats the upstream Chips buffer, and **fat input saturation")
    md.append("substitutes for destroyed capacity in the restoration signal** — Package's")
    md.append("flow returns to reference while ~8% of its capacity is still missing")
    md.append(f"(capacity settles at {diag['pkg_cap_end_frac']:.2f} of baseline). The Pkg")
    md.append("stock downstream stays thin, every Pkg consumer runs depressed, and")
    md.append(f"delivered flow settles at {diag['delivered_end_frac']:.2f} of its")
    md.append("pre-shock mean — below the 95% recovery threshold, forever. Both alarms")
    md.append("are quantitatively silent at the equilibrium: nothing drains (runway")
    md.append("silent) and observed flow matches reference (restoration silent). The")
    md.append("wrong-observable failure from the design history, resurfacing *inside the")
    md.append("control law*. Collapse as attractor, measured.\n")

    md.append("## Arm 1 — processing-margin restoration\n")
    md.append("What ended the 1993 pain in reality was a price spread: packaging output")
    md.append("prices spiked while upstream chips piled up, and that margin financed")
    md.append("repair and rival qualification within months. The formal shadow of that")
    md.append("spread needs no price variable: a transition sees a fat **processing")
    md.append("margin** when its output stock sits below its reference saturation while")
    md.append("its inputs are available. The variant law adds exactly that term, gated")
    md.append("by `restore_gap` like the flow signal — it can only restore destroyed")
    md.append("capacity toward baseline, never expand past it (the restore-vs-expand")
    md.append("distinction from design failure five).\n")
    md.append("| event (v1 protocol) | frozen law | + margin restoration |")
    md.append("|---|---:|---:|")
    for name, (d0, r0), (d1, r1) in table:
        f0 = f"dip {d0:.1f}%, rec {'never' if r0 == float('inf') else f'{r0:.1f}mo'}"
        f1 = f"dip {d1:.1f}%, rec {'never' if r1 == float('inf') else f'{r1:.1f}mo'}"
        md.append(f"| {name} | {f0} | {f1} |")
    md.append("")
    sumi = next(row for row in table if row[0] == "sumitomo_1993")
    d1, r1 = sumi[2]
    md.append(f"The attractor dissolves: delivered flow re-crosses 95% about")
    md.append(f"**{r1:.0f} months** after the shock (vs never under the frozen law),")
    md.append("with the three passing replays essentially untouched. The diagnosis was")
    md.append("right: what the model is missing at Sumitomo is a reallocation signal,")
    md.append("and a margin term one step removed from an actual price is enough to")
    md.append("supply the escape. This is evidence FOR the RFC's mechanism — not a fix.")
    md.append("The scorecard row stays red until this ships properly (full re-fit,")
    md.append("re-freeze, methodology version bump), because a law changed after seeing")
    md.append("the holdout is, by definition, no longer validated against it.\n")

    md.append("## Arm 2 — the cautionary control: gradient-as-price\n")
    md.append("The project's thesis object is the investment gradient. The obvious")
    md.append("misuse is to treat it as a crisis *allocation* rule. Arm 2 does exactly")
    md.append("that: from the shock onward, the contested Pkg stock is allocated across")
    md.append("its three consumers in proportion to the single-scenario capacity gradient")
    for line in wtab:
        md.append(line)
    md.append("")
    md.append(f"Because that gradient was computed on a catastrophe that destroys tools,")
    md.append(f"it prices shipping *negative* ({g_ship:.2f}) and hands the entire stock")
    md.append("to the equipment loop. Applied during a packaging outage it drives")
    md.append("delivered flow to zero and keeps it there — a collapse the frozen law")
    md.append("never produces. The moral, stated plainly: **the gradient is a peacetime")
    md.append("investment ranking under a declared prior, not a wartime allocation")
    md.append("rule.** A gradient computed on one catastrophe is the wrong price list")
    md.append("for another. Real prices re-solve the allocation problem every day with")
    md.append("current information; a frozen derivative does not.\n")

    md.append("**Scope.** Both arms inherit every caveat on the frozen model")
    md.append("(confidence-C parameters, SENSITIVITY.md, the pessimism bias). The")
    md.append("margin law was designed after staring at this specific failure, so its")
    md.append("clean Sumitomo recovery is an in-sample result — the honest claim is")
    md.append("\"the missing-mechanism diagnosis is confirmed,\" not \"the model now")
    md.append("gets 1993 right.\" Promotion path: PLAN.md §4 methodology PR with the")
    md.append("full suite re-run from scratch and predictions registered first.")
    md.append("")
    return "\n".join(md)


def main():
    smoke = "--smoke" in sys.argv
    print("selfcheck against frozen baseline...", flush=True)
    engine.selfcheck()

    diag = diagnose_trap()
    print(f"trap: Package capacity settles at {diag['pkg_cap_end_frac']:.3f} of base;"
          f" delivered at {diag['delivered_end_frac']:.3f} of pre-shock", flush=True)

    table = []
    for name in engine.V1_EVENTS:
        base = run_event(name, "frozen")
        marg = run_event(name, "margin")
        table.append((name, base, marg))
        print(f"   {name:<18} frozen: dip {base[0]:5.1f}% rec {base[1]:5.1f} | "
              f"margin: dip {marg[0]:5.1f}% rec {marg[1]:5.1f}", flush=True)

    g, w = gradient_weights()
    g_ship = float(g[T_IDX["Ship_Strait"]])
    wtab = ["", "| consumer of Pkg | gradient | weight |", "|---|---:|---:|"]
    for j, tname in enumerate(PKG_CONSUMERS):
        wtab.append(f"| {tname} | {float(g[T_IDX[tname]]):.2f} | {float(w[j]):.3f} |")
    d2, r2 = run_event("sumitomo_1993", "rationed", weights=w)
    print(f"   sumitomo, gradient-rationed: dip {d2:5.1f}% rec {r2:5.1f}", flush=True)
    wtab.append("")
    wtab.append(f"Result: dip {d2:.1f}%, recovery "
                f"{'never' if r2 == float('inf') else f'{r2:.1f} mo'} — worse than the")
    wtab.append("frozen law's miss, in the direction of total delivery collapse.")

    md = build_markdown(diag, table, wtab, g_ship)
    if smoke:
        print("\n--- smoke run: not writing PRICE_EXPERIMENT.md ---")
        return
    (ROOT / "PRICE_EXPERIMENT.md").write_text(md)
    print(f"\nwrote {ROOT / 'PRICE_EXPERIMENT.md'}")


if __name__ == "__main__":
    main()
