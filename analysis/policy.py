"""
analysis/policy.py — the marginal dollar, cashed out.

Synthesizes the frozen baseline, the sensitivity audit
(analysis/sensitivity.py -> analysis/out/sensitivity.json), and the replay
suite into the project's actionable conclusion: where a marginal unit of
resilience investment goes, with every claim carrying its measured
robustness. Three computations feed it:

  1. ROBUST MARGINAL-VALUE TABLE — both objectives' gradient vectors, each
     entry annotated with its sign stability and #1 frequency under
     parameter fog (from the sweep).

  2. MEAN vs TAIL OBJECTIVE — the frozen objective is the prior MEAN;
     supply-chain policy usually cares about tails. CVaR(30%): the
     probability-weighted expectation of the worst scenarios holding the
     last 30% of prior mass (scenarios sorted by disrupted throughput).
     Reported as the re-ranked gradient, answering METHODOLOGY.md known
     limit #4 with a measurement instead of a promise.

  3. BUFFER RULE TABLE — the mechanism that decides nearly every replay,
     made explicit per event: a shock bites when buffer-months of cover are
     smaller than the integrated capacity deficit while the ramp completes
     (v0: kill * tau analytically; v1: measured from the emergent capacity
     trajectory).

    python3 -m analysis.policy            # prints + writes POLICY.md
    python3 -m analysis.policy --smoke    # prints only (CI-sized)

Requires analysis/out/sensitivity.json (run analysis.sensitivity first);
the committed POLICY.md is regenerated whenever either changes.
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from core.continuous import P, T_IDX, NP_, PLACES, TRANSITIONS, cap0
from analysis import engine
from analysis.engine import (K_SAT0, UTILIZATION0, ALPHA0, HORIZON0,
                             SCEN_PROB, SCEN_TIDX, SCEN_KILL, SCEN_TAU, DT)

ROOT = Path(__file__).resolve().parents[1]
TNAMES = [t[0] for t in TRANSITIONS]
SCEN_NAMES = ["neon-style gas cut", "strait blockade", "Taiwan fab loss",
              "gallium ban hardens", "Zeiss knocked out"]
CVAR_ALPHA = 0.30

# Events for the buffer-rule table: (name, buffer place+months, v0 kill, tau).
BUFFER_EVENTS = [
    ("neon_2022 (historical)", "Ne_purified", 6.0, "Purify_Ne", 0.5, 9.0),
    ("neon_2022 (lean counterfactual)", "Ne_purified", 0.5, "Purify_Ne", 0.5, 9.0),
    ("tohoku_2011", "Wafers", 2.0, "WaferSupply", 0.25, 4.0),
    ("sumitomo_1993", "Chips", 1.5, "Package", 0.60, 6.0),
    ("photoresist_2019 (realized)", "Ne_purified", 2.0, "Purify_Ne", 0.15, 1.0),
    ("photoresist_2019 (feared)", "Ne_purified", 2.0, "Purify_Ne", 0.90, 1.0),
]


def load_sensitivity():
    p = ROOT / "analysis" / "out" / "sensitivity.json"
    if not p.exists():
        sys.exit("analysis/out/sensitivity.json missing — run "
                 "`python3 -m analysis.sensitivity` first")
    return json.loads(p.read_text())


# ---------------------------------------------------- 1. robust table ----

def gradient_vectors():
    x0 = jnp.full(NP_, 1.0)
    thru = lambda c, x: engine.simulate_throughput(c, x, K_SAT0, 0.9)
    g1c = jax.grad(thru, argnums=0)(cap0, x0)
    g1s = jax.grad(thru, argnums=1)(cap0, x0)
    cap_cal, x_stock, _, _ = engine.calibrate_neon_v0(
        cap0, K_SAT0, UTILIZATION0, 0.5, 6.0)
    f = lambda c, x: engine.expected_throughput(c, x, K_SAT0)
    gec = jax.grad(f, argnums=0)(cap_cal, x_stock)
    ges = jax.grad(f, argnums=1)(cap_cal, x_stock)
    return g1c, g1s, gec, ges, cap_cal, x_stock


def sign_label(node):
    """Robustness annotation from the STRICT directional fractions (draws
    with exactly-zero gradients support neither sign and count for neither —
    pos+neg need not reach 100)."""
    pos, neg = node["pos_pct"], node["neg_pct"]
    conf = max(pos, neg)
    sign = "+" if pos >= neg else "-"
    if conf >= 85.0:
        return f"{sign} robust ({conf:.0f}%)"
    if conf >= 65.0:
        return f"{sign} leaning ({conf:.0f}%)"
    return f"~0 unstable ({conf:.0f}%)"


# ------------------------------------------------------ 2. mean vs tail ----

def cvar_gradients(cap_cal, x_stock):
    """CVaR(30%) over the 5-scenario prior: sort scenarios by disrupted
    throughput, keep the worst tail holding CVAR_ALPHA of probability mass
    (fractional inclusion at the boundary), renormalize, differentiate."""
    T = engine.scenario_throughputs(cap_cal, x_stock, K_SAT0)
    order = jnp.argsort(T)  # worst first
    p_sorted = SCEN_PROB[order]
    cum = jnp.cumsum(p_sorted)
    w_sorted = jnp.clip(CVAR_ALPHA - (cum - p_sorted), 0.0, p_sorted) / CVAR_ALPHA
    w = jnp.zeros_like(SCEN_PROB).at[order].set(w_sorted)

    def cvar(c, x):
        Ts = engine.scenario_throughputs(c, x, K_SAT0)
        return jnp.sum(jax.lax.stop_gradient(w) * Ts)

    gc = jax.grad(cvar, argnums=0)(cap_cal, x_stock)
    gs = jax.grad(cvar, argnums=1)(cap_cal, x_stock)
    return T, w, gc, gs


# ------------------------------------------------------ 3. buffer rule ----

def buffer_table():
    """Per event: buffer-months of cover vs the integrated capacity deficit
    (month-equivalents of lost supply while the ramp completes)."""
    rows = []
    c, x_jit, F = engine.calibrate_v1(cap0, K_SAT0, UTILIZATION0)
    for name, bplace, bmonths, tname, kill, tau in BUFFER_EVENTS:
        deficit_v0 = kill * tau  # integral of kill*exp(-t/tau)
        # v1: measured from the emergent capacity trajectory
        x0 = x_jit.at[P[bplace]].set(bmonths * F)
        fab, ship, c_t = engine.adapt_simulate(
            c, x0, x_jit, K_SAT0, T_IDX[tname], kill, ALPHA0, HORIZON0)
        i0 = int(6.0 / DT)
        chit = c_t[i0:, T_IDX[tname]] / c[T_IDX[tname]]
        deficit_v1 = float(jnp.sum(jnp.clip(1.0 - chit, 0.0)) * DT)
        obs = "ship" if tname == "Package" else "fab"
        d, r = engine.dip_recovery_adapt(ship if obs == "ship" else fab)
        rows.append((name, bmonths, deficit_v0, deficit_v1,
                     100 * float(d), float(r)))
    return rows


# -------------------------------------------------------------- report ----

def fmt_rec(r):
    return "never" if r == float("inf") else f"{r:.1f}mo"


def build_markdown(sens, g1c, g1s, gec, ges, T, w, gcv, gsv, btab):
    a, b = sens["single"], sens["expected"]
    md = []
    md.append("# civgrad — the marginal dollar, cashed out\n")
    md.append("<!-- AUTO-GENERATED by `python3 -m analysis.policy` — do not edit by hand. -->\n")
    md.append("The project's thesis object is d(resilience)/d($). This document is that")
    md.append("vector turned into conclusions a policymaker or investor could act on —")
    md.append("with every claim carrying its measured robustness from the sensitivity")
    md.append("audit (SENSITIVITY.md), because an actionable conclusion that hides its")
    md.append("error bars is a liability, not a contribution.\n")
    md.append("**Scope, before anything else.** The map is one semiconductor slice with")
    md.append("confidence-C parameters and declared fog at every frontier; the model has")
    md.append("no price-mediated allocation and therefore overstates how deep shocks")
    md.append("bite and how long they last (SCORECARD.md, the Sumitomo row). Every")
    md.append("conclusion below is conditional on the map and stated with the fog")
    md.append("audit's numbers attached. \"Robust\" here means: survives x0.5-x2")
    md.append("parameter ignorance and disruption-prior fog in the measured fractions.\n")

    md.append("## The five conclusions\n")
    md.append("**1. Buffers are the cheapest resilience the model can find, and they")
    md.append("have a sizing rule.** A shock bites only when buffer-months of cover are")
    md.append("smaller than the integrated capacity deficit while supply rebuilds (§3")
    md.append("below; it decides every replay in the suite). Neon 2022 is the exhibit:")
    md.append("the industry's ~6 months of stockpile — bought after the 2014 Crimea")
    md.append("price spike — turned a would-be ~24% fab dip into nothing. *Action: at")
    md.append("single-source chokepoints, hold (or mandate disclosure of) months-of-")
    md.append("cover sized to `capacity-lost x expected-ramp-months` for the outages")
    md.append("you consider plausible. Materials inventory at chokepoints is insurance")
    md.append("priced well below the capacity it protects.*\n")
    md.append("**2. The marginal capacity dollar belongs in the production/equipment")
    md.append(f"complex — never in downstream logistics.** In {a['top_cap_upstream_pct']:.0f}%")
    md.append(f"(single-scenario) / {b['top_cap_upstream_pct']:.0f}% (prior-averaged) of")
    md.append("parameter-fog draws the top-gradient capacity is fab, equipment-making,")
    md.append("or a fab-input source; the equipment-making loop (Build_EUV) is the")
    md.append("modal leader. WHICH node inside the complex leads is fog-conditional —")
    md.append("the audit does not license picking a single winner — but the partition")
    md.append("is about as robust as anything this model produces. *Action: capacity")
    md.append("subsidies and industrial-policy dollars go upstream (equipment,")
    md.append("tooling, input processing), not to shipping, packaging, or")
    md.append("consumption-side capacity.*\n")
    md.append("**3. Post-shock, more shipping capacity is negative-value; allocation")
    md.append("authority is what matters — and static priority lists are dangerous.**")
    md.append(f"The shipping gradient is negative in {a['neg_caps']['Ship_Strait']['stable_pct']:.0f}%/"
              f"{b['neg_caps']['Ship_Strait']['stable_pct']:.0f}% of parameter-fog draws")
    md.append("and 100% of prior-fog draws: after a capacity-destroying shock, exports")
    md.append("drain exactly the intermediate the rebuild loop needs — the model")
    md.append("derives wartime rationing from topology. But the cautionary arm of the")
    md.append("price experiment (PRICE_EXPERIMENT.md) shows that rationing by a FROZEN")
    md.append("priority list (the gradient itself) during the *wrong* crisis drives")
    md.append("deliveries to zero. *Action: crisis powers should take the form of")
    md.append("allocation authority exercised on current information (DPA-style")
    md.append("priority ratings recomputed per event), not pre-committed priority")
    md.append("lists and not logistics-capacity subsidies.*\n")
    md.append("**4. Refurbishable equipment is resilience capital comparable to")
    md.append("pristine spares — stop shredding it.** Worn tools are the top stockpile")
    md.append("in the single-scenario baseline (0.85 vs 0.70 for working spares); the")
    md.append("prior-averaged objective flips the pair (0.43 vs 0.51), and under")
    md.append(f"parameter fog worn-on-top survives in only {a['worn_on_top_pct']:.0f}%")
    md.append("of draws — so the audited claim is *comparable value*, not *highest")
    md.append("value*. Comparable is still remarkable for scrap. *Action: a")
    md.append("decommissioned-tool registry / warm boneyard and refurbishment capacity")
    md.append("are cheap resilience buys; an industry that routinely scraps old fab")
    md.append("equipment is discarding a stockpile the model prices near working")
    md.append("spares.*\n")
    tailtop = TNAMES[int(jnp.argmax(gcv))]
    tailstk = PLACES[int(jnp.argmax(gsv))]
    worst = [SCEN_NAMES[int(i)] for i in jnp.argsort(T)][:2]
    md.append("**5. If you care about tails, buy fab resilience first — and note")
    md.append("which catastrophe does NOT make the tail.** The frozen objective is")
    md.append(f"the prior MEAN; a CVaR({int(100*CVAR_ALPHA)}%) tail objective (worst")
    md.append("scenarios holding the last 30% of prior mass) is dominated by the")
    md.append(f"**{worst[0]}** and the **{worst[1]}**, and its top-gradient capacity is")
    md.append(f"**{tailtop}** with **{tailstk}** the top stockpile (§2 below). The")
    md.append("surprise: the Zeiss-knockout scenario does *not* make the 72-month")
    md.append("tail — the refurbishment loop consumes worn tools and chips but no new")
    md.append("optics, so the boneyard carries the system through the replay horizon.")
    md.append("That is the boneyard result wearing its policy clothes, and it comes")
    md.append("with an honest caveat: a 120-month rebuild mostly bites *beyond* a")
    md.append("72-month horizon, so this is the six-year tail, not the forever tail.")
    md.append("*Action: tail-risk planners put the marginal dollar on fab capacity and")
    md.append("geographic redundancy plus tool stockpiles; pricing the optics monopoly")
    md.append("honestly requires extending the model's horizon — a cheap, named next")
    md.append("measurement.*\n")

    md.append("## 1. The robust marginal-value table\n")
    md.append("Gradient of disrupted throughput per unit of capacity/stockpile;")
    md.append("annotations are sign stability under parameter fog (fraction of draws")
    md.append("agreeing with the majority sign) and how often the entry ranks #1.\n")
    md.append("| capacity | single-scenario | prior-averaged | sign under fog (single / prior-avg) | #1 (single / prior-avg) |")
    md.append("|---|---:|---:|---|---:|")
    order = jnp.argsort(-gec)
    for i in [int(j) for j in order]:
        t = TNAMES[i]
        pa, pb = a["per_cap"][t], b["per_cap"][t]
        md.append(f"| {t} | {float(g1c[i]):.2f} | {float(gec[i]):.2f} "
                  f"| {sign_label(pa)} / {sign_label(pb)} "
                  f"| {pa['top1_pct']:.0f}% / {pb['top1_pct']:.0f}% |")
    md.append("")
    md.append("| stockpile | single-scenario | prior-averaged | sign under fog (single / prior-avg) | #1 (single / prior-avg) |")
    md.append("|---|---:|---:|---|---:|")
    order = jnp.argsort(-ges)
    for i in [int(j) for j in order]:
        p = PLACES[i]
        pa, pb = a["per_stk"][p], b["per_stk"][p]
        md.append(f"| {p} | {float(g1s[i]):.3f} | {float(ges[i]):.3f} "
                  f"| {sign_label(pa)} / {sign_label(pb)} "
                  f"| {pa['top1_pct']:.0f}% / {pb['top1_pct']:.0f}% |")
    md.append("")

    md.append("## 2. Mean objective vs tail objective\n")
    md.append("Scenario throughputs under the frozen prior, and the CVaR tail weights")
    md.append("(scenarios sorted worst-first; fractional inclusion at the boundary):\n")
    md.append("| scenario | prior prob | total throughput | CVaR weight |")
    md.append("|---|---:|---:|---:|")
    for i in range(len(SCEN_NAMES)):
        md.append(f"| {SCEN_NAMES[i]} | {float(SCEN_PROB[i]):.2f} "
                  f"| {float(T[i]):.1f} | {float(w[i]):.2f} |")
    md.append("")
    md.append("| rank | mean-objective capacity | tail-objective capacity |")
    md.append("|---:|---|---|")
    om = [TNAMES[int(i)] for i in jnp.argsort(-gec)][:5]
    ot = [TNAMES[int(i)] for i in jnp.argsort(-gcv)][:5]
    for r in range(5):
        md.append(f"| {r+1} | {om[r]} | {ot[r]} |")
    md.append("")
    md.append("The tail objective is an analysis here, not the frozen objective —")
    md.append("METHODOLOGY.md carries it as known limit #4. Its top stockpile is")
    md.append(f"**{PLACES[int(jnp.argmax(gsv))]}**.\n")

    md.append("## 3. The buffer rule, event by event\n")
    md.append("A shock bites iff buffer-months of cover < integrated capacity deficit")
    md.append("(month-equivalents of lost supply while the ramp completes). v0 deficit")
    md.append("= capacity-lost x ramp-tau (analytic); v1 deficit measured from the")
    md.append("emergent capacity trajectory under the frozen adaptation law.\n")
    md.append("| event | buffer (mo) | deficit v0 (mo) | deficit v1 (mo) | v1 outcome |")
    md.append("|---|---:|---:|---:|---|")
    for name, bm, d0, d1, dip, rec in btab:
        md.append(f"| {name} | {bm:.1f} | {d0:.1f} | {d1:.1f} "
                  f"| dip {dip:.1f}%, rec {fmt_rec(rec)} |")
    md.append("")
    md.append("Buffer > deficit -> the shock is invisible (neon historical, Tohoku,")
    md.append("photoresist realized). Buffer < deficit -> it bites (neon lean")
    md.append("counterfactual, Sumitomo, photoresist feared). One inequality decides")
    md.append("all six rows — which is why conclusion #1 is a sizing rule, not a")
    md.append("slogan. Worth savoring: the industry's post-2014 neon stockpile (~6.0")
    md.append("months) sits just above the measured deficit (5.7 month-equivalents) —")
    md.append("the 2014 lesson bought almost exactly the right amount of insurance.")
    md.append("(The Sumitomo row's *depth* is real but its never-recovery is the known")
    md.append("hysteresis artifact; see PRICE_EXPERIMENT.md.)\n")

    md.append("**What this table cannot say.** Magnitudes inherit confidence-C fog;")
    md.append("demand-side crises (chip crunch 2021) are out of scope; and the")
    md.append("pessimism bias means model dips are ceilings, not forecasts. The")
    md.append("measured lever for sharpening all of it: real citations on the map's")
    md.append("parameters (the sensitivity audit shows rankings stabilize when")
    md.append("parameter fog, not prior fog, is removed).")
    md.append("")
    return "\n".join(md)


def main():
    smoke = "--smoke" in sys.argv
    print("selfcheck against frozen baseline...", flush=True)
    engine.selfcheck()
    sens = load_sensitivity()

    g1c, g1s, gec, ges, cap_cal, x_stock = gradient_vectors()
    T, w, gcv, gsv = cvar_gradients(cap_cal, x_stock)
    print("scenario throughputs:", [f"{float(t):.1f}" for t in T], flush=True)
    print("CVaR weights:", [f"{float(x):.2f}" for x in w], flush=True)
    print(f"tail top capacity: {TNAMES[int(jnp.argmax(gcv))]}; "
          f"tail top stockpile: {PLACES[int(jnp.argmax(gsv))]}", flush=True)
    btab = buffer_table()
    for row in btab:
        print(f"   {row[0]:<32} buffer {row[1]:4.1f}mo  deficit v0 {row[2]:4.1f} "
              f"v1 {row[3]:4.1f}  dip {row[4]:6.1f}%  rec {fmt_rec(row[5])}", flush=True)

    md = build_markdown(sens, g1c, g1s, gec, ges, T, w, gcv, gsv, btab)
    if smoke:
        print("\n--- smoke run: not writing POLICY.md ---")
        return
    (ROOT / "POLICY.md").write_text(md)
    print(f"\nwrote {ROOT / 'POLICY.md'}")


if __name__ == "__main__":
    main()
