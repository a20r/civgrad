"""
analysis/sensitivity.py — the fog audit: how much of the headline story
survives when the confidence-C parameters are wrong?

Every capacity in the map and K_SAT are session estimates (confidence C).
The essay's standing claim is "trust the signs and rankings, distrust the
magnitudes" — this sweep measures that claim instead of arguing it.

Protocol (all draws seeded, jax.random.PRNGKey(0); nothing here feeds back
into the frozen model or the scorecard):

  A. PARAMETER FOG, single-scenario objective (core/gradients.py context:
     raw map capacities, uniform stocks, 90% EUV destruction). Each draw
     multiplies the 13 transition capacities and K_SAT by independent
     lognormal factors with sigma = ln(2)/1.96 — 95% of factors inside
     [x0.5, x2], the honest reading of "confidence C". Recompute both
     gradient vectors per draw.

  B. PARAMETER FOG, prior-averaged objective (validation/runner.py context:
     neon-calibrated capacities, stockpiled stocks, expectation over the
     5-scenario disruption prior). Same draws; the calibration is re-run
     per draw, as the frozen protocol would.

  C. DISRUPTION-PRIOR FOG: parameters at their frozen values; the prior
     itself perturbed (probabilities: lognormal then renormalized;
     severities: logit-normal; rebuild taus: lognormal).

  D. STRUCTURAL-MISS CHECK: the Sumitomo v1 replay re-run under the
     parameter-fog draws; a draw "still misses" when it fails the event's
     frozen acceptance band (dip 15±15%, recovery 3±3 mo). If nearly all
     nearby parameterizations miss, no data PR can fix the miss — the gap
     is structural (no demand/allocation mechanism), which is what the
     essay claims.

Validity: a draw is valid when every gradient component is finite (extreme
capacity draws can break the fixed-step Euler integrator). Headline
stability numbers are reported over valid draws, WITH worst-case floors
that count every invalid draw as a flipped sign.

  python3 -m analysis.sensitivity            # full sweep, writes SENSITIVITY.md
                                             # + analysis/out/sensitivity.json
  python3 -m analysis.sensitivity --smoke    # 8 draws, prints only (CI-sized)
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from core.continuous import P, T_IDX, NP_, PLACES, TRANSITIONS, cap0
from analysis import engine
from analysis.engine import (K_SAT0, UTILIZATION0, ALPHA0, HORIZON0,
                             SCEN_PROB, SCEN_TIDX, SCEN_KILL, SCEN_TAU)

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
N_DRAWS = 500
SIGMA_PARAM = float(jnp.log(2.0) / 1.96)   # 95% of factors within [x1/2, x2]
SIGMA_PRIOR = 0.5                           # prior-fog: probs and severities
SIGMA_TAU = SIGMA_PARAM

# The claims under audit, named once. Each claim is audited only in the
# regime where the frozen baseline actually makes it: the negative stockpile
# gradients (Pkg, Chips) and the boneyard-on-top ordering exist in the
# single-scenario run; the prior-averaged baseline has all-positive stockpile
# gradients and the boneyard ordering already flipped (see
# validation/baseline_outputs.txt and the runner's expected-gradient section).
NEG_CAPS = ["Ship_Strait", "Package", "Wear_EUV"]   # negative in BOTH objectives
NEG_STKS = ["Pkg", "Chips"]                          # negative in single-scenario only
TOP3 = {"single": ["Fab", "Build_EUV", "Purify_Ne"],
        "expected": ["Fab", "Build_EUV", "Refine_Ga"]}
SUMITOMO_BAND = dict(dip=(0.0, 30.0), rec=(0.0, 6.0))  # frozen band, from the yaml

# The partition claim: production/equipment complex vs the downstream
# logistics chain (Package, Ship_Strait, Consume, Recycle) on the capacity
# side; upstream + equipment stocks vs downstream inventories (Chips, Pkg,
# Goods, E_waste) on the stockpile side. (Wear_EUV is excluded from the
# upstream capacity set: "wear capacity" is not an investable thing.)
UP_CAPS = {"Mine", "WaferSupply", "OpticsMfg", "Refine_Ga", "Purify_Ne",
           "Fab", "Build_EUV", "Refurb"}
UP_STKS = {"Ga_byproduct", "Ga_refined", "Ne_crude", "Ne_purified",
           "EUV_tools", "EUV_optics", "EUV_worn", "Wafers"}

TNAMES = [t[0] for t in TRANSITIONS]


def _draw_factors(key, n, dim, sigma):
    return jnp.exp(sigma * jax.random.normal(key, (n, dim)))


def _kendall_tau(a, b):
    """Kendall rank correlation between two 1-D vectors (O(n^2), n<=13)."""
    n = a.shape[0]
    ii, jj = jnp.triu_indices(n, k=1)
    sa = jnp.sign(a[ii] - a[jj])
    sb = jnp.sign(b[ii] - b[jj])
    return jnp.sum(sa * sb) / ii.shape[0]


# ------------------------------------------------------------- sweeps ----

def sweep_single(cap_factors, ksat_factors):
    """Objective A: per-draw gradient vectors of disrupted throughput."""
    x0 = jnp.full(NP_, 1.0)
    thru = lambda c, x, k: engine.simulate_throughput(c, x, k, 0.9)

    def one(fc, fk):
        cap = cap0 * fc
        k = K_SAT0 * fk
        g_c = jax.grad(thru, argnums=0)(cap, x0, k)
        g_s = jax.grad(thru, argnums=1)(cap, x0, k)
        return g_c, g_s

    return jax.lax.map(lambda t: one(*t), (cap_factors, ksat_factors))


def sweep_expected(cap_factors, ksat_factors):
    """Objective B: per-draw gradients of E[throughput] over the prior, in
    the runner's context (neon calibration re-run per draw)."""
    def one(fc, fk):
        cap = cap0 * fc
        k = K_SAT0 * fk
        cap_cal, x_stock, _, _ = engine.calibrate_neon_v0(
            cap, k, UTILIZATION0, lean_months=0.5, stock_months=6.0)
        f = lambda c, x: engine.expected_throughput(c, x, k)
        g_c = jax.grad(f, argnums=0)(cap_cal, x_stock)
        g_s = jax.grad(f, argnums=1)(cap_cal, x_stock)
        return g_c, g_s

    return jax.lax.map(lambda t: one(*t), (cap_factors, ksat_factors))


def sweep_prior(key, n):
    """Objective C: frozen parameters, fogged prior."""
    kp, kk, kt = jax.random.split(key, 3)
    probs = SCEN_PROB[None, :] * jnp.exp(
        SIGMA_PRIOR * jax.random.normal(kp, (n, 5)))
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)
    logit = jnp.log(SCEN_KILL / (1 - SCEN_KILL))
    kills = jax.nn.sigmoid(logit[None, :]
                           + SIGMA_PRIOR * jax.random.normal(kk, (n, 5)))
    taus = SCEN_TAU[None, :] * jnp.exp(
        SIGMA_TAU * jax.random.normal(kt, (n, 5)))

    cap_cal, x_stock, _, _ = engine.calibrate_neon_v0(
        cap0, K_SAT0, UTILIZATION0, lean_months=0.5, stock_months=6.0)

    def one(pr, ki, ta):
        f = lambda c, x: engine.expected_throughput(
            c, x, K_SAT0, prob=pr, kill=ki, tau=ta)
        g_c = jax.grad(f, argnums=0)(cap_cal, x_stock)
        g_s = jax.grad(f, argnums=1)(cap_cal, x_stock)
        return g_c, g_s

    return jax.lax.map(lambda t: one(*t), (probs, kills, taus))


def sweep_sumitomo(cap_factors, ksat_factors):
    """Objective D: the Sumitomo v1 replay per parameter-fog draw."""
    ev = engine.V1_EVENTS["sumitomo_1993"]

    def one(fc, fk):
        d, r = engine.v1_replay(cap0 * fc, K_SAT0 * fk, ALPHA0, HORIZON0,
                                UTILIZATION0, **ev)
        return 100.0 * d, r

    return jax.lax.map(lambda t: one(*t), (cap_factors, ksat_factors))


# ------------------------------------------------------------- scoring ----

def score_gradients(g_c, g_s, label):
    """Stability metrics for one sweep's stacked gradient draws."""
    valid = (jnp.all(jnp.isfinite(g_c), axis=1)
             & jnp.all(jnp.isfinite(g_s), axis=1))
    n, nv = int(g_c.shape[0]), int(jnp.sum(valid))
    gc, gs = g_c[valid], g_s[valid]

    out = {"label": label, "draws": n, "valid": nv}
    out["neg_caps"] = {
        t: {"stable_pct": 100.0 * float(jnp.mean(gc[:, T_IDX[t]] < 0)),
            "floor_pct": 100.0 * float(jnp.sum(gc[:, T_IDX[t]] < 0)) / n}
        for t in NEG_CAPS}
    out["neg_stks"] = {
        p: {"stable_pct": 100.0 * float(jnp.mean(gs[:, P[p]] < 0)),
            "floor_pct": 100.0 * float(jnp.sum(gs[:, P[p]] < 0)) / n}
        for p in NEG_STKS}
    out["worn_on_top_pct"] = 100.0 * float(
        jnp.mean(gs[:, P["EUV_worn"]] > gs[:, P["EUV_tools"]]))

    top3_base = set(TOP3[label])
    ranks = jnp.argsort(-gc, axis=1)[:, :3]
    hits = [set(TNAMES[int(j)] for j in row) == top3_base for row in ranks]
    out["top3_same_pct"] = 100.0 * float(jnp.mean(jnp.array(hits)))
    out["fab_first_pct"] = 100.0 * float(
        jnp.mean(jnp.argmax(gc, axis=1) == T_IDX["Fab"]))
    lead = [set(TNAMES[int(j)] for j in row) >= {"Fab", "Build_EUV"}
            for row in ranks]
    out["fab_build_top3_pct"] = 100.0 * float(jnp.mean(jnp.array(lead)))

    import collections
    top_c = [TNAMES[int(i)] for i in jnp.argmax(gc, axis=1)]
    top_s = [PLACES[int(i)] for i in jnp.argmax(gs, axis=1)]
    out["top_cap_upstream_pct"] = 100.0 * sum(t in UP_CAPS for t in top_c) / len(top_c)
    out["top_stk_upstream_pct"] = 100.0 * sum(p in UP_STKS for p in top_s) / len(top_s)
    name, cnt = collections.Counter(top_c).most_common(1)[0]
    out["modal_top_cap"] = name
    out["modal_top_cap_pct"] = 100.0 * cnt / len(top_c)

    # per-node stats, consumed by analysis/policy.py. pos/neg are STRICT
    # directional fractions — they need not sum to 100 (a few draws produce
    # exactly-zero gradients, which support neither sign).
    out["per_cap"] = {
        t: {"pos_pct": 100.0 * float(jnp.mean(gc[:, T_IDX[t]] > 0)),
            "neg_pct": 100.0 * float(jnp.mean(gc[:, T_IDX[t]] < 0)),
            "top1_pct": 100.0 * top_c.count(t) / len(top_c)}
        for t in TNAMES}
    out["per_stk"] = {
        p: {"pos_pct": 100.0 * float(jnp.mean(gs[:, P[p]] > 0)),
            "neg_pct": 100.0 * float(jnp.mean(gs[:, P[p]] < 0)),
            "top1_pct": 100.0 * top_s.count(p) / len(top_s)}
        for p in PLACES}

    base_g = jnp.asarray([float(x) for x in _baseline_for(label)])
    taus = jax.vmap(lambda g: _kendall_tau(g, base_g))(gc)
    out["kendall_tau_mean"] = float(jnp.mean(taus))
    out["kendall_tau_p10"] = float(jnp.percentile(taus, 10))
    return out


_BASELINE_CACHE = {}


def _baseline_for(label, which="cap"):
    """Unperturbed gradient vectors for the sweep's objective."""
    if label not in _BASELINE_CACHE:
        x0 = jnp.full(NP_, 1.0)
        if label == "single":
            thru = lambda c, x: engine.simulate_throughput(c, x, K_SAT0, 0.9)
            _BASELINE_CACHE[label] = (jax.grad(thru, argnums=0)(cap0, x0),
                                      jax.grad(thru, argnums=1)(cap0, x0))
        else:
            cap_cal, x_stock, _, _ = engine.calibrate_neon_v0(
                cap0, K_SAT0, UTILIZATION0, 0.5, 6.0)
            f = lambda c, x: engine.expected_throughput(c, x, K_SAT0)
            _BASELINE_CACHE[label] = (jax.grad(f, argnums=0)(cap_cal, x_stock),
                                      jax.grad(f, argnums=1)(cap_cal, x_stock))
    return _BASELINE_CACHE[label][0 if which == "cap" else 1]


def score_sumitomo(dips, recs):
    finite_ok = jnp.isfinite(dips)
    n = int(dips.shape[0])
    lo_d, hi_d = SUMITOMO_BAND["dip"]
    lo_r, hi_r = SUMITOMO_BAND["rec"]
    in_band = ((dips >= lo_d) & (dips <= hi_d)
               & jnp.isfinite(recs) & (recs >= lo_r) & (recs <= hi_r))
    in_band = in_band & finite_ok
    return {"draws": n,
            "valid": int(jnp.sum(finite_ok)),
            "still_miss_pct": 100.0 * float(1 - jnp.mean(in_band)),
            "median_dip_pct": float(jnp.median(dips[finite_ok])),
            "never_recover_pct": 100.0 * float(jnp.mean(~jnp.isfinite(recs)))}


# -------------------------------------------------------------- report ----

def fmt_pct(x):
    return f"{x:.0f}%"


def build_markdown(res):
    a, b, c, d = res["single"], res["expected"], res["prior"], res["sumitomo"]
    md = []
    md.append("# civgrad — sensitivity audit (the fog, measured)\n")
    md.append("<!-- AUTO-GENERATED by `python3 -m analysis.sensitivity` — do not edit by hand. -->\n")
    md.append("Every capacity in the map and K_SAT are confidence-C session estimates.")
    md.append("The standing claim on every output is *trust the signs and rankings,*")
    md.append("*distrust the magnitudes*. This audit measures that claim: perturb the")
    md.append("confidence-C parameters (and, separately, the disruption prior), recompute")
    md.append("the gradients, and count what survives. Nothing here feeds back into the")
    md.append("frozen model or the scorecard.\n")
    md.append(f"**Protocol.** {res['n']} seeded draws per sweep (PRNGKey({SEED})).")
    md.append(f"Parameter fog: each of the 13 transition capacities and K_SAT multiplied")
    md.append(f"by an independent lognormal factor, sigma=ln(2)/1.96 — 95% of factors")
    md.append("inside [x0.5, x2], the honest reading of confidence C. Prior fog:")
    md.append("scenario probabilities (lognormal, renormalized), severities")
    md.append("(logit-normal, sigma=0.5), and rebuild taus (lognormal) perturbed with")
    md.append("parameters frozen. A draw is *valid* when every gradient component is")
    md.append("finite (extreme draws can break the fixed-step integrator); stability is")
    md.append("reported over valid draws, with a worst-case floor counting every invalid")
    md.append("draw as a flipped sign.\n")

    md.append("## 1. Sign stability of the negative gradients\n")
    md.append("Each claim is audited only in the regime where the frozen baseline makes")
    md.append("it: the negative Pkg/Chips stockpile gradients exist in the")
    md.append("single-scenario run only (the prior-averaged baseline's stockpile")
    md.append("gradients are all positive).\n")
    g1c, gec = _baseline_for("single"), _baseline_for("expected")
    g1s = _baseline_for("single", "stk")
    md.append("| claim (baseline value, single / prior-avg) | single-scenario (param fog) | prior-averaged (param fog) | prior fog |")
    md.append("|---|---:|---:|---:|")
    for t in NEG_CAPS:
        base = f"({float(g1c[T_IDX[t]]):.2f} / {float(gec[T_IDX[t]]):.2f})"
        md.append(f"| d(thru)/d(cap) {t} < 0 {base} "
                  f"| {fmt_pct(a['neg_caps'][t]['stable_pct'])} (floor {fmt_pct(a['neg_caps'][t]['floor_pct'])}) "
                  f"| {fmt_pct(b['neg_caps'][t]['stable_pct'])} (floor {fmt_pct(b['neg_caps'][t]['floor_pct'])}) "
                  f"| {fmt_pct(c['neg_caps'][t]['stable_pct'])} |")
    for p in NEG_STKS:
        md.append(f"| d(thru)/d(stock) {p} < 0 ({float(g1s[P[p]]):.3f} / —) "
                  f"| {fmt_pct(a['neg_stks'][p]['stable_pct'])} (floor {fmt_pct(a['neg_stks'][p]['floor_pct'])}) "
                  f"| — | — |")
    md.append("")
    md.append("Sign stability tracks magnitude, which is what it should do: the two")
    md.append("large negative gradients (shipping, wear) hold in ~85-92% of parameter-fog")
    md.append("draws and 100% of prior-fog draws; the near-zero negatives (Package")
    md.append("capacity, the Pkg/Chips stockpiles) flip freely — the model is confident")
    md.append("about the big signs and honestly noisy about the small ones. Claims built")
    md.append("on the small negatives do not survive audit and are not made anywhere in")
    md.append("the essay's headline results.\n")
    md.append(f"Valid draws: single {a['valid']}/{a['draws']}, "
              f"prior-averaged {b['valid']}/{b['draws']}, prior fog {c['valid']}/{c['draws']}.\n")

    md.append("## 2. Rankings under fog\n")
    md.append("| metric | single-scenario | prior-averaged | prior fog |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| top capacity in the production/equipment complex | {fmt_pct(a['top_cap_upstream_pct'])} "
              f"| {fmt_pct(b['top_cap_upstream_pct'])} | {fmt_pct(c['top_cap_upstream_pct'])} |")
    md.append(f"| top stockpile an equipment or fab-input stock | {fmt_pct(a['top_stk_upstream_pct'])} "
              f"| {fmt_pct(b['top_stk_upstream_pct'])} | {fmt_pct(c['top_stk_upstream_pct'])} |")
    md.append(f"| modal #1 capacity | {a['modal_top_cap']} ({fmt_pct(a['modal_top_cap_pct'])}) "
              f"| {b['modal_top_cap']} ({fmt_pct(b['modal_top_cap_pct'])}) "
              f"| {c['modal_top_cap']} ({fmt_pct(c['modal_top_cap_pct'])}) |")
    md.append(f"| Fab ranked #1 capacity | {fmt_pct(a['fab_first_pct'])} "
              f"| {fmt_pct(b['fab_first_pct'])} | {fmt_pct(c['fab_first_pct'])} |")
    md.append(f"| exact top-3 capacity set unchanged | {fmt_pct(a['top3_same_pct'])} "
              f"| {fmt_pct(b['top3_same_pct'])} | {fmt_pct(c['top3_same_pct'])} |")
    md.append(f"| Kendall tau vs baseline ranking (mean) | {a['kendall_tau_mean']:.2f} "
              f"| {b['kendall_tau_mean']:.2f} | {c['kendall_tau_mean']:.2f} |")
    md.append(f"| Kendall tau (10th percentile) | {a['kendall_tau_p10']:.2f} "
              f"| {b['kendall_tau_p10']:.2f} | {c['kendall_tau_p10']:.2f} |")
    md.append(f"| boneyard: worn > working spares | {fmt_pct(a['worn_on_top_pct'])} "
              f"| {fmt_pct(b['worn_on_top_pct'])} | {fmt_pct(c['worn_on_top_pct'])} |")
    md.append("")
    md.append("Reading: **what survives parameter fog is the partition, not the")
    md.append("ordering.** In ~99%/91% of draws the top capacity sits in the")
    md.append("production/equipment complex — the downstream logistics chain leads")
    md.append("essentially never — and the top stockpile is an equipment or fab-input")
    md.append("stock, not downstream inventory. But WHICH node inside the complex leads")
    md.append("is fog-conditional (Fab is #1 in only ~10% of parameter-fog draws;")
    md.append("Build_EUV is the modal leader), full rank order runs Kendall tau ~0.5,")
    md.append("and the boneyard-on-top ordering survives only in a minority of draws —")
    md.append("the honest version of that result is *comparable value*, not *highest")
    md.append("value*. Under prior fog alone (parameters held at their estimates) the")
    md.append("full ranking is stable (tau ~0.9, Fab #1 in ~94%): the ranking's")
    md.append("weakness is parameter ignorance, not prior choice — which is exactly")
    md.append("what data PRs with real citations would sharpen.\n")

    md.append("## 3. Is the Sumitomo miss structural?\n")
    md.append(f"Re-running the Sumitomo 1993 v1 replay under the same parameter fog:")
    md.append(f"**{fmt_pct(d['still_miss_pct'])} of {d['draws']} draws still miss the")
    md.append(f"frozen acceptance band** (dip 15±15%, recovery 3±3 mo);")
    md.append(f"{fmt_pct(d['never_recover_pct'])} never recover within the 72-month")
    md.append(f"horizon (median dip {d['median_dip_pct']:.0f}%). The miss is not a")
    md.append("bad-parameter accident a data PR could fix — it is structural, which is")
    md.append("exactly what the missing-mechanism diagnosis (no price-mediated")
    md.append("allocation; PLAN.md §5, PRICE_EXPERIMENT.md) predicts.\n")

    md.append("**Caveats.** The fog model is itself a modeling choice (independent")
    md.append("lognormal factors; no correlated errors). The disruption prior has five")
    md.append("scenarios — prior fog explores their neighborhood, not scenarios the")
    md.append("prior cannot express. And the pessimism-bias caveat travels with every")
    md.append("number here: no price-mediated allocation, so depths and durations are")
    md.append("overstated (SCORECARD.md).")
    md.append("")
    return "\n".join(md)


def main():
    smoke = "--smoke" in sys.argv
    n = 8 if smoke else N_DRAWS
    print(f"selfcheck against frozen baseline...", flush=True)
    engine.selfcheck()
    print(f"sweeps: {n} draws each (seed {SEED}, sigma {SIGMA_PARAM:.3f})", flush=True)

    key = jax.random.PRNGKey(SEED)
    k_ab, k_c = jax.random.split(key)
    kc, kk = jax.random.split(k_ab)
    cap_f = _draw_factors(kc, n, len(TNAMES), SIGMA_PARAM)
    ksat_f = _draw_factors(kk, n, 1, SIGMA_PARAM)[:, 0]

    res = {"n": n, "seed": SEED, "sigma_param": SIGMA_PARAM,
           "sigma_prior": SIGMA_PRIOR}
    gc, gs = sweep_single(cap_f, ksat_f)
    res["single"] = score_gradients(gc, gs, "single")
    print(f"   single-scenario sweep done ({res['single']['valid']}/{n} valid)", flush=True)
    gc, gs = sweep_expected(cap_f, ksat_f)
    res["expected"] = score_gradients(gc, gs, "expected")
    print(f"   prior-averaged sweep done ({res['expected']['valid']}/{n} valid)", flush=True)
    gc, gs = sweep_prior(k_c, n)
    res["prior"] = score_gradients(gc, gs, "expected")
    print(f"   prior-fog sweep done ({res['prior']['valid']}/{n} valid)", flush=True)
    dips, recs = sweep_sumitomo(cap_f, ksat_f)
    res["sumitomo"] = score_sumitomo(dips, recs)
    print(f"   sumitomo structural sweep done "
          f"(still miss: {res['sumitomo']['still_miss_pct']:.0f}%)", flush=True)

    md = build_markdown(res)
    if smoke:
        print("\n--- smoke run: not writing SENSITIVITY.md ---\n")
        print(md)
        return

    (ROOT / "SENSITIVITY.md").write_text(md)
    outdir = ROOT / "analysis" / "out"
    outdir.mkdir(exist_ok=True)

    def _plain(o):
        if isinstance(o, dict):
            return {k: _plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_plain(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        return o

    (outdir / "sensitivity.json").write_text(
        json.dumps(_plain(res), indent=1, sort_keys=True))
    print(f"\nwrote {ROOT / 'SENSITIVITY.md'} and analysis/out/sensitivity.json")


if __name__ == "__main__":
    main()
