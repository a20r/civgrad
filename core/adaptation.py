"""
core/adaptation.py — endogenous adaptation: capacity responds to scarcity.

Replaces the imposed recovery curve (tau) with a control law:
    scarcity_p = relu(x_ref_p - x_p) / x_ref_p          (stock below reference)
    dc_i/dt    = ALPHA * c_base_i * max_{p in out(i)} scarcity_p

Recovery time is now EMERGENT. ALPHA is fitted once on the neon training
event, then FROZEN for the holdouts. Disruptions are pure impulses:
capacity is destroyed at t0 and only the adaptation law rebuilds it.

Ported unchanged from gsc_adapt.py — validation/baseline_outputs.txt is the
regression oracle. `python3 -m core.adaptation` reproduces the baseline demo,
including the ALPHA fitting sweep on the training event.
"""

import jax
import jax.numpy as jnp

from core.continuous import (Pre, Post, TRANSITIONS, PLACES, P, T_IDX, NT, NP_,
                             flows, cap0, burn_in, UTILIZATION)
from core.map_loader import frozen_globals

_FROZEN = frozen_globals()

DT = 0.1          # 1 unit = 1 month
HORIZON = _FROZEN["HORIZON"]
                  # forward-looking planning horizon (months of runway)
                  # FROZEN global parameter (map frozen_globals): change only
                  # via a methodology PR (AGENTS.md).
FAB = T_IDX["Fab"]
SHIP = T_IDX["Ship_Strait"]

# out_mask[i, p] = 1 if transition i produces place p
OUT = (Post > 0).astype(jnp.float64)


def simulate(cap_base, x0, x_ref, tidx, kill, alpha, t0=6.0, months=72):
    """Impulse disruption + endogenous rebuild. Returns fab & ship flows."""
    hit = jnp.zeros(NT).at[tidx].set(1.0)
    v_ref = flows(x_ref, cap_base)   # pre-crisis throughput = demand anchor

    def step(carry, k):
        x, c = carry
        t = k * DT
        c = jnp.where(jnp.abs(t - t0) < DT / 2, c * (1.0 - kill * hit), c)
        v0 = flows(x, c)
        inflow  = Post.T @ v0
        outflow = Pre.T @ v0
        net_drain = jnp.maximum(outflow - inflow, 1e-9)
        runway = x / net_drain                      # months of cover
        scar = jnp.clip(1.0 - runway / HORIZON, 0.0, 1.0)
        scar = jnp.where(outflow - inflow > 1e-6, scar, 0.0)
        runway_scar = jnp.max(OUT * scar[None, :], axis=1)
        flow_scar = jnp.clip((v_ref - v0) / (v_ref + 1e-9) / 0.10, 0.0, 1.0)  # saturates at 10% shortfall
        restore_gap = jnp.clip((cap_base - c) / (1e-6 * cap_base + 1e-9), 0.0, 1.0)
        prod_scar = jnp.maximum(runway_scar, flow_scar * restore_gap)
        c = c + DT * alpha * cap_base * prod_scar
        v = flows(x, c)
        x = jnp.clip(x + DT * ((Post - Pre).T @ v), 0.0)
        return (x, c), (v[FAB], v[SHIP])

    (_, _), (fab, ship) = jax.lax.scan(step, (x0, cap_base), jnp.arange(int(months / DT)))
    return fab, ship


def dip_recovery(traj, t0=6.0):
    i0 = int(t0 / DT)
    pre = jnp.mean(traj[i0 - int(3 / DT):i0])
    post = traj[i0:]
    imin = int(jnp.argmin(post))
    dip = float(1.0 - post[imin] / pre)
    after = post[imin:]
    if float(jnp.max(after)) > 0.95 * float(pre):
        rec = (imin + int(jnp.argmax(after > 0.95 * pre))) * DT
    else:
        rec = float("inf")
    return dip, rec


# ---------------- frozen calibration (identical rule to before) ----------------
c = cap0
x = jnp.full(NP_, 1.0)
for _ in range(2):
    xs, F = burn_in(c, x)
    for tn in ["Purify_Ne", "WaferSupply", "Refine_Ga"]:
        c = c.at[T_IDX[tn]].set(F / UTILIZATION)
    x = xs
X_REF, F = burn_in(c, x)
X_JIT = X_REF
for pn in ["Ne_purified", "Wafers", "Ga_refined", "Chips", "Pkg"]:
    X_JIT = X_JIT.at[P[pn]].set(0.5 * F)
X_REF = X_JIT  # reference = normal JIT operating stocks

# Fitted on the neon training event (see the sweep in __main__), then FROZEN
# for all holdouts (map frozen_globals). Change only via a methodology PR
# (AGENTS.md).
ALPHA = _FROZEN["ALPHA"]


def run(buffer_place, buffer_months, tidx, kill, alpha, observe="fab"):
    x0 = X_JIT.at[P[buffer_place]].set(buffer_months * F)
    fab, ship = simulate(c, x0, X_REF, T_IDX[tidx], kill, alpha)
    return dip_recovery(fab if observe == "fab" else ship)


if __name__ == "__main__":
    # ---- fit ALPHA on the neon TRAINING event ----
    # target: counterfactual (no stockpile) normalizes in 12-18mo
    print("== fitting ALPHA on neon (training event) ==")
    for a in [0.02, 0.04, 0.06, 0.08, 0.12]:
        d, r = run("Ne_purified", 0.5, "Purify_Ne", 0.5, a)
        print(f"   alpha={a:.2f}: counterfactual dip {100*d:5.1f}%, recovery {r:5.1f}mo")

    d, r = run("Ne_purified", 6.0, "Purify_Ne", 0.5, ALPHA)
    print(f"\n   ALPHA frozen at {ALPHA}")
    print(f"   neon WITH 6mo stockpile: dip {100*d:.1f}%, recovery "
          f"{'n/a' if d < 0.05 else f'{r:.0f}mo'}   | history: no stoppage, ~12-18mo normalization")

    # ---- holdouts, ALPHA frozen ----
    print("\n== HOLDOUTS (alpha frozen, tau emergent) ==")
    d, r = run("Wafers", 2.0, "WaferSupply", 0.25, ALPHA)
    print(f"   Tohoku 2011:        dip {100*d:5.1f}%, rec {r:5.1f}mo | history: minor impact")
    d, r = run("Chips", 1.5, "Package", 0.60, ALPHA, observe="ship")
    print(f"   Sumitomo 1993:      dip {100*d:5.1f}%, rec {r:5.1f}mo | history: brief pain, no catastrophe")
    d, r = run("Ne_purified", 2.0, "Purify_Ne", 0.15, ALPHA)
    print(f"   Photoresist 2019")
    print(f"     realized (~15%):  dip {100*d:5.1f}%, rec {r:5.1f}mo | history: non-event")
    d, r = run("Ne_purified", 2.0, "Purify_Ne", 0.90, ALPHA)
    print(f"     feared (90%):     dip {100*d:5.1f}%, rec {r:5.1f}mo | (counterfactual: what the fear implied)")
