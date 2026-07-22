"""
gsc_grad.py — Differentiable continuous Petri net (David & Alla style) in JAX.

The discrete net from gsc_net.py, relaxed:
  markings  -> real-valued stocks x(t)
  firings   -> flow rates v_i(x) = capacity_i * min over input saturations
  dynamics  -> dx/dt = (Post - Pre)^T v(x)   (piecewise-smooth ODE, Euler-integrated)

Read arcs multiply flow by a saturation of the read place (equipment enables
flow without being consumed). Saturation is Michaelis-Menten: s(x) = x/(x+k),
smooth so JAX can differentiate through everything.

Experiment:
  - run to steady state
  - at t_d, destroy a fraction of EUV_tools (the disruption)
  - loss = total chip throughput over the horizon
  - compute d(throughput)/d(capacity_i)      -> where to invest in CAPACITY
  -         d(throughput)/d(initial stocks)  -> where to invest in STOCKPILES

The ranked gradient vector is the project thesis made literal:
marginal resilience per marginal unit of investment.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------- net definition (mirrors gsc_net.py) ----------------

PLACES = ["Ga_byproduct", "Ga_refined", "Ne_crude", "Ne_purified",
          "EUV_tools", "EUV_optics", "EUV_worn",
          "Wafers", "Chips", "Pkg", "Goods", "E_waste"]
P = {n: i for i, n in enumerate(PLACES)}
NP_ = len(PLACES)

# (name, inputs{place:w}, outputs{place:w}, reads{place})
TRANSITIONS = [
    ("Mine",        {},                                {"Ga_byproduct": 1, "Ne_crude": 1}, []),
    ("WaferSupply", {},                                {"Wafers": 1},                      []),
    ("OpticsMfg",   {},                                {"EUV_optics": 1},                  []),
    ("Refine_Ga",   {"Ga_byproduct": 1},               {"Ga_refined": 1},                  []),
    ("Purify_Ne",   {"Ne_crude": 1},                   {"Ne_purified": 1},                 []),
    ("Fab",         {"Ga_refined": 1, "Ne_purified": 1,
                     "Wafers": 1},                     {"Chips": 1},                       ["EUV_tools"]),
    ("Package",     {"Chips": 1},                      {"Pkg": 1},                         []),
    ("Ship_Strait", {"Pkg": 1},                        {"Goods": 1},                       []),
    ("Consume",     {"Goods": 1},                      {"E_waste": 1},                     []),
    ("Recycle",     {"E_waste": 1},                    {"Ga_byproduct": 1},                []),
    ("Build_EUV",   {"Pkg": 1, "EUV_optics": 1},       {"EUV_tools": 1},                   []),
    ("Wear_EUV",    {"EUV_tools": 1},                  {"EUV_worn": 1},                    []),
    ("Refurb",      {"EUV_worn": 1, "Pkg": 1},         {"EUV_tools": 1},                   []),
]
NT = len(TRANSITIONS)
T_IDX = {t[0]: i for i, t in enumerate(TRANSITIONS)}

Pre  = jnp.zeros((NT, NP_)).at[tuple(zip(*[(i, P[p]) for i, t in enumerate(TRANSITIONS) for p in t[1]]))].set(
       jnp.array([float(w) for t in TRANSITIONS for w in t[1].values()]))
Post = jnp.zeros((NT, NP_)).at[tuple(zip(*[(i, P[p]) for i, t in enumerate(TRANSITIONS) for p in t[2]]))].set(
       jnp.array([float(w) for t in TRANSITIONS for w in t[2].values()]))
Read = jnp.zeros((NT, NP_)).at[tuple(zip(*[(i, P[p]) for i, t in enumerate(TRANSITIONS) for p in t[3]]))].set(1.0)

K_SAT = 0.05  # ~1.5 days of stock: JIT cliff — full flow on thin inventory

def flows(x, cap):
    """v_i = cap_i * min(input saturations) * prod(read saturations)."""
    sat = x / (x + K_SAT)                                  # (NP,)
    inp = jnp.where(Pre > 0, sat[None, :], jnp.inf)        # (NT, NP)
    lim = jnp.min(inp, axis=1)
    lim = jnp.where(jnp.isinf(lim), 1.0, lim)              # source transitions
    rd  = jnp.prod(jnp.where(Read > 0, sat[None, :], 1.0), axis=1)
    return cap * lim * rd

DT, STEPS, T_DISRUPT = 0.05, 4000, 1000

def simulate(cap, x0, kill_frac):
    kill_frac = jnp.asarray(kill_frac)
    """Euler-integrate; at T_DISRUPT destroy kill_frac of EUV_tools.
    Returns total Fab throughput (the thing civilization wants maximized)."""
    fab = T_IDX["Fab"]
    def step(carry, t):
        x, tot = carry
        x = jax.lax.cond(
            t == T_DISRUPT,
            lambda x: x.at[P["EUV_tools"]].mul(1.0 - kill_frac),
            lambda x: x, x)
        v = flows(x, cap)
        x = jnp.clip(x + DT * ((Post - Pre).T @ v), 0.0)
        return (x, tot + DT * v[fab]), None
    (xf, tot), _ = jax.lax.scan(step, (x0, 0.0), jnp.arange(STEPS))
    return tot, xf

throughput = lambda cap, x0: simulate(cap, x0, kill_frac=0.9)[0]

# ---------------- baseline capacities & stocks ----------------

cap0 = jnp.ones(NT).at[T_IDX["Build_EUV"]].set(0.03)   # tools are slow to build
cap0 = cap0.at[T_IDX["OpticsMfg"]].set(0.03)           # Zeiss is slow to scale
cap0 = cap0.at[T_IDX["Wear_EUV"]].set(0.15)           # tools wear ~15%/period
cap0 = cap0.at[T_IDX["Refurb"]].set(0.2)
cap0 = cap0.at[T_IDX["Recycle"]].set(0.05)            # near-dead loop closure

x0 = jnp.full(NP_, 1.0).at[P["EUV_tools"]].set(1.0)

if __name__ == "__main__":
    base, _ = simulate(cap0, x0, 0.0)
    hit,  _ = simulate(cap0, x0, 0.9)
    print(f"chip throughput, no disruption:      {base:8.2f}")
    print(f"chip throughput, 90% EUV destroyed:  {hit:8.2f}   "
          f"({100*(1-hit/base):.0f}% civilizational haircut)\n")

    g_cap = jax.grad(throughput, argnums=0)(cap0, x0)
    g_stk = jax.grad(throughput, argnums=1)(cap0, x0)

    print("d(throughput)/d(capacity) — where to invest in CAPACITY:")
    for i in jnp.argsort(-g_cap):
        print(f"   {float(g_cap[i]):8.3f}  {TRANSITIONS[int(i)][0]}")

    print("\nd(throughput)/d(initial stock) — where to invest in STOCKPILES:")
    for i in jnp.argsort(-g_stk):
        print(f"   {float(g_stk[i]):8.3f}  {PLACES[int(i)]}")
