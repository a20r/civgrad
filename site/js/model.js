/*
 * site/js/model.js — the v1 adaptive continuous Petri net, ported to JS.
 *
 * Direct port of core/continuous.py (flows, Michaelis-Menten saturation,
 * Euler integration, burn_in) and core/adaptation.py (scarcity signals,
 * restoration gating, frozen calibration). Constants are transcribed EXACTLY
 * from core/ — do not tune here; site/test/regression.mjs asserts this port
 * matches the v1 rows of SCORECARD.md within 1%.
 *
 * Plain ES module, float64 (JS numbers), no dependencies; runs in the
 * browser and in node.
 */

// ---------------- net definition (mirrors core/continuous.py) ----------------

export const PLACES = ["Ga_byproduct", "Ga_refined", "Ne_crude", "Ne_purified",
                       "EUV_tools", "EUV_optics", "EUV_worn",
                       "Wafers", "Chips", "Pkg", "Goods", "E_waste"];
export const P = Object.fromEntries(PLACES.map((n, i) => [n, i]));
const NP = PLACES.length;

// [name, inputs{place:w}, outputs{place:w}, reads[place]]
export const TRANSITIONS = [
  ["Mine",        {},                                          {Ga_byproduct: 1, Ne_crude: 1}, []],
  ["WaferSupply", {},                                          {Wafers: 1},                    []],
  ["OpticsMfg",   {},                                          {EUV_optics: 1},                []],
  ["Refine_Ga",   {Ga_byproduct: 1},                           {Ga_refined: 1},                []],
  ["Purify_Ne",   {Ne_crude: 1},                               {Ne_purified: 1},               []],
  ["Fab",         {Ga_refined: 1, Ne_purified: 1, Wafers: 1},  {Chips: 1},                     ["EUV_tools"]],
  ["Package",     {Chips: 1},                                  {Pkg: 1},                       []],
  ["Ship_Strait", {Pkg: 1},                                    {Goods: 1},                     []],
  ["Consume",     {Goods: 1},                                  {E_waste: 1},                   []],
  ["Recycle",     {E_waste: 1},                                {Ga_byproduct: 1},              []],
  ["Build_EUV",   {Pkg: 1, EUV_optics: 1},                     {EUV_tools: 1},                 []],
  ["Wear_EUV",    {EUV_tools: 1},                              {EUV_worn: 1},                  []],
  ["Refurb",      {EUV_worn: 1, Pkg: 1},                       {EUV_tools: 1},                 []],
];
const NT = TRANSITIONS.length;
export const T_IDX = Object.fromEntries(TRANSITIONS.map((t, i) => [t[0], i]));

const Pre  = TRANSITIONS.map(t => { const r = new Float64Array(NP); for (const [p, w] of Object.entries(t[1])) r[P[p]] = w; return r; });
const Post = TRANSITIONS.map(t => { const r = new Float64Array(NP); for (const [p, w] of Object.entries(t[2])) r[P[p]] = w; return r; });
const READS = TRANSITIONS.map(t => t[3].map(p => P[p]));
const INPUTS = TRANSITIONS.map((t, i) => Object.keys(t[1]).map(p => P[p]));
const OUTPUTS = TRANSITIONS.map((t, i) => Object.keys(t[2]).map(p => P[p]));

// ---------------- frozen globals (transcribed from core/) ----------------

export const K_SAT = 0.05;        // JIT cliff — full flow on thin inventory
export const DT = 0.1;            // 1.0 = one month
export const HORIZON = 6.0;       // planning horizon (months of runway)
export const ALPHA = 0.06;        // fitted on neon 2022, then FROZEN
export const UTILIZATION = 0.90;  // pre-shock calibration rule

// cap0 — baseline relative capacities (core/continuous.py)
export const cap0 = new Float64Array(NT).fill(1.0);
cap0[T_IDX.Build_EUV] = 0.03;     // tools are slow to build
cap0[T_IDX.OpticsMfg] = 0.03;     // Zeiss is slow to scale
cap0[T_IDX.Wear_EUV]  = 0.15;     // tools wear ~15%/period
cap0[T_IDX.Refurb]    = 0.2;
cap0[T_IDX.Recycle]   = 0.05;     // near-dead loop closure

// ---------------- dynamics ----------------

export function flows(x, cap) {
  const sat = new Float64Array(NP);
  for (let p = 0; p < NP; p++) sat[p] = x[p] / (x[p] + K_SAT);
  const v = new Float64Array(NT);
  for (let i = 0; i < NT; i++) {
    let lim = 1.0;                              // source transitions
    if (INPUTS[i].length) {
      lim = Infinity;
      for (const p of INPUTS[i]) if (sat[p] < lim) lim = sat[p];
    }
    let rd = 1.0;
    for (const p of READS[i]) rd *= sat[p];
    v[i] = cap[i] * lim * rd;
  }
  return v;
}

function eulerStep(x, v, dt) {
  const nx = new Float64Array(NP);
  for (let p = 0; p < NP; p++) {
    let d = 0.0;
    for (let i = 0; i < NT; i++) d += (Post[i][p] - Pre[i][p]) * v[i];
    const val = x[p] + dt * d;
    nx[p] = val > 0.0 ? val : 0.0;
  }
  return nx;
}

export function burnIn(cap, x0, months = 48, dt = 0.1) {
  let x = Float64Array.from(x0);
  const steps = Math.trunc(months / dt);
  for (let k = 0; k < steps; k++) x = eulerStep(x, flows(x, cap), dt);
  return [x, flows(x, cap)[T_IDX.Fab]];
}

// ---------------- frozen calibration (identical rule to core/adaptation.py) ----------------

function calibrate() {
  let c = Float64Array.from(cap0);
  let x = new Float64Array(NP).fill(1.0);
  let F = 0.0, xs;
  for (let r = 0; r < 2; r++) {
    [xs, F] = burnIn(c, x);
    for (const tn of ["Purify_Ne", "WaferSupply", "Refine_Ga"]) c[T_IDX[tn]] = F / UTILIZATION;
    x = xs;
  }
  let X_REF;
  [X_REF, F] = burnIn(c, x);
  const X_JIT = Float64Array.from(X_REF);
  for (const pn of ["Ne_purified", "Wafers", "Ga_refined", "Chips", "Pkg"]) X_JIT[P[pn]] = 0.5 * F;
  return { c, F, X_JIT, X_REF: X_JIT };   // reference = normal JIT operating stocks
}

export const CAL = calibrate();

// ---------------- adaptive simulation (core/adaptation.py::simulate) ----------------

export function simulateAdaptive(capBase, x0, xRef, tidx, kill, alpha,
                                 t0 = 6.0, months = 72) {
  const vRef = flows(xRef, capBase);            // pre-crisis throughput anchor
  let x = Float64Array.from(x0);
  let c = Float64Array.from(capBase);
  const steps = Math.trunc(months / DT);
  const fab = new Float64Array(steps), ship = new Float64Array(steps);
  const stocks = [];                            // sampled monthly, for the UI
  for (let k = 0; k < steps; k++) {
    const t = k * DT;
    if (Math.abs(t - t0) < DT / 2) for (let i = 0; i < NT; i++) c[i] *= (1.0 - kill * (i === tidx ? 1.0 : 0.0));
    const v0 = flows(x, c);
    const inflow = new Float64Array(NP), outflow = new Float64Array(NP);
    for (let i = 0; i < NT; i++) for (let p = 0; p < NP; p++) {
      inflow[p] += Post[i][p] * v0[i];
      outflow[p] += Pre[i][p] * v0[i];
    }
    const scar = new Float64Array(NP);
    for (let p = 0; p < NP; p++) {
      const drain = outflow[p] - inflow[p];
      const netDrain = drain > 1e-9 ? drain : 1e-9;
      const runway = x[p] / netDrain;
      let s = 1.0 - runway / HORIZON;
      s = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
      scar[p] = drain > 1e-6 ? s : 0.0;
    }
    for (let i = 0; i < NT; i++) {
      let runwayScar = 0.0;
      for (const p of OUTPUTS[i]) if (scar[p] > runwayScar) runwayScar = scar[p];
      let flowScar = (vRef[i] - v0[i]) / (vRef[i] + 1e-9) / 0.10;   // saturates at 10% shortfall
      flowScar = flowScar < 0.0 ? 0.0 : (flowScar > 1.0 ? 1.0 : flowScar);
      let restoreGap = (capBase[i] - c[i]) / (1e-6 * capBase[i] + 1e-9);
      restoreGap = restoreGap < 0.0 ? 0.0 : (restoreGap > 1.0 ? 1.0 : restoreGap);
      const prodScar = Math.max(runwayScar, flowScar * restoreGap);
      c[i] += DT * alpha * capBase[i] * prodScar;
    }
    const v = flows(x, c);
    x = eulerStep(x, v, DT);
    fab[k] = v[T_IDX.Fab];
    ship[k] = v[T_IDX.Ship_Strait];
    if (k % 10 === 0) stocks.push(Float64Array.from(x));
  }
  return { fab, ship, stocks };
}

// ---------------- dip/recovery metric (core/adaptation.py::dip_recovery) ----------------

export function dipRecovery(traj, t0 = 6.0) {
  const i0 = Math.trunc(t0 / DT);
  const w = Math.trunc(3 / DT);
  let pre = 0.0;
  for (let k = i0 - w; k < i0; k++) pre += traj[k];
  pre /= w;
  const post = traj.subarray(i0);
  let imin = 0;
  for (let k = 1; k < post.length; k++) if (post[k] < post[imin]) imin = k;
  const dip = 1.0 - post[imin] / pre;
  let maxAfter = -Infinity;
  for (let k = imin; k < post.length; k++) if (post[k] > maxAfter) maxAfter = post[k];
  let rec = Infinity;
  if (maxAfter > 0.95 * pre) {
    for (let k = imin; k < post.length; k++) {
      if (post[k] > 0.95 * pre) { rec = k * DT; break; }
    }
  }
  return { dip, rec, pre };
}

// ---------------- one-call event replay (v1 adaptive protocol) ----------------

/**
 * Replay a disruption under the frozen adaptation law.
 * opts: { transition, kill, buffers: {place: months}, alpha, observe:
 *         "fab"|"delivered", t0, months }
 * Returns dip/recovery of the observed flow plus both trajectories.
 */
export function runEvent(opts) {
  const { transition, kill, buffers = {}, alpha = ALPHA,
          observe = "fab", t0 = 6.0, months = 72 } = opts;
  const x0 = Float64Array.from(CAL.X_JIT);
  for (const [place, m] of Object.entries(buffers)) x0[P[place]] = m * CAL.F;
  const { fab, ship, stocks } = simulateAdaptive(
    CAL.c, x0, CAL.X_REF, T_IDX[transition], kill, alpha, t0, months);
  const observed = observe === "fab" ? fab : ship;
  const { dip, rec, pre } = dipRecovery(observed, t0);
  return { dipPct: 100 * dip, recoveryMonths: rec, pre,
           fab, ship, stocks, observed: observe };
}
