/*
 * site/js/app.js — the civgrad page: essay + live model + scorecard mirror.
 * No framework, no build step. Data comes from site/data/*.json, exported
 * from the repo's yamls (single source of truth) by
 * validation/export_site_data.py; the simulator is site/js/model.js,
 * CI-checked against the Python scorecard.
 */

import { runEvent, ALPHA, DT, TRANSITIONS, T_IDX } from "./model.js";

const $ = (sel) => document.querySelector(sel);
const SVG = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, text) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
};

// ------------------------------------------------------------ markdown ----

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function sanitizeUrl(url) {
  // Allowlist: http(s), mailto, and relative/anchor URLs. Anything with
  // another scheme (javascript:, data:, ...) is dropped; quotes are encoded
  // so the href can never break out of its attribute.
  const u = url.replace(/"/g, "%22").replace(/'/g, "%27");
  if (/^(https?:|mailto:)/i.test(u) || !/^[a-z][a-z0-9+.-]*:/i.test(u)) return u;
  return null;
}

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, text, url) => {
      const safe = sanitizeUrl(url);
      return safe === null ? text : `<a href="${safe}">${text}</a>`;
    });
}

function renderMarkdown(md) {
  const out = [];
  const lines = escapeHtml(md).split("\n");
  let i = 0;
  const isList = (l) => /^\s*[-*] /.test(l) || /^\s*\d+\. /.test(l);
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (/^#{1,4} /.test(line)) {
      const level = line.match(/^#+/)[0].length;
      out.push(`<h${level}>${inline(line.replace(/^#+ /, ""))}</h${level}>`);
      i++;
    } else if (/^---+\s*$/.test(line)) {
      out.push("<hr>"); i++;
    } else if (/^&gt; ?/.test(line)) {
      const q = [];
      while (i < lines.length && /^&gt; ?/.test(lines[i])) q.push(lines[i].replace(/^&gt; ?/, "")), i++;
      out.push(`<blockquote><p>${inline(q.join(" "))}</p></blockquote>`);
    } else if (isList(line)) {
      const ordered = /^\s*\d+\. /.test(line);
      const items = [];
      while (i < lines.length && isList(lines[i]))
        items.push(inline(lines[i].replace(/^\s*([-*]|\d+\.) /, ""))), i++;
      out.push(`<${ordered ? "ol" : "ul"}>${items.map((x) => `<li>${x}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
    } else {
      const p = [];
      while (i < lines.length && lines[i].trim() && !/^#{1,4} |^---+\s*$|^&gt; ?/.test(lines[i]) && !isList(lines[i]))
        p.push(lines[i]), i++;
      out.push(`<p>${inline(p.join(" "))}</p>`);
    }
  }
  return out.join("\n");
}

// ---------------------------------------------------------- page assembly ----
// The essay's source of truth is the blog post (it embeds this same demo and
// scorecard); this page hosts the model, live, and points there for the prose.

const ESSAY_URL = "https://a20r.github.io/blog/posts/the-civilization-gradient/";

function mountPage() {
  const essay = $("#essay");
  const demo = document.getElementById("demo-template").content.firstElementChild;
  const scorecard = document.getElementById("scorecard-template").content.firstElementChild;
  const note = document.createElement("p");
  note.className = "essay-pending";
  note.innerHTML =
    "This page is the model, live. The essay — the project's main " +
    "deliverable, with the full audit and policy tables — lives on " +
    `<a href="${ESSAY_URL}">the blog</a>; everything shown here is generated ` +
    'from <a href="https://github.com/a20r/civgrad">the repository</a>.';
  essay.appendChild(note);
  essay.appendChild(demo);
  essay.appendChild(scorecard);
}

// ------------------------------------------------------------- diagram ----

// Hand layout. Transitions are squares, places circles, oracles fog.
const POS = {
  // fogged oracle frontier
  Mine:        [120, 62],  WaferSupply: [340, 62],  OpticsMfg: [640, 62],
  // places
  Ga_byproduct: [70, 168], Ne_crude: [170, 168], Wafers: [340, 168], EUV_optics: [640, 168],
  Ga_refined:   [70, 332], Ne_purified: [170, 332],
  EUV_tools:   [640, 332], EUV_worn: [862, 332],
  Chips: [475, 428], Pkg: [612, 428], Goods: [762, 428], E_waste: [900, 428],
  // transitions
  Refine_Ga: [70, 252], Purify_Ne: [170, 252],
  Fab: [400, 428], Package: [545, 428], Ship_Strait: [688, 428],
  Consume: [832, 428], Recycle: [948, 332],
  Build_EUV: [560, 252], Wear_EUV: [752, 332], Refurb: [752, 252],
  // ghosts
  packaging_resin: [478, 300], fertilizer: [930, 62],
};
const ORACLE_OF = { Mine: "mining", WaferSupply: "wafer_supply", OpticsMfg: "optics" };
const TRANS_LABEL_ABOVE = new Set(["Fab", "Package", "Ship_Strait", "Consume"]);

function arc(from, to, opts = {}) {
  const [x1, y1] = POS[from], [x2, y2] = POS[to];
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
  const trim = 18, ex = x1 + (dx * (len - trim)) / len, ey = y1 + (dy * (len - trim)) / len;
  const sx = x1 + (dx * trim) / len, sy = y1 + (dy * trim) / len;
  const a = el("path", { d: `M${sx},${sy} L${ex},${ey}`, class: `arc ${opts.cls || ""}` });
  return a;
}

function buildDiagram(oracles, onPick) {
  const svg = $("#net");
  const oracleByName = Object.fromEntries(oracles.map((o) => [o.name, o]));

  const defs = el("defs");
  defs.innerHTML =
    `<marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
       <path d="M0,0.5 L7.5,4 L0,7.5 Z" fill="currentColor" opacity="0.75"/>
     </marker>
     <filter id="fogblur" x="-40%" y="-40%" width="180%" height="180%">
       <feGaussianBlur stdDeviation="2.1"/>
     </filter>
     <pattern id="hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
       <rect width="6" height="6" fill="transparent"/>
       <line x1="0" y1="0" x2="0" y2="6" stroke="currentColor" stroke-opacity="0.35" stroke-width="2"/>
     </pattern>`;
  svg.appendChild(defs);

  svg.appendChild(el("text", { x: 40, y: 28, class: "lane-label" }, "oracle frontier — fog: unmapped, wrong in unknown directions"));
  svg.appendChild(el("text", { x: 40, y: 480, class: "lane-label" }, "material flow →"));

  // arcs (drawn first, under nodes)
  const arcs = [
    ["Mine", "Ga_byproduct"], ["Mine", "Ne_crude"],
    ["WaferSupply", "Wafers"], ["OpticsMfg", "EUV_optics"],
    ["Ga_byproduct", "Refine_Ga"], ["Refine_Ga", "Ga_refined"], ["Ga_refined", "Fab"],
    ["Ne_crude", "Purify_Ne"], ["Purify_Ne", "Ne_purified"], ["Ne_purified", "Fab"],
    ["Wafers", "Fab"],
    ["Fab", "Chips"], ["Chips", "Package"], ["Package", "Pkg"], ["Pkg", "Ship_Strait"],
    ["Ship_Strait", "Goods"], ["Goods", "Consume"], ["Consume", "E_waste"],
    ["E_waste", "Recycle"],
    ["EUV_optics", "Build_EUV"], ["Pkg", "Build_EUV"], ["Build_EUV", "EUV_tools"],
    ["EUV_tools", "Wear_EUV"], ["Wear_EUV", "EUV_worn"], ["EUV_worn", "Refurb"],
    ["Refurb", "EUV_tools"], ["Pkg", "Refurb"],
  ];
  for (const [a, b] of arcs) svg.appendChild(arc(a, b));
  svg.appendChild(el("path", {
    d: "M948,314 C 948,120 500,10 88,152", class: "arc", "stroke-dasharray": "1 3",
  })); // Recycle -> Ga_byproduct: the long way home
  const read = el("path", { d: "M624,344 C 540,392 470,412 418,424", class: "arc read" });
  svg.appendChild(read); // EUV_tools ⇢ Fab (read arc: enables, not consumed)
  svg.appendChild(el("path", { d: "M478,318 C 500,360 525,395 538,414", class: "arc ghost" })); // resin ghost

  const tooltip = $("#tooltip");
  const show = (evt, html) => {
    tooltip.innerHTML = html;
    tooltip.hidden = false;
    const wrap = svg.parentElement.getBoundingClientRect();
    tooltip.style.left = Math.min(evt.clientX - wrap.left + 14, wrap.width - 320) + "px";
    tooltip.style.top = evt.clientY - wrap.top + 14 + "px";
  };
  const hide = () => (tooltip.hidden = true);

  const contTrans = new Set(Object.keys(T_IDX));
  for (const [name, [x, y]] of Object.entries(POS)) {
    const isOracleSource = name in ORACLE_OF;
    const isGhost = name === "packaging_resin" || name === "fertilizer";
    const isTrans = contTrans.has(name) && !isOracleSource;

    let g;
    if (isOracleSource || isGhost) {
      g = el("g", { class: "fog", "data-name": name });
      g.appendChild(name === "fertilizer" || name === "packaging_resin"
        ? el("circle", { cx: x, cy: y, r: 15 })
        : el("rect", { x: x - 15, y: y - 13, width: 30, height: 26 }));
      g.appendChild(el("text", { x, y: y - 21, "text-anchor": "middle" }, name));
      const o = oracleByName[ORACLE_OF[name] || name];
      const tip = o
        ? `<span class="t-name">${o.name}</span> — unexpanded oracle<br>${escapeHtml(String(o.contract))}`
        : name;
      g.addEventListener("pointermove", (evt) => show(evt, tip));
      g.addEventListener("pointerleave", hide);
      if (isOracleSource) {
        g.style.cursor = "pointer";
        g.addEventListener("click", () => onPick(name));
      }
    } else if (isTrans) {
      g = el("g", { class: "trans", "data-name": name });
      g.appendChild(el("rect", { x: x - 9, y: y - 9, width: 18, height: 18 }));
      const above = TRANS_LABEL_ABOVE.has(name);
      g.appendChild(el("text", { x, y: above ? y - 16 : y + 26, "text-anchor": "middle" }, name));
      g.addEventListener("pointermove", (evt) =>
        show(evt, `<span class="t-name">${name}</span> — process · click to disrupt`));
      g.addEventListener("pointerleave", hide);
      g.addEventListener("click", () => onPick(name));
    } else {
      g = el("g", { class: "place", "data-name": name });
      g.appendChild(el("circle", { cx: x, cy: y, r: 13 }));
      g.appendChild(el("text", { x, y: y - 20, "text-anchor": "middle" }, name));
      g.addEventListener("pointermove", (evt) =>
        show(evt, `<span class="t-name">${name}</span> — stock`));
      g.addEventListener("pointerleave", hide);
    }
    svg.appendChild(g);
  }
}

function markSelected(name) {
  document.querySelectorAll("#net .trans, #net .fog").forEach((g) =>
    g.classList.toggle("selected", g.dataset.name === name));
}

// --------------------------------------------------------------- charts ----

function drawChart(fab, ship, t0) {
  const svg = $("#chart");
  svg.innerHTML = "";
  const W = 1000, H = 320, L = 54, R = 16, T = 16, B = 34;
  const months = fab.length * DT;

  const preOf = (traj) => {
    const i0 = Math.trunc(t0 / DT), w = Math.trunc(3 / DT);
    let s = 0; for (let k = i0 - w; k < i0; k++) s += traj[k];
    return s / w;
  };
  const preF = preOf(fab), preS = preOf(ship);
  let ymax = 1.15;
  for (let k = 0; k < fab.length; k++) {
    ymax = Math.max(ymax, fab[k] / preF, ship[k] / preS);
  }
  const X = (m) => L + ((W - L - R) * m) / months;
  const Y = (r) => T + (H - T - B) * (1 - r / ymax);

  svg.appendChild(el("line", { x1: L, y1: Y(0), x2: W - R, y2: Y(0), class: "axis" }));
  svg.appendChild(el("line", { x1: L, y1: T, x2: L, y2: Y(0), class: "axis" }));
  for (const m of [0, 12, 24, 36, 48, 60, 72]) {
    svg.appendChild(el("text", { x: X(m), y: H - 12, "text-anchor": "middle", class: "gridlabel" }, `${m}mo`));
  }
  for (const r of [0.5, 1.0]) {
    svg.appendChild(el("line", { x1: L, y1: Y(r), x2: W - R, y2: Y(r), class: "ref" }));
    svg.appendChild(el("text", { x: L - 8, y: Y(r) + 4, "text-anchor": "end", class: "gridlabel" }, `${Math.round(r * 100)}%`));
  }
  svg.appendChild(el("line", { x1: L, y1: Y(0.95), x2: W - R, y2: Y(0.95), class: "thresh" }));
  svg.appendChild(el("text", { x: W - R, y: Y(0.95) - 5, "text-anchor": "end", class: "gridlabel" }, "95% recovery threshold"));
  svg.appendChild(el("line", { x1: X(t0), y1: T, x2: X(t0), y2: Y(0), class: "shock" }));
  svg.appendChild(el("text", { x: X(t0) + 4, y: T + 10, class: "gridlabel" }, "shock"));

  const path = (traj, pre, cls) => {
    let d = "";
    for (let k = 0; k < traj.length; k += 2)
      d += `${k ? "L" : "M"}${X(k * DT).toFixed(1)},${Y(traj[k] / pre).toFixed(1)} `;
    svg.appendChild(el("path", { d, class: cls }));
  };
  path(ship, preS, "ship");
  path(fab, preF, "fab");
}

// ------------------------------------------------------------ scorecard ----

function fmtRec(v) {
  return v === "inf" ? "never (72 mo)" : `${Number(v).toFixed(1)} mo`;
}
function statusCell(status) {
  if (status === "pass") return `<span class="status pass">✅ pass</span>`;
  if (status === "xfail") return `<span class="status xfail">❌ xfail (documented miss)</span>`;
  return `<span class="status oos">⛔ out-of-scope (no demand dynamics)</span>`;
}

function buildScorecard(events) {
  const table = $("#scorecard-table");
  const rows = [
    `<tr><th>event</th><th>protocol</th><th>model dip</th><th>model recovery</th><th>historical record</th><th>status</th></tr>`,
  ];
  for (const ev of events) {
    if (ev.status === "out-of-scope") {
      rows.push(`<tr><td>${escapeHtml(ev.title)}</td><td>—</td><td class="num">—</td><td class="num">—</td>
        <td>${escapeHtml(ev.history)}</td><td>${statusCell(ev.status)}</td></tr>`);
      continue;
    }
    const b = ev.baseline_v0_1 || {};
    for (const [proto, key] of [["v0 imposed-τ", "tau_replay"], ["v1 adaptive", "adaptive"]]) {
      const r = b[key];
      if (!r) continue;
      rows.push(`<tr><td>${escapeHtml(ev.title)}</td><td>${proto}</td>
        <td class="num">${Number(r.dip_pct).toFixed(1)}%</td>
        <td class="num">${fmtRec(r.recovery_months)}</td>
        <td>${escapeHtml(ev.history)}</td><td>${statusCell(ev.status)}</td></tr>`);
    }
  }
  table.innerHTML = rows.join("\n");
  const xf = events.find((e) => e.status === "xfail");
  if (xf?.xfail_reason) $("#xfail-reason").textContent = xf.xfail_reason;
  else $("#xfail-details").hidden = true;
}

// ------------------------------------------------------------- controls ----

const state = {
  target: "Purify_Ne", kill: 0.5, bufferPlace: "Ne_purified", bufferMonths: 6.0,
  alpha: ALPHA, observe: "fab", t0: 6.0, preset: null,
};

function outputPlaceOf(trans) {
  const t = TRANSITIONS[T_IDX[trans]];
  const outs = Object.keys(t[2]);
  return outs[0] || "Ne_purified";
}

function syncControls() {
  $("#target").value = state.target;
  $("#severity").value = Math.round(state.kill * 100);
  $("#severity-val").textContent = `${Math.round(state.kill * 100)}%`;
  $("#buffer").value = state.bufferMonths;
  $("#buffer-val").textContent = `${state.bufferMonths} mo`;
  $("#buffer-place").textContent = `buffer applies to ${state.bufferPlace}`;
  $("#alpha").value = state.alpha;
  $("#alpha-val").textContent = state.alpha.toFixed(3);
  document.querySelectorAll("#presets button").forEach((b) =>
    b.classList.toggle("active", b.dataset.name === state.preset));
  markSelected(state.target);
}

function inBand(dipPct, rec, expected) {
  const tol = expected.tolerance;
  const okDip = Math.abs(dipPct - expected.dip_pct) <= tol.dip_pct;
  const okRec = rec !== Infinity && Math.abs(rec - expected.recovery_months) <= tol.recovery_months;
  return okDip && okRec;
}

function run(events) {
  const res = runEvent({
    transition: state.target, kill: state.kill,
    buffers: { [state.bufferPlace]: state.bufferMonths },
    alpha: state.alpha, observe: state.observe, t0: state.t0,
  });
  drawChart(res.fab, res.ship, state.t0);
  $("#ro-observe").textContent = state.observe === "fab" ? "fab flow" : "delivered flow";
  const dipEl = $("#ro-dip");
  dipEl.textContent = `${res.dipPct.toFixed(1)}%`;
  dipEl.className = "v " + (res.dipPct > 20 ? "bad" : res.dipPct < 5 ? "good" : "");
  const recEl = $("#ro-rec");
  recEl.textContent = res.recoveryMonths === Infinity ? "never (72 mo horizon)"
    : res.dipPct < 5 ? "n/a (never below 95%)" : `${res.recoveryMonths.toFixed(1)} mo`;
  recEl.className = "v " + (res.recoveryMonths === Infinity ? "bad" : "");

  const ev = events.find((e) => e.name === state.preset);
  const bandWrap = $("#ro-band-wrap");
  if (ev?.expected) {
    bandWrap.hidden = false;
    const ok = inBand(res.dipPct, res.recoveryMonths, ev.expected);
    const bandEl = $("#ro-band");
    bandEl.textContent = ok ? "inside band ✓" : ev.status === "xfail" ? "outside band — the documented miss" : "outside band ✗";
    bandEl.className = "v " + (ok ? "good" : "bad");
    $("#history-line").textContent = `history: ${ev.history}` +
      (state.alpha !== ALPHA ? "  ·  (alpha moved off the frozen value — no longer the scored replay)" : "");
  } else {
    bandWrap.hidden = true;
    $("#history-line").textContent = "custom scenario — no historical anchor; the scorecard below holds the scored replays.";
  }
}

function applyPreset(ev, events) {
  state.preset = ev.name;
  state.target = ev.transition;
  state.kill = ev.capacity_lost;
  const [bp, bm] = Object.entries(ev.buffers)[0];
  state.bufferPlace = bp; state.bufferMonths = bm;
  state.alpha = ALPHA;
  state.observe = ev.observable === "delivered" ? "delivered" : "fab";
  state.t0 = ev.shock_month;
  syncControls();
  run(events);
}

function buildControls(events) {
  const sel = $("#target");
  for (const t of TRANSITIONS) {
    const o = document.createElement("option");
    o.value = t[0]; o.textContent = t[0];
    sel.appendChild(o);
  }
  const presets = $("#presets");
  for (const ev of events) {
    if (ev.status === "out-of-scope") continue;
    const b = document.createElement("button");
    b.dataset.name = ev.name;
    b.textContent = ev.short;
    b.title = `history: ${ev.history}`;
    b.addEventListener("click", () => applyPreset(ev, events));
    presets.appendChild(b);
  }
  const custom = () => { state.preset = null; state.observe = "fab"; };
  sel.addEventListener("input", () => {
    custom(); state.target = sel.value;
    state.bufferPlace = outputPlaceOf(state.target);
    syncControls(); run(events);
  });
  $("#severity").addEventListener("input", (e) => {
    custom(); state.kill = e.target.value / 100; syncControls(); run(events);
  });
  $("#buffer").addEventListener("input", (e) => {
    if (state.preset) { /* keep the preset's buffer place, just resize */ }
    else state.bufferPlace = state.bufferPlace || outputPlaceOf(state.target);
    state.preset = null;
    state.bufferMonths = Number(e.target.value); syncControls(); run(events);
  });
  $("#alpha").addEventListener("input", (e) => {
    state.alpha = Number(e.target.value); syncControls(); run(events);
  });
}

// ----------------------------------------------------------------- boot ----

async function boot() {
  const [events, oracles] = await Promise.all([
    fetch("data/events.json").then((r) => r.json()),
    fetch("data/oracles.json").then((r) => r.json()),
  ]);
  mountPage();
  buildDiagram(oracles, (name) => {
    state.preset = null; state.target = name; state.observe = "fab";
    state.bufferPlace = outputPlaceOf(name);
    syncControls(); run(events);
  });
  buildControls(events);
  buildScorecard(events);
  const neon = events.find((e) => e.role === "training") || events[0];
  applyPreset(neon, events);
}

boot().catch((err) => {
  $("#essay").innerHTML = `<p class="essay-pending">Failed to load page data (${escapeHtml(String(err))}).
  The repository is at <a href="https://github.com/a20r/civgrad">github.com/a20r/civgrad</a>.</p>`;
});
