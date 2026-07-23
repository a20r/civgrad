/*
 * site/test/regression.mjs — the JS port must match the Python model.
 *
 * Runs the historical events under the v1 adaptive protocol using
 * site/js/model.js and asserts dip/recovery match the v1 rows of
 * SCORECARD.md (frozen in each event yaml's baseline_v0_1.adaptive,
 * exported to site/data/events.json) within 1% — i.e. one percentage
 * point of dip and one month of recovery; an infinite recovery must be
 * infinite in both. If JS and Python disagree, fix JS (the Python model
 * plus validation/baseline_outputs.txt is the authority).
 *
 * Node, no dependencies:  node site/test/regression.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { runEvent, ALPHA } from "../js/model.js";

const here = dirname(fileURLToPath(import.meta.url));
const events = JSON.parse(readFileSync(join(here, "..", "data", "events.json"), "utf8"));

const DIP_TOL_PP = 1.0;    // percentage points
const REC_TOL_MO = 1.0;    // months

let failures = 0;
let checked = 0;

for (const ev of events) {
  if (ev.status === "out-of-scope") continue;
  const expected = ev.baseline_v0_1?.adaptive;
  if (!expected) { console.error(`SKIP ${ev.name}: no baseline_v0_1.adaptive`); failures++; continue; }

  const res = runEvent({
    transition: ev.transition,
    kill: ev.capacity_lost,
    buffers: ev.buffers,
    alpha: ALPHA,
    observe: ev.observable === "delivered" ? "delivered" : "fab",
    t0: ev.shock_month,
  });

  const expDip = expected.dip_pct;
  const expRec = expected.recovery_months === "inf" ? Infinity : expected.recovery_months;

  const dipOk = Math.abs(res.dipPct - expDip) <= DIP_TOL_PP;
  const recOk = expRec === Infinity
    ? res.recoveryMonths === Infinity
    : (res.recoveryMonths !== Infinity && Math.abs(res.recoveryMonths - expRec) <= REC_TOL_MO);

  const fmt = (v) => (v === Infinity ? "never" : v.toFixed(1));
  const verdict = dipOk && recOk ? "ok  " : "FAIL";
  console.log(`${verdict} ${ev.name.padEnd(18)} dip ${res.dipPct.toFixed(2).padStart(7)}% ` +
              `(py ${expDip}%)  rec ${fmt(res.recoveryMonths).padStart(5)} mo (py ${fmt(expRec)})`);
  if (!(dipOk && recOk)) failures++;
  checked++;
}

if (checked === 0) { console.error("no events checked"); process.exit(1); }
if (failures) {
  console.error(`\n${failures} regression(s): the JS port disagrees with the scorecard. Fix JS.`);
  process.exit(1);
}
console.log(`\nJS port matches the v1 scorecard rows on all ${checked} events (±${DIP_TOL_PP}pp dip, ±${REC_TOL_MO}mo recovery).`);
