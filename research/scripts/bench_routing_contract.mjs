// Optional native-SDK half of bench_routing_contract.py; inputs and raw measurements are JSON.
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import * as after from "../../veya-ts/dist/lang.js";
import { expandFixture } from "../../tests/routing_parity_support.mjs";

const input = JSON.parse(readFileSync(0, "utf8"));
const source = stripTypeScriptTypes(input.source, { mode: "strip" });
const before = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
const rows = [];
for (const workload of input.workloads) {
  const state = expandFixture(workload.state);
  const operations = {};
  for (const operation of ["stateText", "analyse"]) {
    const prior = before[operation](state);
    const current = after[operation](state);
    const samples = { before: [], after: [] };
    for (let i = 0; i < 20; i++) { before[operation](state); after[operation](state); }
    for (let repeat = 0; repeat < input.repeats; repeat++) {
      for (const mode of repeat % 2 ? ["after", "before"] : ["before", "after"]) {
        const fn = mode === "before" ? before[operation] : after[operation];
        const start = process.hrtime.bigint();
        for (let i = 0; i < workload.iterations; i++) fn(state);
        samples[mode].push(Number(process.hrtime.bigint() - start) / workload.iterations / 1000);
      }
    }
    const median = (values) => {
      const sorted = values.toSorted((a, b) => a - b);
      const index = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[index] : (sorted[index - 1] + sorted[index]) / 2;
    };
    const medians = { before: median(samples.before), after: median(samples.after) };
    operations[operation] = { mean_call_samples_us: samples, median_us: medians,
      after_over_before: medians.after / medians.before,
      outputs_equal: JSON.stringify(prior) === JSON.stringify(current) };
  }
  rows.push({ name: workload.name, iterations: workload.iterations, operations });
}
process.stdout.write(JSON.stringify({ node: process.version, unicode: process.versions.unicode, cases: rows }));
