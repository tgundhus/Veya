import { readFileSync } from "node:fs";
import { analyse, latinProfile, stateText } from "../veya-ts/dist/lang.js";
import { Router } from "../veya-ts/dist/router.js";
import { routingSnapshot } from "./routing_parity_support.mjs";

const rows = JSON.parse(readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(rows.map((row) => routingSnapshot(row, { analyse, latinProfile, stateText, Router }))));
