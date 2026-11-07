import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { analyse, latinProfile, stateText, ROUTING_CONTRACT_VERSION } from "../src/lang.js";
import { Router } from "../src/router.js";
import { TABLE_SHA256, VERSION } from "../src/routing-unicode.js";
import { routingSnapshot } from "../../tests/routing_parity_support.mjs";

const corpus = JSON.parse(readFileSync(new URL("../../tests/fixtures/routing-parity-v1.json", import.meta.url), "utf8"));

describe("shared routing contract", () => {
  it("identifies the same contract and frozen Unicode semantics as Python", () => {
    expect(ROUTING_CONTRACT_VERSION).toBe(corpus.contract);
    expect(VERSION).toBe(corpus.unicode_version);
    expect(TABLE_SHA256).toBe(corpus.unicode_table_sha256);
  });

  it.each(corpus.cases)("matches all recorded evidence and route fields: $id", (row) => {
    expect(routingSnapshot(row, { analyse, latinProfile, stateText, Router })).toEqual(row.expected);
  });

  it("keeps explanation opt-in and forwards it through route batches", () => {
    const state = { note: "Please review the invoice and tell me what happened. ".repeat(100),
      customer: "😀".repeat(4001) + "\nQuero cancelar meu plano agora mesmo" };
    expect(analyse(state)).not.toHaveProperty("diagnostic");
    expect(analyse(state, { explain: true }).diagnostic).toEqual({
      stage: "value", reason: "named_value", source: { leafIndex: 1, path: ["customer"], line: 2 },
    });
    const router = new Router();
    expect(router.routeBatch([{ state, questions: {} }], { explain: true })[0])
      .toEqual(router.route(state, {}, { explain: true }));
    expect(router.loaded).toEqual([]);
  });

  it("does not read values beyond the state-text budget", () => {
    const state = Object.defineProperty({ first: "hello" }, "later", {
      enumerable: true, get: () => { throw new Error("unread value"); },
    });
    expect(stateText(state, 2)).toBe("he");
  });

  it("keeps BOM model aliases invalid while trimming Python control whitespace", () => {
    const router = new Router();
    expect(() => router.route("hello", {}, { model: "\ufeffen" })).toThrow();
    expect(() => router.route("hello", {}, { task: "en\ufeff" })).toThrow();
    expect(router.route("hello", {}, { model: "\x1cen\x1c" }).model).toBe("english");
  });
});
