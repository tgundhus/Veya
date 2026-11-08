import { describe, expect, it } from "vitest";
import { analyse, latinProfile, stateText } from "../src/lang.js";
import { Router } from "../src/router.js";

// Expectations recorded from Python's veya.lang.state_text/analyse on these exact inputs.
// Python budgets/counts code points; JS string.length/slice count UTF-16 code units.
const PORTUGUESE = "Quero o meu dinheiro de volta agora mesmo porque já esperei demais";
const NOTE = "Please check the attached invoice and let me know what happened. ";

describe("Unicode language evidence matches Python", () => {
  it("counts emoji once in the diacritic denominator at the routing boundary", () => {
    const state = "é😀😀" + "-".repeat(46);
    expect(latinProfile(state).diacriticRate).toBe(1 / 49);
    expect(analyse(state)).toMatchObject({
      script: "latin", language: null, isEnglish: false,
      languageUndecided: true, diacriticRate: 0.0204,
    });
    expect(new Router().route(state, {}).model).toBe("multilingual");
  });

  it("counts lowercasing expansions and astral code points separately", () => {
    // Python lower() yields é, i, combining dot, emoji: four code points.
    expect(latinProfile("éİ😀").diacriticRate).toBe(0.25);
  });

  it("reads prose after emoji padding within the 4000-code-point state budget", () => {
    const state = "😀".repeat(3500) + "\n" + PORTUGUESE;
    expect(analyse(state)).toMatchObject({
      script: "latin", language: "pt", isEnglish: false,
      languageUndecided: false, diacriticRate: 0.0003, mixedSegment: null,
    });
    expect(new Router().route(state, {}).model).toBe("multilingual");
  });

  it("does not charge a previous line's emoji twice against the mixed-segment budget", () => {
    const state = NOTE.repeat(15) + "😀".repeat(2600) + "\n" + PORTUGUESE;
    expect(analyse(state)).toMatchObject({
      script: "latin", language: "pt", isEnglish: false,
      languageUndecided: false, diacriticRate: 0.0003, mixedSegment: PORTUGUESE,
    });
  });

  it("uses the same code-point budget when sampling a buried non-English value", () => {
    const state = { note: NOTE.repeat(100), customer: "😀".repeat(3500) + PORTUGUESE };
    expect(analyse(state)).toMatchObject({
      script: "latin", language: "pt", isEnglish: false,
      languageUndecided: false, diacriticRate: 0, mixedSegment: null,
    });
  });
});

describe("Unicode state budgets match Python", () => {
  it.each([
    [-5, ""], [-1, ""], [0, ""], [1, "😀"], [2, "😀"],
    [3, "😀 é"], [4, "😀 é"], [5, "😀 é X"], [6, "😀 é X"],
  ] as const)("keeps separators and complete characters at budget %d", (budget, expected) => {
    expect(stateText(["😀", "é", "X"], budget)).toBe(expected);
  });

  it("never produces an orphan surrogate when the default boundary meets an emoji", () => {
    const state = "A".repeat(3999) + "😀" + "tail";
    expect(stateText(state)).toBe("A".repeat(3999) + "😀");
  });

  it("preserves combining marks as individual code points rather than grapheme clusters", () => {
    expect(stateText("e\u0301😀", 1)).toBe("e");
    expect(stateText("e\u0301😀", 2)).toBe("e\u0301");
    expect(stateText("e\u0301😀", 3)).toBe("e\u0301😀");
  });

  it("decodes UTF-8 byte leaves before applying the character budget", () => {
    expect(stateText(new TextEncoder().encode("😀éX"), 2)).toBe("😀é");
    expect(stateText(new Uint8Array([0xff, 0xfe]), 2)).toBe("");
  });

  it("does not read a later field after a partial prefix has exhausted the budget", () => {
    const state = {
      first: "😀".repeat(100_000),
      get later(): string { throw new Error("unread field"); },
    };
    expect(stateText(state, 2)).toBe("😀😀");
  });

  it("does not read a later field after an exact leaf consumes the remaining budget", () => {
    const state = {
      first: "😀",
      get later(): string { throw new Error("unread field"); },
    };
    expect(stateText(state, 1)).toBe("😀");
    expect(stateText(state, 2)).toBe("😀");
  });

  it("does not flatten any field at a zero or negative budget", () => {
    const state = { get first(): string { throw new Error("unread state"); } };
    expect(stateText(state, 0)).toBe("");
    expect(stateText(state, -1)).toBe("");
  });
});
