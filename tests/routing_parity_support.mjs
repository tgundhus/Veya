// Shared fixture shorthand only; production routing accepts ordinary states, never this DSL.
export function expandFixture(value) {
  if (Array.isArray(value)) return value.map(expandFixture);
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    if (keys.length === 1 && keys[0] === "$repeat") return value.$repeat[0].repeat(value.$repeat[1]);
    if (keys.length === 1 && keys[0] === "$concat") return value.$concat.map(expandFixture).join("");
    if (keys.length === 1 && keys[0] === "$bytes") return new Uint8Array(value.$bytes);
    if (keys.length === 1 && keys[0] === "$pairs") {
      return Object.fromEntries(value.$pairs.map(([key, item]) => [key, expandFixture(item)]));
    }
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, expandFixture(item)]));
  }
  return value;
}

export function routingSnapshot(row, { analyse, latinProfile, stateText, Router }) {
  const state = expandFixture(row.state);
  const router = new Router(row.router ?? {});
  return {
    text: Object.fromEntries([-1, 0, 1, 2, 7, 31, 4000].map((budget) => [String(budget), stateText(state, budget)])),
    analysis: analyse(state),
    explained: analyse(state, { explain: true }),
    profile: latinProfile(stateText(state)),
    route: router.route(state, row.questions ?? {}, { ...row.options, explain: true }),
  };
}
