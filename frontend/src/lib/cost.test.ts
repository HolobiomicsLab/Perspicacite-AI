import { describe, expect, it } from "vitest";

import { estimateCostUsd, formatUsd } from "./cost";

describe("estimateCostUsd", () => {
  it("returns null when the model is missing", () => {
    expect(estimateCostUsd(null, 1000, 1000)).toBeNull();
    expect(estimateCostUsd(undefined, 1000, 1000)).toBeNull();
  });

  it("returns null for an unknown model", () => {
    expect(estimateCostUsd("some/unknown-model", 1000, 1000)).toBeNull();
  });

  it("prices a known model by per-million rates", () => {
    // deepseek-chat: $0.14 in / $0.28 out per million tokens.
    const cost = estimateCostUsd("deepseek-chat", 1_000_000, 1_000_000);
    expect(cost).toBeCloseTo(0.14 + 0.28, 10);
  });

  it("matches against provider-prefixed model strings", () => {
    const cost = estimateCostUsd(
      "openrouter/deepseek/deepseek-chat",
      2_000_000,
      0,
    );
    expect(cost).toBeCloseTo(0.28, 10);
  });

  it("prefers the longest matching key", () => {
    // "deepseek-reasoner" must win over the shorter "deepseek-chat" when
    // both could appear; here the reasoner rates ($0.55/$2.19) apply.
    const cost = estimateCostUsd("deepseek-reasoner", 1_000_000, 1_000_000);
    expect(cost).toBeCloseTo(0.55 + 2.19, 10);
  });
});

describe("formatUsd", () => {
  it("returns an empty string for null/undefined", () => {
    expect(formatUsd(null)).toBe("");
    expect(formatUsd(undefined)).toBe("");
  });

  it("floors tiny amounts to a sentinel", () => {
    expect(formatUsd(0.00001)).toBe("<$0.0001");
  });

  it("uses 4 decimals below a cent", () => {
    expect(formatUsd(0.0025)).toBe("$0.0025");
  });

  it("uses 3 decimals below a dime", () => {
    expect(formatUsd(0.05)).toBe("$0.050");
  });

  it("uses 2 decimals at or above a dime", () => {
    expect(formatUsd(1.2345)).toBe("$1.23");
  });
});
