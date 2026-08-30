/**
 * Money formatting is the thing a reader trusts most and checks least, so the
 * boundaries between k / M / B get their own cases — an off-by-one there turns
 * €1,000,000 into "€1000k" on a card nobody re-reads.
 */
import { describe, expect, it } from "vitest";
import { eur, eurExact, featureLabel, number, percent, shortDate } from "./format";

describe("eur", () => {
  it.each([
    [0, "€0"],
    [999, "€999"],
    [1_000, "€1k"],
    [999_499, "€999k"],
    // Promoted rather than rendered "€1000k": the smaller unit can no longer
    // hold the number in three digits.
    [999_999, "€1.00M"],
    [1_000_000, "€1.00M"],
    [999_994_999, "€999.99M"],
    [999_995_000, "€1.00B"],
    [29_799_038, "€29.80M"],
    [1_000_000_000, "€1.00B"],
  ])("formats %d as %s", (value, expected) => {
    expect(eur(value)).toBe(expected);
  });

  it("abbreviates negatives by magnitude, keeping the sign", () => {
    expect(eur(-2_500_000)).toBe("€-2.50M");
  });

  it.each([null, undefined, NaN])("renders %s as an em dash, never as 0", (value) => {
    // A missing value shown as "€0" is a claim; an em dash is an absence.
    expect(eur(value as number | null | undefined)).toBe("—");
  });
});

describe("eurExact", () => {
  it("groups thousands and drops the cents", () => {
    expect(eurExact(29_799_038)).toBe("€29,799,038");
  });

  it.each([null, undefined, NaN])("renders %s as an em dash", (value) => {
    expect(eurExact(value as number | null | undefined)).toBe("—");
  });
});

describe("percent", () => {
  it("scales by 100 and defaults to one decimal", () => {
    expect(percent(0.441)).toBe("44.1%");
  });

  it("honours a requested precision", () => {
    expect(percent(0.4410002, 3)).toBe("44.100%");
  });

  it("renders a missing value as an em dash", () => {
    expect(percent(null)).toBe("—");
  });
});

describe("number", () => {
  it("groups thousands", () => {
    expect(number(36_880)).toBe("36,880");
  });

  it("renders a missing value as an em dash", () => {
    expect(number(undefined)).toBe("—");
  });
});

describe("featureLabel", () => {
  it("strips the ColumnTransformer prefix the model adds", () => {
    expect(featureLabel("numeric__goals_per_90")).toBe("goals per 90");
    expect(featureLabel("categorical__position_Attack")).toBe("position Attack");
  });

  it("uppercases EUR, which is a currency and not a word", () => {
    expect(featureLabel("numeric__prev_log_market_value_in_eur")).toContain("EUR");
  });

  it("leaves an already-clean name alone", () => {
    expect(featureLabel("age")).toBe("age");
  });
});

describe("shortDate", () => {
  it("keeps the date and drops the time", () => {
    expect(shortDate("2026-08-29T09:07:53.371592+00:00")).toBe("2026-08-29");
  });

  it.each([null, undefined, "not a date"])("renders %s as an em dash", (value) => {
    expect(shortDate(value as string | null | undefined)).toBe("—");
  });
});
