import { describe, expect, it } from "vitest";
import { readPair, writePair } from "./comparePair";

describe("readPair", () => {
  it("reads both ids", () => {
    expect(readPair("?a=937958&b=433177")).toEqual({ a: 937958, b: 433177 });
  });

  it("reads one side when only one is present", () => {
    expect(readPair("?a=937958")).toEqual({ a: 937958, b: null });
  });

  it("returns nulls for an empty query", () => {
    expect(readPair("")).toEqual({ a: null, b: null });
  });

  it("ignores anything that is not a player id", () => {
    // A stale or hand-edited link must leave an empty picker rather than send
    // the API a request for player NaN.
    for (const bad of ["abc", "", "-1", "0", "1.5", "1e9999"]) {
      expect(readPair(`?a=${bad}`).a).toBeNull();
    }
  });
});

describe("writePair", () => {
  it("writes both ids", () => {
    expect(writePair("/compare", 1, 2)).toBe("/compare?a=1&b=2");
  });

  it("drops the side that was cleared", () => {
    // The bug this guards: leaving a stale id behind means the next reload
    // resurrects a player the user just removed.
    expect(writePair("/compare", 1, null)).toBe("/compare?a=1");
    expect(writePair("/compare", null, 2)).toBe("/compare?b=2");
  });

  it("returns a bare path when nothing is chosen", () => {
    expect(writePair("/compare", null, null)).toBe("/compare");
    expect(writePair("/compare", undefined, undefined)).toBe("/compare");
  });

  it("round-trips through readPair", () => {
    const url = writePair("/compare", 937958, 433177);
    expect(readPair(url.slice(url.indexOf("?")))).toEqual({
      a: 937958,
      b: 433177,
    });
  });
});
