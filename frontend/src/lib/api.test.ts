/**
 * The client's job is to turn every backend failure into an ApiError carrying
 * the server's own `code`, so a page can branch on "player_not_found" instead
 * of matching on a message. That mapping is the part worth testing: it is the
 * only place the two error shapes (the documented envelope, and a dead socket)
 * become one type.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, API_BASE } from "./api";

function mockFetch(
  response: Partial<Response> & { json?: () => Promise<unknown> },
) {
  const spy = vi.fn().mockResolvedValue({ ok: true, status: 200, ...response });
  vi.stubGlobal("fetch", spy);
  return spy;
}

afterEach(() => vi.unstubAllGlobals());

describe("request", () => {
  it("returns the parsed body on success", async () => {
    mockFetch({ json: async () => ({ status: "ok", ready: true }) });
    await expect(api.health()).resolves.toEqual({ status: "ok", ready: true });
  });

  it("carries the server's code so a caller can branch on it", async () => {
    mockFetch({
      ok: false,
      status: 404,
      json: async () => ({
        error: {
          code: "player_not_found",
          message: "no player with id 9",
          detail: null,
        },
      }),
    });

    await expect(api.player(9)).rejects.toMatchObject({
      code: "player_not_found",
      message: "no player with id 9",
      status: 404,
    });
  });

  it("keeps the field-level detail from a 422", async () => {
    mockFetch({
      ok: false,
      status: 422,
      json: async () => ({
        error: {
          code: "validation_error",
          message:
            "invalid feature value(s): goals: cannot be negative, got -1",
          detail: ["goals: cannot be negative, got -1"],
        },
      }),
    });

    const error = await api
      .predictFromFeatures({ goals: -1 })
      .catch((caught) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe("validation_error");
    // "which field, and why" is the entire value of a 422.
    expect(error.detail).toEqual(["goals: cannot be negative, got -1"]);
  });

  it("survives an error body that is not the envelope", async () => {
    // A proxy or a load balancer can answer before the app does.
    mockFetch({
      ok: false,
      status: 502,
      json: async () => {
        throw new SyntaxError("Unexpected token < in JSON");
      },
    });

    await expect(api.models()).rejects.toMatchObject({
      code: "unknown_error",
      status: 502,
    });
  });

  it("turns a dead backend into an actionable message, not 'Failed to fetch'", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );
    const error = await api.health().catch((caught) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe("network_error");
    expect(error.status).toBe(0);
    // Names the command that fixes it — the common failure in development.
    expect(error.message).toContain("make serve");
    expect(error.message).toContain(API_BASE);
  });
});

describe("url construction", () => {
  it("encodes a query so a name with a space or a slash survives", async () => {
    const spy = mockFetch({ json: async () => ({ query: "", results: [] }) });
    await api.searchPlayers("De Bruyne / K");
    expect(spy.mock.calls[0][0]).toContain("q=De%20Bruyne%20%2F%20K");
  });

  it("omits the variant entirely rather than sending 'undefined'", async () => {
    const spy = mockFetch({ json: async () => ({ points: [] }) });
    await api.predictionHistory(42);
    expect(spy.mock.calls[0][0]).toBe(`${API_BASE}/api/v1/players/42/history`);
  });

  it("includes the variant when one is given", async () => {
    const spy = mockFetch({ json: async () => ({ points: [] }) });
    await api.predictionHistory(42, "with_prior_value");
    expect(spy.mock.calls[0][0]).toContain("?variant=with_prior_value");
  });

  it("sends exactly one of player_id or features, never both", async () => {
    const spy = mockFetch({ json: async () => ({ prediction_eur: 1 }) });
    await api.predictForPlayer(8198);
    const body = JSON.parse(spy.mock.calls[0][1].body);
    expect(body.player_id).toBe(8198);
    expect(body.features).toBeUndefined();

    await api.predictFromFeatures({ age: 25 });
    const second = JSON.parse(spy.mock.calls[1][1].body);
    expect(second.features).toEqual({ age: 25 });
    expect(second.player_id).toBeUndefined();
  });

  it("never caches, so a prediction is not served from a stale response", async () => {
    const spy = mockFetch({ json: async () => ({}) });
    await api.models();
    expect(spy.mock.calls[0][1].cache).toBe("no-store");
  });

  it("hits the unversioned health path, which orchestrators depend on", async () => {
    const spy = mockFetch({ json: async () => ({}) });
    await api.health();
    expect(spy.mock.calls[0][0]).toBe(`${API_BASE}/health`);
  });
});
