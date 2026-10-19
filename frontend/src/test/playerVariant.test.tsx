/**
 * Switching model variant on the player page.
 *
 * The page's data — player record, baseline prediction and the what-if seed —
 * is fetched per variant. Switching used to leave the previous variant's data
 * on screen as if current while the new request was pending, and the what-if
 * panel, keyed on the *new* variant, remounted at once with the *old* seed: a
 * slider moved in that window sent performance-only features to the
 * prior-value model, without the prior inputs it requires. Every response
 * here is a deferred promise, so the order is fixed and nothing depends on
 * timing.
 */
import { Suspense } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Player, PredictResponse, Variant } from "@/lib/api";

vi.mock("@/components/Chart", () => ({ default: () => <div data-testid="chart" /> }));
vi.mock("@/components/ModelMetadata", () => ({ default: () => null }));
vi.mock("@/components/SimilarPlayers", () => ({ default: () => null }));
vi.mock("@/components/PredictionHistory", () => ({ default: () => null }));
vi.mock("@/components/ValueHistory", () => ({ default: () => null }));

const mocked = vi.hoisted(() => ({
  player: vi.fn(),
  predictForPlayer: vi.fn(),
  predictFromFeatures: vi.fn(),
  similarPlayers: vi.fn(),
  modelInfo: vi.fn(),
  modelMetrics: vi.fn(),
  predictionHistory: vi.fn(),
}));

vi.mock("@/lib/api", async (original) => ({
  ...(await original<typeof import("@/lib/api")>()),
  api: mocked,
}));

import PlayerPage from "@/app/players/[id]/page";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

const PRIOR = "prev_log_market_value_in_eur";

function player(variant: Variant): Player {
  // The API seeds the what-if panel with exactly the features the variant's
  // model takes; only the prior-value model takes the prior.
  const features: Record<string, unknown> = { goals: 10, age: 24 };
  if (variant === "with_prior_value") features[PRIOR] = 16.1;
  return {
    player_id: 7,
    name: "Player 7",
    club: null,
    league: null,
    league_country: null,
    position: "Attack",
    sub_position: null,
    foot: null,
    height_in_cm: null,
    country_of_citizenship: null,
    seasons: [],
    features,
  } as unknown as Player;
}

function prediction(variant: Variant, eur: number): PredictResponse {
  return {
    prediction_eur: eur,
    variant,
    model: { name: "lightgbm", variant, trained_at: "" },
    player_id: 7,
    season: 2025,
    confidence: null,
    explanation: null,
  };
}

const BASELINE: Record<Variant, number> = {
  performance_only: 1_000_000,
  with_prior_value: 2_000_000,
};

beforeEach(() => {
  for (const fn of Object.values(mocked)) fn.mockReset();
  mocked.predictForPlayer.mockImplementation((_id: number, variant: Variant) =>
    Promise.resolve(prediction(variant, BASELINE[variant])),
  );
  mocked.similarPlayers.mockResolvedValue({ player_id: 7, season: null, results: [] });
  mocked.modelInfo.mockResolvedValue({});
  mocked.modelMetrics.mockResolvedValue({});
  mocked.predictionHistory.mockResolvedValue({ points: [] });
  mocked.predictFromFeatures.mockImplementation(
    (_features: Record<string, unknown>, variant: Variant) =>
      Promise.resolve(prediction(variant, 3_000_000)),
  );
});

async function renderPage() {
  // The page unwraps its route params with `use`, which suspends; rendering
  // inside an async act lets React settle that before the test looks.
  const params = Promise.resolve({ id: "7" });
  await act(async () => {
    render(
      <Suspense fallback={null}>
        <PlayerPage params={params} />
      </Suspense>,
    );
  });
  await screen.findByLabelText(/Goals/);
}

const slider = () => screen.queryByLabelText(/Goals/);
const choose = (label: string) => fireEvent.click(screen.getByRole("button", { name: label }));

/** Every what-if request must carry the inputs its own variant's model takes. */
function expectNoMismatchedRequest() {
  for (const [features, variant] of mocked.predictFromFeatures.mock.calls) {
    expect(variant === "with_prior_value" ? PRIOR in features : !(PRIOR in features)).toBe(true);
  }
}

describe("switching model variant", () => {
  it("hides the what-if controls until the selected variant's data has arrived", async () => {
    const prior = deferred<Player>();
    mocked.player.mockImplementation((_id: number, variant: Variant) =>
      variant === "with_prior_value" ? prior.promise : Promise.resolve(player(variant)),
    );
    await renderPage();

    choose("With prior value");
    // The old variant's seed must not be offered for the new variant.
    expect(slider()).not.toBeInTheDocument();
    expect(screen.queryByText("€1.00M")).not.toBeInTheDocument();

    await act(async () => prior.resolve(player("with_prior_value")));
    fireEvent.change(await screen.findByLabelText(/Goals/), { target: { value: "20" } });
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(1));

    const [features, variant] = mocked.predictFromFeatures.mock.calls[0];
    expect(variant).toBe("with_prior_value");
    expect(features).toMatchObject({ goals: 20, [PRIOR]: 16.1 });
  });

  it("never sends the old seed with the new variant when a slider moves mid-switch", async () => {
    const prior = deferred<Player>();
    mocked.player.mockImplementation((_id: number, variant: Variant) =>
      variant === "with_prior_value" ? prior.promise : Promise.resolve(player(variant)),
    );
    await renderPage();

    choose("With prior value");
    // What a user could do in the old window: the slider was still there.
    const stale = slider();
    if (stale) fireEvent.change(stale, { target: { value: "25" } });
    // Longer than the 300 ms debounce, so a request would have gone out.
    await act(() => new Promise((done) => setTimeout(done, 400)));

    expectNoMismatchedRequest();
    await act(async () => prior.resolve(player("with_prior_value")));
    await screen.findByLabelText(/Goals/);
    expectNoMismatchedRequest();
  });

  it("keeps the latest choice when an earlier variant's data lands last", async () => {
    const prior = deferred<Player>();
    const back = deferred<Player>();
    let performanceCalls = 0;
    mocked.player.mockImplementation((_id: number, variant: Variant) => {
      if (variant === "with_prior_value") return prior.promise;
      performanceCalls += 1;
      return performanceCalls === 1 ? Promise.resolve(player(variant)) : back.promise;
    });
    await renderPage();

    choose("With prior value");
    choose("Performance only");
    await act(async () => back.resolve(player("performance_only")));
    await screen.findByLabelText(/Goals/);
    // The abandoned prior-value response arrives after the current one.
    await act(async () => prior.resolve(player("with_prior_value")));

    expect(screen.getByRole("button", { name: "Performance only" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.queryByText("€2.00M")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/Goals/), { target: { value: "15" } });
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(1));
    expect(mocked.predictFromFeatures.mock.calls[0][1]).toBe("performance_only");
    expectNoMismatchedRequest();
  });

  it("resets to the selected variant's own baseline after a switch", async () => {
    mocked.player.mockImplementation((_id: number, variant: Variant) =>
      Promise.resolve(player(variant)),
    );
    await renderPage();

    choose("With prior value");
    fireEvent.change(await screen.findByLabelText(/Goals/), { target: { value: "30" } });
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Reset to actual season" }));

    expect(screen.getByLabelText(/Goals/)).toHaveValue("10");
    // The baseline the panel returns to is the prior-value model's, €2.00M,
    // shown in the prediction panel and as the what-if's resting value.
    expect(screen.getAllByText("€2.00M").length).toBeGreaterThanOrEqual(2);
    expectNoMismatchedRequest();
  });
});
