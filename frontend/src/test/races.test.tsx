/**
 * Audit F13: out-of-order responses.
 *
 * Every test resolves the *earlier* request last, which is the order that used
 * to put a stale answer on screen: search results for "Ha" under the query
 * "Haaland", one player's valuation under another's name, a what-if
 * prediction for slider values the user had already moved past. The order is
 * fixed by deferred promises, so nothing here depends on timing.
 */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Player, PredictResponse, SearchResult } from "@/lib/api";

vi.mock("@/components/Chart", () => ({ default: () => <div data-testid="chart" /> }));
vi.mock("@/components/Radar", () => ({ default: () => <div data-testid="radar" /> }));

const mocked = vi.hoisted(() => ({
  searchPlayers: vi.fn(),
  predictForPlayer: vi.fn(),
  predictFromFeatures: vi.fn(),
  player: vi.fn(),
  similarPlayers: vi.fn(),
  featureDistribution: vi.fn(),
}));

vi.mock("@/lib/api", async (original) => ({
  ...(await original<typeof import("@/lib/api")>()),
  api: mocked,
}));

import PlayerSearch from "@/app/players/page";
import ComparePage from "@/app/compare/page";
import WhatIf from "@/components/WhatIf";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function hit(id: number, name: string): SearchResult {
  return {
    player_id: id,
    name,
    club: null,
    league: null,
    league_country: null,
    position: "Attack",
    latest_season: 2025,
    market_value_in_eur: null,
    predictable: true,
  };
}

function prediction(eur: number, season = 2025): PredictResponse {
  return {
    prediction_eur: eur,
    variant: "performance_only",
    model: { name: "lightgbm", variant: "performance_only", trained_at: "" },
    player_id: null,
    season,
    confidence: null,
    explanation: null,
  };
}

function player(id: number): Player {
  return {
    player_id: id,
    name: `Player ${id}`,
    club: null,
    league: null,
    league_country: null,
    position: "Attack",
    sub_position: null,
    foot: null,
    height_in_cm: null,
    country_of_citizenship: null,
    seasons: [],
    features: {},
  } as Player;
}

beforeEach(() => {
  // The compare page writes its pair into the URL and restores it on mount;
  // jsdom keeps the URL between tests, so each starts from a clean one.
  window.history.replaceState(null, "", "/");
  for (const fn of Object.values(mocked)) fn.mockReset();
  mocked.featureDistribution.mockResolvedValue({ variant: "performance_only", grid: [], features: {} });
  mocked.similarPlayers.mockResolvedValue({ player_id: 0, season: null, results: [] });
});

describe("player search", () => {
  it("shows the answer to the latest query, not the slowest", async () => {
    const short = deferred<{ query: string; results: SearchResult[] }>();
    const long = deferred<{ query: string; results: SearchResult[] }>();
    mocked.searchPlayers.mockImplementation((term: string) =>
      term === "Ha" ? short.promise : long.promise,
    );
    render(<PlayerSearch />);
    const box = screen.getByLabelText("Player name");

    fireEvent.change(box, { target: { value: "Ha" } });
    await waitFor(() => expect(mocked.searchPlayers).toHaveBeenCalledWith("Ha"));
    fireEvent.change(box, { target: { value: "Haaland" } });
    await waitFor(() => expect(mocked.searchPlayers).toHaveBeenCalledWith("Haaland"));

    await act(async () => long.resolve({ query: "Haaland", results: [hit(1, "Erling Haaland")] }));
    await act(async () => short.resolve({ query: "Ha", results: [hit(2, "Achraf Hakimi")] }));

    expect(await screen.findByText("Erling Haaland")).toBeInTheDocument();
    expect(screen.queryByText("Achraf Hakimi")).not.toBeInTheDocument();
  });

  it("ignores a failure of a query it has moved on from", async () => {
    const short = deferred<{ query: string; results: SearchResult[] }>();
    mocked.searchPlayers.mockImplementation((term: string) =>
      term === "Ha" ? short.promise : Promise.resolve({ query: term, results: [hit(1, "Erling Haaland")] }),
    );
    render(<PlayerSearch />);
    const box = screen.getByLabelText("Player name");

    fireEvent.change(box, { target: { value: "Ha" } });
    await waitFor(() => expect(mocked.searchPlayers).toHaveBeenCalledWith("Ha"));
    fireEvent.change(box, { target: { value: "Haaland" } });
    expect(await screen.findByText("Erling Haaland")).toBeInTheDocument();

    await act(async () => short.reject(new Error("timeout")));
    expect(screen.getByText("Erling Haaland")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("stays empty when the box is cleared while a request is in flight", async () => {
    const pending = deferred<{ query: string; results: SearchResult[] }>();
    mocked.searchPlayers.mockReturnValue(pending.promise);
    render(<PlayerSearch />);
    const box = screen.getByLabelText("Player name");

    fireEvent.change(box, { target: { value: "Ha" } });
    await waitFor(() => expect(mocked.searchPlayers).toHaveBeenCalled());
    fireEvent.change(box, { target: { value: "" } });
    await new Promise((done) => setTimeout(done, 300)); // past the debounce
    await act(async () => pending.resolve({ query: "Ha", results: [hit(2, "Achraf Hakimi")] }));

    expect(screen.queryByText("Achraf Hakimi")).not.toBeInTheDocument();
  });
});

// Re-audit: a response landing *inside* the debounce window, before the next
// request has been sent. Invalidation used to wait for that request to take a
// ticket, so the stale answer still counted as current.
describe("responses landing before the next debounced request", () => {
  it("search: drops the old query's answer as soon as the box changes", async () => {
    const short = deferred<{ query: string; results: SearchResult[] }>();
    const long = deferred<{ query: string; results: SearchResult[] }>();
    mocked.searchPlayers.mockImplementation((term: string) =>
      term === "Ha" ? short.promise : long.promise,
    );
    render(<PlayerSearch />);
    const box = screen.getByLabelText("Player name");

    fireEvent.change(box, { target: { value: "Ha" } });
    await waitFor(() => expect(mocked.searchPlayers).toHaveBeenCalledWith("Ha"));
    fireEvent.change(box, { target: { value: "Haaland" } });
    // Inside the 250ms window: "Haaland" has not been requested yet.
    await act(async () => short.resolve({ query: "Ha", results: [hit(2, "Achraf Hakimi")] }));
    expect(mocked.searchPlayers).not.toHaveBeenCalledWith("Haaland");
    expect(screen.queryByText("Achraf Hakimi")).not.toBeInTheDocument();

    await waitFor(() => expect(mocked.searchPlayers).toHaveBeenCalledWith("Haaland"));
    await act(async () => long.resolve({ query: "Haaland", results: [hit(1, "Erling Haaland")] }));
    expect(await screen.findByText("Erling Haaland")).toBeInTheDocument();
    expect(screen.queryByText("Achraf Hakimi")).not.toBeInTheDocument();
  });

  it("search: a box cleared mid-request stays empty even if the answer lands at once", async () => {
    const pending = deferred<{ query: string; results: SearchResult[] }>();
    mocked.searchPlayers.mockReturnValue(pending.promise);
    render(<PlayerSearch />);
    const box = screen.getByLabelText("Player name");

    fireEvent.change(box, { target: { value: "Ha" } });
    await waitFor(() => expect(mocked.searchPlayers).toHaveBeenCalled());
    fireEvent.change(box, { target: { value: "" } });
    await act(async () => pending.resolve({ query: "Ha", results: [hit(2, "Achraf Hakimi")] }));

    expect(screen.queryByText("Achraf Hakimi")).not.toBeInTheDocument();
  });

  it("what-if: drops a prediction for slider values already moved past", async () => {
    const first = deferred<PredictResponse>();
    const second = deferred<PredictResponse>();
    mocked.predictFromFeatures
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    render(<WhatIf baseline={1_000_000} features={{ goals: 10 }} variant="performance_only" />);
    const slide = (value: number) =>
      fireEvent.change(screen.getByLabelText(/Goals/), { target: { value: String(value) } });

    slide(11);
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(1));
    slide(30);
    // Inside the 300ms window: goals=30 has not been requested yet.
    await act(async () => first.resolve(prediction(1_100_000)));
    expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("€1.10M")).not.toBeInTheDocument();

    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(2));
    await act(async () => second.resolve(prediction(5_000_000)));
    expect(screen.getByText("€5.00M")).toBeInTheDocument();
  });

  it("what-if: Reset abandons the request in flight, its answer and its error", async () => {
    const first = deferred<PredictResponse>();
    const third = deferred<PredictResponse>();
    mocked.predictFromFeatures
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(third.promise);
    render(<WhatIf baseline={1_000_000} features={{ goals: 10 }} variant="performance_only" />);
    const slide = (value: number) =>
      fireEvent.change(screen.getByLabelText(/Goals/), { target: { value: String(value) } });

    slide(11);
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Reset to actual season" }));
    // No request follows a Reset, so nothing else would ever supersede it.
    await act(async () => first.resolve(prediction(1_100_000)));
    expect(screen.getByText("Actual season")).toBeInTheDocument();
    expect(screen.queryByText("predicting…")).not.toBeInTheDocument();

    // Moving again must not resurrect the abandoned answer while the new
    // request is pending.
    slide(12);
    expect(screen.queryByText("€1.10M")).not.toBeInTheDocument();
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(2));
    expect(screen.queryByText("€1.10M")).not.toBeInTheDocument();
    await act(async () => third.resolve(prediction(1_200_000)));
    expect(screen.getByText("€1.20M")).toBeInTheDocument();
  });

  it("what-if: an error from a request abandoned by Reset is not shown", async () => {
    const first = deferred<PredictResponse>();
    mocked.predictFromFeatures.mockReturnValueOnce(first.promise);
    render(<WhatIf baseline={1_000_000} features={{ goals: 10 }} variant="performance_only" />);

    fireEvent.change(screen.getByLabelText(/Goals/), { target: { value: "11" } });
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Reset to actual season" }));
    await act(async () => first.reject(new Error("timeout")));

    expect(screen.queryByText("What-if prediction failed")).not.toBeInTheDocument();
    expect(screen.queryByText("predicting…")).not.toBeInTheDocument();
  });

  it("compare: a new selection drops the side's search still in flight", async () => {
    const stale = deferred<{ query: string; results: SearchResult[] }>();
    mocked.searchPlayers.mockImplementation((term: string) =>
      term === "old" ? stale.promise : Promise.resolve({ query: term, results: [] }),
    );
    render(<ComparePage />);
    const a = screen.getByLabelText("Player A");
    const picker = within(a.parentElement as HTMLElement);

    fireEvent.change(a, { target: { value: "old" } });
    await waitFor(() => expect(mocked.searchPlayers).toHaveBeenCalledWith("old", 8));
    fireEvent.change(a, { target: { value: "new" } });
    await act(async () => stale.resolve({ query: "old", results: [hit(9, "Stale Result")] }));

    expect(picker.queryByRole("button", { name: /Stale Result/ })).not.toBeInTheDocument();
    await waitFor(() => expect(mocked.searchPlayers).toHaveBeenCalledWith("new", 8));
    expect(picker.queryByRole("button", { name: /Stale Result/ })).not.toBeInTheDocument();
  });
});

describe("compare", () => {
  it("keeps a side's numbers with the player it names after a quick re-pick", async () => {
    const slow = deferred<PredictResponse>();
    mocked.searchPlayers.mockResolvedValue({
      query: "p",
      results: [hit(1, "First Choice"), hit(2, "Second Choice"), hit(3, "Other Side")],
    });
    mocked.player.mockImplementation((id: number) => Promise.resolve(player(id)));
    mocked.predictForPlayer.mockImplementation((id: number) =>
      id === 1 ? slow.promise : Promise.resolve(prediction(id * 1_000_000)),
    );
    render(<ComparePage />);

    const a = screen.getByLabelText("Player A");
    fireEvent.change(a, { target: { value: "p" } });
    fireEvent.click(await screen.findByRole("button", { name: /First Choice/ }));
    fireEvent.change(a, { target: { value: "p" } });
    fireEvent.click(await screen.findByRole("button", { name: /Second Choice/ }));

    const b = screen.getByLabelText("Player B");
    fireEvent.change(b, { target: { value: "p" } });
    fireEvent.click(await screen.findByRole("button", { name: /Other Side/ }));

    await screen.findByText("€2.00M");
    await act(async () => slow.resolve(prediction(1_000_000)));

    expect(screen.getByText("€2.00M")).toBeInTheDocument();
    expect(screen.getByText("€3.00M")).toBeInTheDocument();
    expect(screen.queryByText("€1.00M")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Player A")).toHaveValue("Second Choice");
  });

  it("drops a pick still in flight when its side is retyped", async () => {
    const slow = deferred<PredictResponse>();
    mocked.searchPlayers.mockResolvedValue({
      query: "p",
      results: [hit(1, "First Choice"), hit(3, "Other Side")],
    });
    mocked.player.mockImplementation((id: number) => Promise.resolve(player(id)));
    mocked.predictForPlayer.mockImplementation((id: number) =>
      id === 1 ? slow.promise : Promise.resolve(prediction(id * 1_000_000)),
    );
    render(<ComparePage />);

    const a = screen.getByLabelText("Player A");
    fireEvent.change(a, { target: { value: "p" } });
    fireEvent.click(await screen.findByRole("button", { name: /First Choice/ }));
    fireEvent.change(a, { target: { value: "someone else" } });

    // Side B complete, so a stale side A landing would render a comparison.
    const b = screen.getByLabelText("Player B");
    fireEvent.change(b, { target: { value: "p" } });
    // Scoped to B's picker: retyping A gave A a result list of its own.
    const picker = within(b.parentElement as HTMLElement);
    fireEvent.click(await picker.findByRole("button", { name: /Other Side/ }));
    await waitFor(() => expect(mocked.predictForPlayer).toHaveBeenCalledTimes(2));
    await act(async () => slow.resolve(prediction(1_000_000)));

    expect(screen.getByLabelText("Player A")).toHaveValue("someone else");
    expect(screen.queryByText("€1.00M")).not.toBeInTheDocument();
    expect(screen.getByText("Choose two players to compare.")).toBeInTheDocument();
  });
});

describe("compare across seasons (F14)", () => {
  async function compare(seasonA: number, seasonB: number) {
    mocked.searchPlayers.mockResolvedValue({
      query: "p",
      results: [hit(1, "Retired Player"), hit(2, "Active Player")],
    });
    mocked.player.mockImplementation((id: number) => Promise.resolve(player(id)));
    mocked.predictForPlayer.mockImplementation((id: number) =>
      Promise.resolve(prediction(id * 1_000_000, id === 1 ? seasonA : seasonB)),
    );
    render(<ComparePage />);
    for (const [label, name] of [
      ["Player A", /Retired Player/],
      ["Player B", /Active Player/],
    ] as const) {
      const box = screen.getByLabelText(label);
      fireEvent.change(box, { target: { value: "p" } });
      const picker = within(box.parentElement as HTMLElement);
      fireEvent.click(await picker.findByRole("button", { name }));
    }
    await screen.findByText("€2.00M");
  }

  it("labels each side's season and warns when they differ", async () => {
    await compare(2014, 2025);
    expect(screen.getByText("Season 2014")).toBeInTheDocument();
    expect(screen.getByText("Season 2025")).toBeInTheDocument();
    const note = screen.getByRole("note");
    expect(note).toHaveTextContent("Different seasons");
    expect(note).toHaveTextContent("season 2014");
    expect(note).toHaveTextContent("season 2025");
    expect(note).toHaveTextContent("11 seasons apart");
  });

  it("does not warn when both come from the same season", async () => {
    await compare(2025, 2025);
    expect(screen.getAllByText("Season 2025")).toHaveLength(2);
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });
});

describe("what-if", () => {
  function slide(value: number) {
    fireEvent.change(screen.getByLabelText(/Goals/), { target: { value: String(value) } });
  }

  it("shows the prediction for the latest slider position", async () => {
    const first = deferred<PredictResponse>();
    const second = deferred<PredictResponse>();
    mocked.predictFromFeatures
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    render(<WhatIf baseline={1_000_000} features={{ goals: 10 }} variant="performance_only" />);

    slide(11);
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(1));
    slide(30);
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(2));
    expect(mocked.predictFromFeatures.mock.calls[1][0]).toMatchObject({ goals: 30 });

    await act(async () => second.resolve(prediction(5_000_000)));
    await act(async () => first.resolve(prediction(1_100_000)));

    expect(screen.getByText("€5.00M")).toBeInTheDocument();
    expect(screen.queryByText("predicting…")).not.toBeInTheDocument();
  });

  it("does not show an error from a request it has superseded", async () => {
    const first = deferred<PredictResponse>();
    mocked.predictFromFeatures
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce(prediction(5_000_000));
    render(<WhatIf baseline={1_000_000} features={{ goals: 10 }} variant="performance_only" />);

    slide(11);
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalledTimes(1));
    slide(30);
    expect(await screen.findByText("€5.00M")).toBeInTheDocument();
    await act(async () => first.reject(new Error("timeout")));

    expect(screen.queryByText("What-if prediction failed")).not.toBeInTheDocument();
    expect(screen.getByText("€5.00M")).toBeInTheDocument();
  });

  it("changes nothing when its response lands after unmount", async () => {
    const pending = deferred<PredictResponse>();
    mocked.predictFromFeatures.mockReturnValue(pending.promise);
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    const { unmount } = render(
      <WhatIf baseline={1_000_000} features={{ goals: 10 }} variant="performance_only" />,
    );

    slide(11);
    await waitFor(() => expect(mocked.predictFromFeatures).toHaveBeenCalled());
    unmount();
    await act(async () => pending.resolve(prediction(5_000_000)));

    expect(errors).not.toHaveBeenCalled();
  });
});
