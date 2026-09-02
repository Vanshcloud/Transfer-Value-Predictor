"use client";

/** Two players, side by side, on the same model and the same season basis. */

import { useCallback, useEffect, useRef, useState } from "react";
import { useAsync } from "@/lib/useAsync";
import Link from "next/link";
import {
  api,
  type FeatureDistribution,
  type Player,
  type PredictResponse,
  type SearchResult,
  type SimilarPlayer,
  type Variant,
} from "@/lib/api";
import { readPair, writePair } from "@/lib/comparePair";
import { eur, featureLabel } from "@/lib/format";
import Chart from "@/components/Chart";
import Radar from "@/components/Radar";
import { Avatar, Card, ClubTag, Empty, ErrorPanel, Loading } from "@/components/ui";

const COLOURS = ["#0284c7", "#f59e0b"] as const;

/** All the picker needs of a player: enough to label them and fetch them. */
type Chosen = Pick<
  SearchResult,
  "player_id" | "name" | "club" | "league" | "league_country"
>;

interface Side {
  query: string;
  results: SearchResult[];
  chosen: Chosen | null;
  prediction: PredictResponse | null;
  player: Player | null;
  similar: SimilarPlayer[];
}

const EMPTY: Side = {
  query: "",
  results: [],
  chosen: null,
  prediction: null,
  player: null,
  similar: [],
};

function PlayerPicker({
  side,
  label,
  onQuery,
  onPick,
}: {
  side: Side;
  label: string;
  onQuery: (value: string) => void;
  onPick: (result: SearchResult) => void;
}) {
  // Two identical search boxes sit on this page. Without htmlFor, a screen
  // reader announces both as an unlabelled "search" field and the visible
  // "Player A" / "Player B" never reaches it — so which box sets which side of
  // the comparison is unknowable without sight.
  const inputId = `compare-${label.replace(/\s+/g, "-").toLowerCase()}`;

  return (
    <div>
      <label
        htmlFor={inputId}
        className="text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400"
      >
        {label}
      </label>
      <input
        id={inputId}
        type="search"
        value={side.query}
        onChange={(event) => onQuery(event.target.value)}
        placeholder="Search a player"
        className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-900"
      />
      {side.results.length > 0 && !side.chosen && (
        <ul className="mt-2 max-h-48 overflow-y-auto rounded-lg border border-slate-200 dark:border-slate-800">
          {side.results.slice(0, 8).map((result) => (
            <li key={result.player_id}>
              <button
                onClick={() => onPick(result)}
                className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-slate-100 dark:hover:bg-slate-800"
              >
                <Avatar name={result.name} seed={result.player_id} size={24} />
                {/* min-w-0: a flex child will not shrink below its text's own
                    width without it, so `truncate` never fires and a long name
                    overflows the button instead of ellipsing. */}
                <span className="min-w-0 flex-1">
                  <span className="block truncate">{result.name}</span>
                  <ClubTag
                    club={result.club}
                    league={result.league}
                    country={result.league_country}
                  />
                </span>
                <span className="shrink-0 text-xs text-slate-400">{result.position}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {/* No confirmation chip. Picking a player writes their full name into
          the input itself, which is where a search box is expected to show
          what it found — the previous version left the raw query ("Jude
          bell") in the field and repeated the resolved name underneath it. */}
    </div>
  );
}

export default function ComparePage() {
  const [variant] = useState<Variant>("performance_only");
  const [left, setLeft] = useState<Side>(EMPTY);
  const [right, setRight] = useState<Side>(EMPTY);
  const [error, setError] = useState<unknown>(null);
  const [pending, setPending] = useState(false);

  // Fetched once: the population quantiles the radar normalises against do not
  // change between comparisons.
  const distributionFetcher = useCallback(
    () => api.featureDistribution(variant),
    [variant],
  );
  const { state: distributionState } =
    useAsync<FeatureDistribution>(distributionFetcher);
  const distribution =
    distributionState.status === "ready" ? distributionState.data : null;

  const runSearch = useCallback(
    async (query: string, set: (updater: (side: Side) => Side) => void) => {
      if (!query.trim()) {
        set(() => EMPTY);
        return;
      }
      try {
        const { results } = await api.searchPlayers(query, 8);
        set((side) => ({
          ...side,
          results: results.filter((r) => r.predictable),
        }));
      } catch (caught) {
        setError(caught);
      }
    },
    [],
  );

  useEffect(() => {
    const timer = setTimeout(() => runSearch(left.query, setLeft), 250);
    return () => clearTimeout(timer);
  }, [left.query, runSearch]);

  useEffect(() => {
    const timer = setTimeout(() => runSearch(right.query, setRight), 250);
    return () => clearTimeout(timer);
  }, [right.query, runSearch]);

  const pick = useCallback(
    async (result: Chosen, set: (updater: (side: Side) => Side) => void) => {
      // The query becomes the resolved name, so the field shows who was
      // actually chosen rather than whatever was typed to find them.
      set((side) => ({
        ...side,
        chosen: result,
        results: [],
        query: result.name,
      }));
      setPending(true);
      try {
        const [prediction, player, neighbours] = await Promise.all([
          // 25 is the API's maximum. Both players need a value for the same
          // features before their bars can be compared, and eight of each sign
          // left gaps the chart drew as a missing bar.
          api.predictForPlayer(result.player_id, variant, undefined, 25),
          api.player(result.player_id),
          api
            .similarPlayers(result.player_id, 5, variant)
            .catch(() => ({ results: [] })),
        ]);
        set((side) => ({
          ...side,
          prediction,
          player,
          similar: neighbours.results,
        }));
      } catch (caught) {
        setError(caught);
      } finally {
        setPending(false);
      }
    },
    [variant],
  );

  // ---- the comparison lives in the URL ------------------------------------
  //
  // Both sides were React state and nothing else, so opening a player's page
  // and pressing Back returned an empty form: the component had unmounted and
  // remounted with no memory of who was being compared, and both names had to
  // be typed again. The pair is the page's entire meaning, so it belongs in the
  // address bar — which also makes a comparison a link somebody can send.
  //
  // `replaceState`, not `pushState`: picking a player is not a navigation, and
  // pushing would make Back walk backwards through every selection instead of
  // leaving the page.
  const restored = useRef(false);

  useEffect(() => {
    if (restored.current) return;
    restored.current = true;

    const pair = readPair(window.location.search);
    const load = async (id: number | null, set: typeof setLeft) => {
      if (id === null) return;
      try {
        const player = await api.player(id);
        // The same labels the picker showed, rebuilt from the record: a
        // restored comparison must not lose the clubs the link was made with.
        await pick(
          {
            player_id: id,
            name: player.name ?? `Player ${id}`,
            club: player.club,
            league: player.league,
            league_country: player.league_country,
          },
          set,
        );
      } catch {
        // A stale or hand-edited link should leave an empty picker rather than
        // an error banner: there is nothing for the reader to fix.
      }
    };
    void Promise.all([load(pair.a, setLeft), load(pair.b, setRight)]);
  }, [pick]);

  useEffect(() => {
    if (!restored.current) return;
    window.history.replaceState(
      null,
      "",
      writePair(
        window.location.pathname,
        left.chosen?.player_id,
        right.chosen?.player_id,
      ),
    );
  }, [left.chosen, right.chosen]);

  const both = left.prediction && right.prediction;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Compare</h1>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          Two players, the same model, the same basis.
        </p>
      </div>

      {error != null && <ErrorPanel error={error} />}

      <Card>
        <div className="grid gap-6 sm:grid-cols-2">
          <PlayerPicker
            side={left}
            label="Player A"
            onQuery={(value) => setLeft({ ...EMPTY, query: value })}
            onPick={(result) => pick(result, setLeft)}
          />
          <PlayerPicker
            side={right}
            label="Player B"
            onQuery={(value) => setRight({ ...EMPTY, query: value })}
            onPick={(result) => pick(result, setRight)}
          />
        </div>
      </Card>

      {pending && (
        <Card>
          <Loading />
        </Card>
      )}

      {!both && !pending && (
        <Card>
          <Empty>Choose two players to compare.</Empty>
        </Card>
      )}

      {both && (
        <>
          <Card title="Predicted value">
            <div className="grid gap-6 sm:grid-cols-2">
              {[left, right].map((side, index) => (
                <div key={index}>
                  <div className="flex min-w-0 items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
                    {side.chosen && (
                      <Avatar name={side.chosen.name} seed={side.chosen.player_id} size={28} />
                    )}
                    <span className="truncate">{side.chosen?.name}</span>
                    {side.chosen && (
                      <ClubTag
                        club={side.chosen.club}
                        league={side.chosen.league}
                        country={side.chosen.league_country}
                      />
                    )}
                  </div>
                  <div className="mt-1 text-4xl font-semibold tabular-nums">
                    {eur(side.prediction!.prediction_eur)}
                  </div>
                  {side.prediction!.confidence && (
                    <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                      {eur(side.prediction!.confidence.lower_eur)} —{" "}
                      {eur(side.prediction!.confidence.upper_eur)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>

          <Card
            title="What drives each"
            subtitle="Contributions side by side, in the model's log space"
          >
            <Chart
              title="Contribution comparison"
              height={420}
              data={(() => {
                // Both traces must span the same features, or Plotly unions the
                // two category lists and any feature only one player returned
                // draws a single bar — which read as "this player has no value
                // here" when it meant "this was not in that player's top slice".
                //
                // So: build one ordered axis from the union, look each player's
                // own contribution up on it, and rank by the larger of the two
                // magnitudes so a feature that matters to either of them earns
                // its row.
                const byFeature = [left, right].map((side) => {
                  const explanation = side.prediction!.explanation;
                  return new Map(
                    [
                      ...(explanation?.top_positive_features ?? []),
                      ...(explanation?.top_negative_features ?? []),
                    ].map((c) => [c.feature, c.shap_value]),
                  );
                });

                const features = [
                  ...new Set(byFeature.flatMap((m) => [...m.keys()])),
                ]
                  .sort(
                    (a, b) =>
                      Math.max(
                        ...byFeature.map((m) => Math.abs(m.get(a) ?? 0)),
                      ) -
                      Math.max(
                        ...byFeature.map((m) => Math.abs(m.get(b) ?? 0)),
                      ),
                  )
                  .slice(-16);

                return [left, right].map((side, index) => ({
                  type: "bar" as const,
                  name: side.chosen?.name ?? `Player ${index + 1}`,
                  orientation: "h" as const,
                  x: features.map((f) => byFeature[index].get(f) ?? 0),
                  y: features.map((f) => featureLabel(f)),
                  marker: { color: index === 0 ? "#0284c7" : "#f59e0b" },
                }));
              })()}
              layout={{
                // Grouped is Plotly's default for bar traces, so barmode is
                // left unset rather than restated.
                showlegend: true,
                legend: { orientation: "h", y: 1.08 },
                margin: { l: 190, r: 20, t: 40, b: 40 },
                xaxis: { title: { text: "contribution (log space)" } },
              }}
            />
          </Card>

          {distribution && left.player && right.player && (
            <Card
              title="Profile"
              subtitle="Each axis is a percentile across the whole panel"
            >
              <Radar
                distribution={distribution}
                series={[
                  {
                    label: left.chosen?.name ?? "A",
                    colour: COLOURS[0],
                    features: left.player.features,
                  },
                  {
                    label: right.chosen?.name ?? "B",
                    colour: COLOURS[1],
                    features: right.player.features,
                  },
                ]}
              />
            </Card>
          )}

          <div className="grid gap-6 sm:grid-cols-2">
            {[left, right].map((side, index) => (
              <Card key={index} title={`Similar to ${side.chosen?.name ?? ""}`}>
                {side.similar.length === 0 ? (
                  <Empty>No comparable seasons.</Empty>
                ) : (
                  <ul className="divide-y divide-slate-100 dark:divide-slate-800">
                    {side.similar.map((player) => (
                      <li key={`${player.player_id}-${player.season}`}>
                        <Link
                          href={`/players/${player.player_id}`}
                          className="flex justify-between gap-3 py-2 text-sm hover:bg-slate-50 dark:hover:bg-slate-800/50"
                        >
                          <span className="truncate">
                            {player.name ?? `Player ${player.player_id}`}
                          </span>
                          <span className="shrink-0 tabular-nums text-slate-500 dark:text-slate-400">
                            {eur(player.market_value_in_eur)}
                          </span>
                        </Link>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            ))}
          </div>

          <Card>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              To change either player&apos;s season and watch the value move,
              open their page:{" "}
              {[left, right].map((side, index) => (
                <span key={index}>
                  {index > 0 && " · "}
                  <Link
                    href={`/players/${side.chosen?.player_id}`}
                    className="text-sky-600 hover:underline dark:text-sky-400"
                  >
                    {side.chosen?.name}
                  </Link>
                </span>
              ))}
            </p>
          </Card>
        </>
      )}
    </div>
  );
}
