"use client";

/**
 * The product flow, on one page:
 *   prediction → interval → contributions → similar seasons → what-if
 *
 * Everything is fetched in parallel and rendered as it arrives, so a slow SHAP
 * call does not hold back the prediction the reader came for.
 */

import { use, useCallback, useState } from "react";
import { useAsync } from "@/lib/useAsync";
import Link from "next/link";
import {
  api,
  type HistoryPoint,
  type ModelInfo,
  type ModelMetrics,
  type Player,
  type PredictResponse,
  type SimilarPlayer,
  type Variant,
} from "@/lib/api";
import Contributions from "@/components/Contributions";
import ModelMetadata from "@/components/ModelMetadata";
import PredictionHistory from "@/components/PredictionHistory";
import PredictionPanel from "@/components/PredictionPanel";
import SimilarPlayers from "@/components/SimilarPlayers";
import ValueHistory from "@/components/ValueHistory";
import WhatIf from "@/components/WhatIf";
import { Avatar, Card, ClubTag, ErrorPanel, Loading } from "@/components/ui";

const VARIANTS: { key: Variant; label: string; blurb: string }[] = [
  {
    key: "performance_only",
    label: "Performance only",
    blurb:
      "Has never been told the market's opinion, so it can disagree with it.",
  },
  {
    key: "with_prior_value",
    label: "With prior value",
    blurb: "Anchored to a known valuation: more accurate, less independent.",
  },
];

interface PageData {
  player: Player;
  prediction: PredictResponse;
  similar: SimilarPlayer[];
  info: ModelInfo;
  metrics: ModelMetrics;
  history: HistoryPoint[];
}

export default function PlayerPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const playerId = Number(id);

  const [variant, setVariant] = useState<Variant>("performance_only");

  const fetcher = useCallback(async (): Promise<PageData> => {
    // The player record is what the page cannot render without, so it is
    // awaited first: a 404 here should not wait on four other requests.
    const player = await api.player(playerId);

    const [prediction, neighbours, info, metrics, history] = await Promise.all([
      api.predictForPlayer(playerId, variant),
      // Neighbours and history are nice-to-haves. Losing either should not
      // blank a page that has a prediction to show.
      api.similarPlayers(playerId, 8, variant).catch(() => ({ results: [] })),
      api.modelInfo(variant),
      api.modelMetrics(variant),
      api.predictionHistory(playerId, variant).catch(() => ({ points: [] })),
    ]);

    return {
      player,
      prediction,
      similar: neighbours.results,
      info,
      metrics,
      history: history.points,
    };
  }, [playerId, variant]);

  const { state, reload } = useAsync(fetcher);
  const loading = state.status === "loading";
  const error = state.status === "error" ? state.error : null;
  const { player, prediction, similar, info, metrics, history } =
    state.status === "ready"
      ? state.data
      : {
          player: null,
          prediction: null,
          similar: [] as SimilarPlayer[],
          info: null,
          metrics: null,
          history: [] as HistoryPoint[],
        };

  if (error != null) {
    return (
      <div className="space-y-4">
        <ErrorPanel error={error} onRetry={reload} />
        <Link
          href="/players"
          className="text-sm text-sky-600 hover:underline dark:text-sky-400"
        >
          ← Back to search
        </Link>
      </div>
    );
  }

  // The most recent season that actually carries a valuation. `latest` is the
  // season being played for anyone currently active, and that row has no
  // market value by construction — so reading the recorded value off it made
  // the figure vanish for exactly the players people look up. The search list
  // and the similar-seasons panel both show that value, and the page it links
  // to was the one place it could not be seen.
  const lastValued =
    [...(player?.seasons ?? [])]
      .reverse()
      .find((s) => s.market_value_in_eur != null) ?? null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link
            href="/players"
            className="text-sm text-slate-500 hover:underline dark:text-slate-400"
          >
            ← Search
          </Link>
          <div className="mt-1 flex items-center gap-3">
            <Avatar name={player?.name ?? `Player ${playerId}`} seed={playerId} size={56} />
            <h1 className="text-3xl font-semibold tracking-tight">
              {player?.name ?? `Player ${playerId}`}
            </h1>
            {player && (
              <ClubTag
                club={player.club}
                league={player.league}
                country={player.league_country}
              />
            )}
          </div>
          {player && (
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              {[
                player.sub_position ?? player.position,
                player.country_of_citizenship,
                player.foot && `${player.foot} footed`,
                player.height_in_cm && `${player.height_in_cm.toFixed(0)} cm`,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
          )}
        </div>

        <fieldset className="flex gap-1 rounded-lg border border-slate-200 p-1 dark:border-slate-800">
          <legend className="sr-only">Model variant</legend>
          {VARIANTS.map((option) => (
            <button
              key={option.key}
              onClick={() => setVariant(option.key)}
              title={option.blurb}
              aria-pressed={variant === option.key}
              className={`rounded px-3 py-1.5 text-sm transition ${
                variant === option.key
                  ? "bg-sky-600 font-medium text-white"
                  : "text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
              }`}
            >
              {option.label}
            </button>
          ))}
        </fieldset>
      </div>

      {loading && (
        <Card>
          <Loading label="Loading player" />
        </Card>
      )}

      {!loading && prediction && (
        <>
          <PredictionPanel
            prediction={prediction}
            actual={lastValued?.market_value_in_eur ?? null}
            actualSeason={lastValued?.season ?? null}
          />

          <div className="grid gap-6 lg:grid-cols-3">
            <div className="space-y-6 lg:col-span-2">
              {prediction.explanation && (
                <Contributions
                  positive={prediction.explanation.top_positive_features}
                  negative={prediction.explanation.top_negative_features}
                />
              )}
              {history.length > 0 ? (
                <PredictionHistory points={history} />
              ) : (
                player && <ValueHistory seasons={player.seasons} />
              )}
            </div>

            <div className="space-y-6">
              <ModelMetadata info={info} metrics={metrics} />
              <SimilarPlayers
                players={similar}
                season={similar[0]?.season ?? null}
              />
            </div>
          </div>

          {player && Object.keys(player.features).length > 0 && (
            // Keyed so switching player or variant remounts with fresh
            // sliders. Resetting via an effect would be a state write during
            // render for something a key expresses directly.
            <WhatIf
              key={`${player.player_id}-${variant}`}
              baseline={prediction.prediction_eur}
              features={player.features}
              variant={variant}
            />
          )}
        </>
      )}
    </div>
  );
}
