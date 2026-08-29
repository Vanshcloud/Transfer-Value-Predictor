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
  type ModelInfo,
  type ModelMetrics,
  type Player,
  type PredictResponse,
  type SimilarPlayer,
  type Variant,
} from "@/lib/api";
import Contributions from "@/components/Contributions";
import ModelMetadata from "@/components/ModelMetadata";
import PredictionPanel from "@/components/PredictionPanel";
import SimilarPlayers from "@/components/SimilarPlayers";
import ValueHistory from "@/components/ValueHistory";
import WhatIf from "@/components/WhatIf";
import { Card, ErrorPanel, Loading } from "@/components/ui";

const VARIANTS: { key: Variant; label: string; blurb: string }[] = [
  {
    key: "performance_only",
    label: "Performance only",
    blurb: "Has never been told the market's opinion, so it can disagree with it.",
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
}

export default function PlayerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const playerId = Number(id);

  const [variant, setVariant] = useState<Variant>("performance_only");

  const fetcher = useCallback(async (): Promise<PageData> => {
    // The player record is what the page cannot render without, so it is
    // awaited first: a 404 here should not wait on four other requests.
    const player = await api.player(playerId);

    const [prediction, neighbours, info, metrics] = await Promise.all([
      api.predictForPlayer(playerId, variant),
      // Neighbours are a nice-to-have. Losing them should not blank a page
      // that has a prediction to show.
      api.similarPlayers(playerId, 8, variant).catch(() => ({ results: [] })),
      api.modelInfo(variant),
      api.modelMetrics(variant),
    ]);

    return { player, prediction, similar: neighbours.results, info, metrics };
  }, [playerId, variant]);

  const { state, reload } = useAsync(fetcher);
  const loading = state.status === "loading";
  const error = state.status === "error" ? state.error : null;
  const { player, prediction, similar, info, metrics } =
    state.status === "ready"
      ? state.data
      : {
          player: null,
          prediction: null,
          similar: [] as SimilarPlayer[],
          info: null,
          metrics: null,
        };

  if (error != null) {
    return (
      <div className="space-y-4">
        <ErrorPanel error={error} onRetry={reload} />
        <Link href="/players" className="text-sm text-sky-600 hover:underline dark:text-sky-400">
          ← Back to search
        </Link>
      </div>
    );
  }

  const latest = player?.seasons.at(-1) ?? null;

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
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">
            {player?.name ?? `Player ${playerId}`}
          </h1>
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
            actual={latest?.market_value_in_eur ?? null}
          />

          <div className="grid gap-6 lg:grid-cols-3">
            <div className="space-y-6 lg:col-span-2">
              {prediction.explanation && (
                <Contributions
                  positive={prediction.explanation.top_positive_features}
                  negative={prediction.explanation.top_negative_features}
                />
              )}
              {player && <ValueHistory seasons={player.seasons} />}
            </div>

            <div className="space-y-6">
              <ModelMetadata info={info} metrics={metrics} />
              <SimilarPlayers players={similar} season={similar[0]?.season ?? null} />
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
