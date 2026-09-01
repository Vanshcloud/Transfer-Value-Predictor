"use client";

/**
 * League-level view, assembled from search results rather than a bespoke
 * endpoint: the API is read-only and this page asks a question it already
 * answers. If this ever needs the whole panel, it wants a real aggregate
 * endpoint, not a hundred client-side requests.
 */

import { useCallback } from "react";
import { useAsync } from "@/lib/useAsync";
import { api, type SearchResult } from "@/lib/api";
import { eur } from "@/lib/format";
import Chart from "@/components/Chart";
import { Card, Empty, ErrorPanel, Loading } from "@/components/ui";

/** Common name fragments, used to pull a broad slice without a bulk endpoint. */
const SEEDS = ["a", "e", "i", "o", "u"];

export default function AnalyticsPage() {
  const fetcher = useCallback(async (): Promise<SearchResult[]> => {
    const batches = await Promise.all(
      SEEDS.map((seed) => api.searchPlayers(seed, 50)),
    );
    const seen = new Map<number, SearchResult>();
    for (const batch of batches) {
      for (const result of batch.results) {
        if (result.predictable && result.market_value_in_eur != null) {
          seen.set(result.player_id, result);
        }
      }
    }
    return [...seen.values()];
  }, []);

  const { state, reload } = useAsync(fetcher);
  const loading = state.status === "loading";
  const error = state.status === "error" ? state.error : null;
  const players = state.status === "ready" ? state.data : [];

  const byPosition = new Map<string, number[]>();
  for (const player of players) {
    const key = player.position ?? "Unknown";
    byPosition.set(key, [
      ...(byPosition.get(key) ?? []),
      player.market_value_in_eur ?? 0,
    ]);
  }

  const values = players
    .map((p) => p.market_value_in_eur ?? 0)
    .filter((v) => v > 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Analytics</h1>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          How market value is distributed across the sampled panel.
        </p>
      </div>

      {error != null && <ErrorPanel error={error} onRetry={reload} />}
      {loading && (
        <Card>
          <Loading label="Loading panel" />
        </Card>
      )}

      {!loading && players.length === 0 && (
        <Card>
          <Empty>No players available.</Empty>
        </Card>
      )}

      {!loading && values.length > 0 && (
        <>
          <Card
            title="Value distribution"
            subtitle={`${players.length} sampled players — log scale, because the target spans four orders of magnitude`}
          >
            <Chart
              title="Distribution of market value"
              height={320}
              data={[
                {
                  type: "histogram",
                  x: values.map((v) => Math.log10(v)),
                  marker: { color: "#0ea5e9" },
                  hovertemplate:
                    "10^%{x:.1f} EUR<br>%{y} players<extra></extra>",
                },
              ]}
              layout={{
                xaxis: { title: { text: "log₁₀ market value (EUR)" } },
                yaxis: { title: { text: "players" } },
              }}
            />
            <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
              This skew is why the model trains on log1p of the value: raw, a
              handful of €100M players dominate the loss and the model learns
              little about everyone else.
            </p>
          </Card>

          <Card
            title="By position"
            subtitle="Median value per position in the sample"
          >
            <Chart
              title="Median value by position"
              height={280}
              data={[
                {
                  type: "bar",
                  x: [...byPosition.keys()],
                  y: [...byPosition.values()].map((group) => {
                    const sorted = [...group].sort((a, b) => a - b);
                    return sorted[Math.floor(sorted.length / 2)] ?? 0;
                  }),
                  marker: { color: "#6366f1" },
                  hovertemplate: "%{x}<br>median %{y:,.0f} EUR<extra></extra>",
                },
              ]}
              layout={{
                yaxis: { title: { text: "median market value (EUR)" } },
                margin: { l: 80, r: 20, t: 8, b: 60 },
              }}
            />
            <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
              {[...byPosition.entries()].map(([position, group]) => (
                <div key={position}>
                  <dt className="text-xs text-slate-500 uppercase dark:text-slate-400">
                    {position}
                  </dt>
                  <dd className="tabular-nums">
                    {eur(group.reduce((a, b) => a + b, 0) / group.length)}
                    <span className="ml-1 text-xs text-slate-400">mean</span>
                  </dd>
                </div>
              ))}
            </dl>
          </Card>
        </>
      )}
    </div>
  );
}
