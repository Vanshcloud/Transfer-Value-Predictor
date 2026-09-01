"use client";

/** How good the model is, and how that was measured. */

import { useCallback, useState } from "react";
import { useAsync } from "@/lib/useAsync";
import {
  api,
  type FeatureImportance,
  type ModelInfo,
  type ModelMetrics,
  type Variant,
} from "@/lib/api";
import { eur, featureLabel, number, percent, shortDate } from "@/lib/format";
import Chart from "@/components/Chart";
import { Card, ErrorPanel, Loading, Stat } from "@/components/ui";

const VARIANTS: Variant[] = ["performance_only", "with_prior_value"];

export default function ModelPage() {
  const [variant, setVariant] = useState<Variant>("performance_only");

  const fetcher = useCallback(
    async (): Promise<[ModelInfo, ModelMetrics, FeatureImportance]> =>
      Promise.all([
        api.modelInfo(variant),
        api.modelMetrics(variant),
        api.featureImportance(variant, 15, true),
      ]),
    [variant],
  );
  const { state, reload } = useAsync(fetcher);

  const loading = state.status === "loading";
  const error = state.status === "error" ? state.error : null;
  const [info, metrics, importance] =
    state.status === "ready" ? state.data : [null, null, null];
  const leaderboard = (metrics?.leaderboard ?? []) as Record<
    string,
    number | string
  >[];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Model</h1>
          <p className="mt-2 text-slate-600 dark:text-slate-400">
            Measured on held-out seasons, in euros.
          </p>
        </div>
        <fieldset className="flex gap-1 rounded-lg border border-slate-200 p-1 dark:border-slate-800">
          <legend className="sr-only">Model variant</legend>
          {VARIANTS.map((option) => (
            <button
              key={option}
              onClick={() => setVariant(option)}
              aria-pressed={variant === option}
              className={`rounded px-3 py-1.5 text-sm ${
                variant === option
                  ? "bg-sky-600 font-medium text-white"
                  : "text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
              }`}
            >
              {option.replace(/_/g, " ")}
            </button>
          ))}
        </fieldset>
      </div>

      {error != null && <ErrorPanel error={error} onRetry={reload} />}
      {loading && (
        <Card>
          <Loading />
        </Card>
      )}

      {!loading && metrics && (
        <Card title="Test performance" subtitle="Seasons the model never saw">
          <div className="flex flex-wrap gap-8">
            <Stat
              label="MAE"
              value={eur(metrics.test.mae_eur)}
              hint="typical miss"
            />
            <Stat
              label="RMSE"
              value={eur(metrics.test.rmse_eur)}
              hint="penalises big misses"
            />
            <Stat label="R²" value={metrics.test.r2?.toFixed(3) ?? "—"} />
            <Stat label="MAPE" value={percent(metrics.test.mape)} />
            <Stat label="Rows" value={number(metrics.test.n)} />
          </div>
          <p className="mt-5 text-sm leading-relaxed text-slate-500 dark:text-slate-400">
            Validation MAE was {eur(metrics.validation.mae_eur)} over{" "}
            {number(metrics.validation.n)} rows. Validation tunes; test is
            touched once. A test set consulted repeatedly is a validation set
            wearing a disguise.
          </p>
        </Card>
      )}

      {!loading && info && (
        <Card title="Provenance">
          <dl className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
            {[
              ["Family", info.model_name],
              ["Trained", shortDate(info.trained_at)],
              ["Artifact version", `v${info.artifact_version}`],
              ["Seed", String(info.seed)],
              ["Features", String(info.feature_columns.length)],
              [
                "Dataset",
                `${number(Number(info.dataset?.rows ?? 0))} player-seasons`,
              ],
              [
                "Split",
                `train ≤${info.split.train_end_season} · validate ${info.split.validation_season} · test ≥${info.split.test_start_season}`,
              ],
              ["Target", `${info.target_column} (log1p, reported in EUR)`],
            ].map(([label, value]) => (
              <div
                key={label}
                className="flex justify-between gap-4 border-b border-slate-100 py-1.5 dark:border-slate-800"
              >
                <dt className="text-sm text-slate-500 dark:text-slate-400">
                  {label}
                </dt>
                <dd className="text-right text-sm font-medium">
                  {String(value)}
                </dd>
              </div>
            ))}
          </dl>
        </Card>
      )}

      {!loading && leaderboard.length > 0 && (
        <Card
          title="Why this family"
          subtitle="Every family tried, ranked by validation MAE"
        >
          <Chart
            title="Validation MAE by model family"
            height={Math.max(260, leaderboard.length * 30)}
            data={[
              {
                type: "bar",
                orientation: "h",
                x: leaderboard
                  .map((row) => Number(row.validation_mae_eur))
                  .reverse(),
                y: leaderboard.map((row) => String(row.model)).reverse(),
                marker: {
                  color: leaderboard
                    .map((row) =>
                      row.model === info?.model_name ? "#0284c7" : "#94a3b8",
                    )
                    .reverse(),
                },
                hovertemplate: "%{y}<br>%{x:,.0f} EUR MAE<extra></extra>",
              },
            ]}
            layout={{
              margin: { l: 150, r: 20, t: 8, b: 42 },
              xaxis: { title: { text: "validation MAE (EUR)" } },
            }}
          />
          {/* Derived, never hardcoded. This read "nine families ... about 10%"
              until the final audit: the registry had grown to eleven and the
              spread was 38%, so a sentence written once had been quietly wrong
              on every page load since. Computing it from the same array the
              chart plots means it cannot drift again. */}
          <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
            The spread across {leaderboard.length} families is about{" "}
            {Math.round(
              (100 *
                (Number(
                  leaderboard[leaderboard.length - 1].validation_mae_eur,
                ) -
                  Number(leaderboard[0].validation_mae_eur))) /
                Number(leaderboard[0].validation_mae_eur),
            )}
            %. The signal in this data is in the features, not the estimator.
          </p>
        </Card>
      )}

      {!loading && importance && (
        <Card
          title="What the model relies on"
          subtitle="Mean absolute SHAP — how much each feature actually moves predictions"
        >
          {importance.shap?.features?.length ? (
            <Chart
              title="Feature impact"
              height={Math.max(280, importance.shap.features.length * 26)}
              data={[
                {
                  type: "bar",
                  orientation: "h",
                  x: importance.shap.features
                    .map((f) => f.mean_abs_shap)
                    .reverse(),
                  y: importance.shap.features
                    .map((f) => featureLabel(f.feature))
                    .reverse(),
                  marker: { color: "#0ea5e9" },
                  hovertemplate: "%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>",
                },
              ]}
              layout={{
                margin: { l: 200, r: 20, t: 8, b: 42 },
                xaxis: { title: { text: "mean |SHAP| (log space)" } },
              }}
            />
          ) : (
            <ul className="text-sm">
              {importance.features.map((feature) => (
                <li
                  key={feature.feature}
                  className="flex justify-between border-b border-slate-100 py-1.5 dark:border-slate-800"
                >
                  <span>{featureLabel(feature.feature)}</span>
                  <span className="tabular-nums">
                    {feature.importance.toFixed(3)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}
    </div>
  );
}
