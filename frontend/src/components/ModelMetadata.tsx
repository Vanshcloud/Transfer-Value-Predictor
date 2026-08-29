"use client";

/**
 * Model provenance, shown next to the prediction that came from it.
 *
 * A number with no attribution invites more trust than it has earned. This
 * panel answers "which model, trained when, on what, and how wrong is it
 * usually" without exposing anything a reader would have to be an engineer to
 * interpret — no hyperparameters, no file paths.
 */

import { eur, number, shortDate } from "@/lib/format";
import type { ModelInfo, ModelMetrics } from "@/lib/api";
import { Card } from "./ui";

export default function ModelMetadata({
  info,
  metrics,
}: {
  info: ModelInfo | null;
  metrics: ModelMetrics | null;
}) {
  if (!info) return null;

  const split = info.split as Record<string, number | string | undefined>;
  const rows: [string, string][] = [
    ["Model", info.model_name],
    ["Variant", info.variant.replace(/_/g, " ")],
    ["Trained", shortDate(info.trained_at)],
    ["Dataset", `${number(Number(info.dataset?.rows ?? 0))} player-seasons`],
    ["Artifact version", `v${info.artifact_version}`],
    ["Seed", String(info.seed)],
    [
      "Evaluation",
      `temporal — train ≤${split.train_end_season}, test ≥${split.test_start_season}`,
    ],
  ];

  if (metrics?.test) {
    rows.push(["Temporal MAE", eur(metrics.test.mae_eur)]);
    rows.push(["Temporal R²", metrics.test.r2?.toFixed(3) ?? "—"]);
  }

  return (
    <Card title="Model" subtitle="Where this prediction came from">
      <dl className="grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-4 border-b border-slate-100 py-1.5 dark:border-slate-800">
            <dt className="text-sm text-slate-500 dark:text-slate-400">{label}</dt>
            <dd className="text-right text-sm font-medium tabular-nums text-slate-900 dark:text-slate-100">
              {value}
            </dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        Metrics are measured on <strong>seasons the model never saw</strong>. A random
        split would report roughly 60% better error and would not describe how the
        model behaves on a season that has not happened yet.
      </p>
    </Card>
  );
}
