"use client";

import type { PredictResponse } from "@/lib/api";
import { eur, eurExact, percent } from "@/lib/format";
import { Card, Stat } from "./ui";

/**
 * The prediction, its interval, and what the model expected before seeing this
 * player. The interval is shown as prominently as the number because it is
 * wide, and a point estimate presented alone claims a precision the model does
 * not have.
 */
export default function PredictionPanel({
  prediction,
  actual,
}: {
  prediction: PredictResponse;
  actual?: number | null;
}) {
  const confidence = prediction.confidence;
  const error = actual != null ? prediction.prediction_eur - actual : null;

  return (
    <Card>
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <div className="text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
            Predicted market value
            {prediction.season != null && ` · season ${prediction.season}`}
          </div>
          <div className="mt-1 text-5xl font-semibold tabular-nums text-slate-900 dark:text-slate-50">
            {eur(prediction.prediction_eur)}
          </div>
          <div className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            {eurExact(prediction.prediction_eur)}
          </div>
        </div>

        <div className="flex flex-wrap gap-8">
          {actual != null && (
            <Stat
              label="Recorded value"
              value={eur(actual)}
              hint={
                error != null
                  ? `${error >= 0 ? "over" : "under"} by ${eur(Math.abs(error))}`
                  : undefined
              }
            />
          )}
          {prediction.explanation && (
            <Stat
              label="Model baseline"
              value={eur(prediction.explanation.base_value_eur)}
              hint="before any feature"
            />
          )}
        </div>
      </div>

      {confidence && (
        <div className="mt-6 rounded-lg border border-slate-200 p-4 dark:border-slate-800">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <span className="text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
              {percent(confidence.level, 0)} prediction interval
            </span>
            <span className="font-mono text-sm tabular-nums text-slate-900 dark:text-slate-100">
              {eur(confidence.lower_eur)} — {eur(confidence.upper_eur)}
            </span>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
            Not a probability. This is measured from the model&apos;s own errors on{" "}
            {confidence.reference_rows.toLocaleString()} held-out rows —{" "}
            {confidence.basis}. It is wide because predicting market value a season
            ahead is genuinely hard, not because something is broken.
          </p>
        </div>
      )}
    </Card>
  );
}
