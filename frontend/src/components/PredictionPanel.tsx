"use client";

import type { Confidence, PredictResponse } from "@/lib/api";
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
  actualSeason,
}: {
  prediction: PredictResponse;
  actual?: number | null;
  /**
   * Which season `actual` was recorded in. Usually the predicted season, but
   * not for a player whose newest row is the season being played — that row
   * has no valuation, so the most recent published one is older and saying
   * which year it belongs to is the difference between context and a
   * misleading comparison.
   */
  actualSeason?: number | null;
}) {
  const confidence = prediction.confidence;
  // Only a like-for-like comparison. Differencing this season's prediction
  // against an earlier season's valuation would present the market moving as
  // though the model had got something wrong.
  const sameSeason = actualSeason == null || actualSeason === prediction.season;
  const error =
    actual != null && sameSeason ? prediction.prediction_eur - actual : null;

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
              label={
                actualSeason != null && !sameSeason
                  ? `Recorded value · ${actualSeason}`
                  : "Recorded value"
              }
              value={eur(actual)}
              hint={
                error != null
                  ? `${error >= 0 ? "over" : "under"} by ${eur(Math.abs(error))}`
                  : !sameSeason
                    ? "last published valuation"
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
              {percent(confidence.level, 0)} nominal interval
            </span>
            <span className="font-mono text-sm tabular-nums text-slate-900 dark:text-slate-100">
              {eur(confidence.lower_eur)} — {eur(confidence.upper_eur)}
            </span>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
            Not a probability. This is measured from the model&apos;s own errors
            on {confidence.reference_rows.toLocaleString()} held-out rows —{" "}
            {confidence.basis}. It is wide because the target is hard to pin
            down: the first valuation published on or after the season&apos;s
            cut-off, which can come the same day or up to a year later.
          </p>
          <IntervalRecord confidence={confidence} />
        </div>
      )}
    </Card>
  );
}

/**
 * What the interval achieved on the test seasons — not fitted on, but
 * inspected during development, so the record is retrospective — for
 * predictions in this one's value band when that was measured, overall
 * otherwise — with the rows behind the figure.
 *
 * Worded as a frequency over a group, never as a chance for this player: an
 * 80% interval that covered 74% of a band's test rows says nothing about
 * whether this player's value lies inside it.
 */
export function IntervalRecord({ confidence }: { confidence: Confidence }) {
  const band =
    typeof confidence.band_measured_coverage === "number" && confidence.band
      ? {
          coverage: confidence.band_measured_coverage,
          rows: confidence.band_coverage_rows ?? null,
          group: `predictions in the ${confidence.band} band`,
        }
      : null;
  const overall =
    typeof confidence.measured_coverage === "number"
      ? { coverage: confidence.measured_coverage, rows: null, group: "all predictions" }
      : null;
  const record = band ?? overall;
  if (record == null) return null;

  return (
    <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
      On the test seasons, intervals for {record.group}
      {record.rows != null && ` (${record.rows.toLocaleString()} rows)`} contained{" "}
      <span className="font-medium text-slate-700 dark:text-slate-200">
        {percent(record.coverage, 0)}
      </span>{" "}
      of true values, against the {percent(confidence.level, 0)} targeted
      {band && overall && ` (${percent(overall.coverage, 0)} across all predictions)`}. That
      is a retrospective record for the group, not a probability for this player.
    </p>
  );
}
