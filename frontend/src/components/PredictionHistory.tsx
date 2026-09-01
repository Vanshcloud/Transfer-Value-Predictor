"use client";

/**
 * Predicted against actual, season by season.
 *
 * One prediction says what the model thinks. A series says whether it tracks a
 * career or merely lands somewhere near the middle — and it makes the split
 * visible, because agreement on a season the model trained on is not evidence
 * of anything. Training seasons are drawn faintly for exactly that reason.
 */

import type { HistoryPoint } from "@/lib/api";
import { eur } from "@/lib/format";
import Chart from "./Chart";
import { Card, Empty } from "./ui";

export default function PredictionHistory({
  points,
}: {
  points: HistoryPoint[];
}) {
  if (points.length === 0) {
    return (
      <Card title="Predicted vs actual">
        <Empty>No seasons this model can predict for.</Empty>
      </Card>
    );
  }

  const heldOut = points.filter((p) => p.held_out);
  const meanAbsHeldOut =
    heldOut.length > 0
      ? heldOut.reduce((total, p) => total + Math.abs(p.error_eur), 0) /
        heldOut.length
      : null;

  return (
    <Card
      title="Predicted vs actual"
      subtitle="How the model tracks this career, season by season"
    >
      <Chart
        title="Predicted against actual market value by season"
        height={320}
        data={[
          {
            type: "scatter",
            mode: "lines+markers",
            name: "actual",
            x: points.map((p) => p.season),
            y: points.map((p) => p.actual_eur),
            line: { color: "#0284c7", width: 2 },
            marker: { size: 8 },
            hovertemplate: "%{x} actual<br>%{y:,.0f} EUR<extra></extra>",
          },
          {
            type: "scatter",
            mode: "lines+markers",
            name: "predicted",
            x: points.map((p) => p.season),
            y: points.map((p) => p.predicted_eur),
            line: { color: "#f59e0b", width: 2, dash: "dot" },
            // Held-out seasons are drawn solid and training seasons hollow:
            // the model has seen the training ones, so how close it lands
            // there says nothing about how it will behave in future.
            marker: {
              size: 10,
              symbol: points.map((p) =>
                p.held_out ? "circle" : "circle-open",
              ),
            },
            customdata: points.map((p) =>
              p.held_out ? "held out" : "in training range",
            ),
            hovertemplate:
              "%{x} predicted<br>%{y:,.0f} EUR<br>%{customdata}<extra></extra>",
          },
        ]}
        layout={{
          showlegend: true,
          legend: { orientation: "h", y: 1.14 },
          xaxis: { title: { text: "season" }, dtick: 1 },
          yaxis: { title: { text: "market value (EUR)" }, rangemode: "tozero" },
          margin: { l: 75, r: 20, t: 34, b: 45 },
        }}
      />

      <div className="mt-4 flex flex-wrap gap-6 text-sm">
        <div>
          <div className="text-xs text-slate-500 uppercase dark:text-slate-400">
            Held-out seasons
          </div>
          <div className="tabular-nums">{heldOut.length}</div>
        </div>
        {meanAbsHeldOut != null && (
          <div>
            <div className="text-xs text-slate-500 uppercase dark:text-slate-400">
              Mean error, held out
            </div>
            <div className="tabular-nums">{eur(meanAbsHeldOut)}</div>
          </div>
        )}
      </div>

      <p className="mt-3 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        Hollow markers are seasons inside the training range — the model has
        already seen them, so landing close there is not evidence. Solid markers
        are held-out seasons, and those are the ones worth judging it on.
      </p>
    </Card>
  );
}
