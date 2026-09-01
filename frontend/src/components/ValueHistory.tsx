"use client";

import type { SeasonRow } from "@/lib/api";
import { eur } from "@/lib/format";
import Chart from "./Chart";
import { Card, Empty } from "./ui";

/** Recorded value per season, with goals on a second axis for context. */
export default function ValueHistory({ seasons }: { seasons: SeasonRow[] }) {
  // The season in progress has no recorded value yet. Plotting it as 0 would
  // draw a cliff at the right-hand edge of every chart and read as a collapse
  // in value; the honest chart simply stops where the record stops.
  const recorded = seasons.filter(
    (s): s is SeasonRow & { market_value_in_eur: number } =>
      s.has_label !== false && s.market_value_in_eur !== null,
  );
  if (recorded.length === 0) {
    return (
      <Card title="History">
        <Empty>No seasons on record.</Empty>
      </Card>
    );
  }

  return (
    <Card title="History" subtitle="Recorded market value by season">
      <Chart
        title="Market value by season"
        height={280}
        // One axis, not two. A secondary axis puts goals and euros on
        // scales chosen by nothing in particular, and the reader is invited
        // to see a relationship in whatever the scaling happens to produce.
        // Goals travel in the hover instead, where they inform without
        // implying.
        data={[
          {
            type: "scatter",
            mode: "lines+markers",
            name: "market value",
            x: recorded.map((s) => s.season),
            y: recorded.map((s) => s.market_value_in_eur),
            line: { color: "#0284c7", width: 2 },
            marker: { size: 8 },
            customdata: recorded.map((s) => [s.goals, s.appearances, s.minutes_played]),
            hovertemplate:
              "<b>%{x}</b><br>%{y:,.0f} EUR<br>" +
              "%{customdata[0]} goals in %{customdata[1]} apps<br>" +
              "%{customdata[2]:,} minutes<extra></extra>",
          },
        ]}
        layout={{
          xaxis: { title: { text: "season" }, dtick: 1 },
          yaxis: { title: { text: "market value (EUR)" }, rangemode: "tozero" },
          margin: { l: 75, r: 20, t: 16, b: 45 },
        }}
      />
      <dl className="mt-4 flex flex-wrap gap-6 text-sm">
        <div>
          <dt className="text-xs text-slate-500 uppercase dark:text-slate-400">Peak</dt>
          <dd className="tabular-nums">
            {eur(Math.max(...recorded.map((s) => s.market_value_in_eur)))}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500 uppercase dark:text-slate-400">Seasons</dt>
          <dd className="tabular-nums">{seasons.length}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500 uppercase dark:text-slate-400">
            Total minutes
          </dt>
          <dd className="tabular-nums">
            {seasons.reduce((total, s) => total + s.minutes_played, 0).toLocaleString()}
          </dd>
        </div>
      </dl>
    </Card>
  );
}
