"use client";

/**
 * A radar comparison, drawn as inline SVG.
 *
 * Not Plotly: `scatterpolar` lives in the polar trace module, which the slim
 * cartesian bundle does not register. Getting it would mean shipping full
 * plotly.js — 4 MB against the 1.4 MB the whole dashboard currently costs —
 * for one chart type. Radar geometry is a sine and a cosine, so this is
 * seventy lines instead of two and a half megabytes.
 *
 * Axes are **population percentiles**, not raw values. Normalising against
 * only the two players on screen would peg one of them at the outer ring on
 * every axis and communicate nothing; a percentile says "92nd of everyone",
 * which is a fact about the panel rather than about the comparison.
 */

import type { FeatureDistribution } from "@/lib/api";
import { featureLabel } from "@/lib/format";

/** The axes worth comparing: rate stats and volume, not identifiers. */
export const RADAR_FEATURES = [
  "goals_per_90",
  "assists_per_90",
  "minutes_played",
  "appearances",
  "minutes_per_appearance",
  "years_since_debut",
] as const;

/** Where `value` sits in the population, from the quantile grid, by interpolation. */
export function percentileOf(
  value: number,
  quantiles: number[],
  grid: number[],
): number {
  if (quantiles.length === 0) return 0;
  if (value <= quantiles[0]) return 0;
  if (value >= quantiles[quantiles.length - 1]) return 1;

  for (let i = 1; i < quantiles.length; i += 1) {
    if (value <= quantiles[i]) {
      const span = quantiles[i] - quantiles[i - 1];
      const within = span === 0 ? 0 : (value - quantiles[i - 1]) / span;
      return grid[i - 1] + within * (grid[i] - grid[i - 1]);
    }
  }
  return 1;
}

export interface RadarSeries {
  label: string;
  colour: string;
  features: Record<string, unknown>;
}

function point(cx: number, cy: number, radius: number, index: number, count: number) {
  // Start at twelve o'clock and go clockwise, which is how people read a dial.
  const angle = (Math.PI * 2 * index) / count - Math.PI / 2;
  return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)] as const;
}

export default function Radar({
  series,
  distribution,
  size = 340,
}: {
  series: RadarSeries[];
  distribution: FeatureDistribution;
  size?: number;
}) {
  const axes = RADAR_FEATURES.filter((name) => name in distribution.features);
  if (axes.length < 3) return null;

  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 58;
  const rings = [0.25, 0.5, 0.75, 1];

  return (
    <div className="flex flex-col items-center">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        width="100%"
        style={{ maxWidth: size }}
        role="img"
        aria-label={`Percentile comparison across ${axes.length} metrics`}
      >
        {rings.map((ring) => (
          <polygon
            key={ring}
            points={axes
              .map((_, index) => point(cx, cy, radius * ring, index, axes.length).join(","))
              .join(" ")}
            className="fill-none stroke-slate-200 dark:stroke-slate-700"
            strokeWidth={1}
          />
        ))}

        {axes.map((_, index) => {
          const [x, y] = point(cx, cy, radius, index, axes.length);
          return (
            <line
              key={index}
              x1={cx}
              y1={cy}
              x2={x}
              y2={y}
              className="stroke-slate-200 dark:stroke-slate-700"
              strokeWidth={1}
            />
          );
        })}

        {series.map((entry) => {
          const points = axes.map((name, index) => {
            const feature = distribution.features[name];
            const raw = Number(entry.features[name]);
            const percentile = Number.isFinite(raw)
              ? percentileOf(raw, feature.quantiles, distribution.grid)
              : 0;
            return point(cx, cy, radius * percentile, index, axes.length);
          });

          return (
            <polygon
              key={entry.label}
              points={points.map((p) => p.join(",")).join(" ")}
              fill={entry.colour}
              fillOpacity={0.22}
              stroke={entry.colour}
              strokeWidth={2}
            />
          );
        })}

        {axes.map((name, index) => {
          const [x, y] = point(cx, cy, radius + 22, index, axes.length);
          return (
            <text
              key={name}
              x={x}
              y={y}
              textAnchor={Math.abs(x - cx) < 4 ? "middle" : x > cx ? "start" : "end"}
              dominantBaseline="middle"
              className="fill-slate-600 text-[10px] dark:fill-slate-300"
            >
              {featureLabel(name)}
            </text>
          );
        })}
      </svg>

      <div className="mt-2 flex flex-wrap justify-center gap-4">
        {series.map((entry) => (
          <span key={entry.label} className="flex items-center gap-2 text-sm">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ backgroundColor: entry.colour }}
            />
            {entry.label}
          </span>
        ))}
      </div>

      <p className="mt-3 max-w-md text-center text-xs text-slate-500 dark:text-slate-400">
        Each axis is a percentile across all {distribution.features[axes[0]]?.n.toLocaleString()}{" "}
        player-seasons, so the outer ring means best in the panel — not merely
        better than the other player here.
      </p>
    </div>
  );
}
