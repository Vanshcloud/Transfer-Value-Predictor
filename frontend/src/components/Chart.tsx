"use client";

/**
 * The only place this project touches Plotly.
 *
 * Two traps live here so no page has to know about them:
 *
 * 1. **SSR.** A plain `import Plot from "react-plotly.js"` fails `next build`
 *    with `ReferenceError: self is not defined`, even inside a client
 *    component, because client components are prerendered on the server.
 *    `next/dynamic` with `ssr: false` is the fix.
 *
 * 2. **Bundle size.** Full `plotly.js` is 4028 KB. The cartesian-only
 *    distribution through `react-plotly.js/factory` is 1412 KB and draws every
 *    chart this dashboard needs. `@types/react-plotly.js` is deliberately NOT
 *    installed — v4 ships its own types, and the DefinitelyTyped package
 *    conflicts with them.
 *
 * 3. **Accessibility.** Plotly renders an SVG of positioned shapes. A screen
 *    reader gets an `aria-label` and nothing else — the numbers, which are the
 *    entire content, are unreachable. Every chart therefore also renders a
 *    visually-hidden table of its own data, derived from the same `data` prop
 *    so it cannot describe a different chart from the one on screen.
 *
 * Every chart imports this wrapper, so all three exist in exactly one file.
 */

import dynamic from "next/dynamic";
import type { Data, Layout, Config } from "plotly.js";
import { useIsDark } from "./ThemeToggle";

const Plot = dynamic(
  async () => {
    const [{ default: createPlotly }, Plotly] = await Promise.all([
      import("react-plotly.js/factory"),
      import("plotly.js-cartesian-dist-min"),
    ]);
    return createPlotly(Plotly as never);
  },
  {
    ssr: false,
    loading: () => <ChartSkeleton />,
  },
);

export function ChartSkeleton({ height = 320 }: { height?: number }) {
  return (
    <div
      style={{ height }}
      className="w-full animate-pulse rounded-lg bg-slate-100 dark:bg-slate-800"
      aria-hidden
    />
  );
}

export interface ChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
  /** Accessible description of what the chart shows. */
  title: string;
  /**
   * Units for the hidden data table's value column, e.g. "EUR" or "%".
   * Only affects the text alternative; the visible chart carries its own axes.
   */
  valueLabel?: string;
}

/** A trace's points as (label, value) pairs, for the text alternative. */
function points(trace: Data): [string, string][] {
  const record = trace as unknown as { x?: unknown[]; y?: unknown[]; orientation?: string };
  const horizontal = record.orientation === "h";
  const labels = (horizontal ? record.y : record.x) ?? [];
  const values = (horizontal ? record.x : record.y) ?? [];
  return labels
    .slice(0, 60)
    .map((label, index): [string, string] => [String(label), String(values[index] ?? "")]);
}

/**
 * The same numbers the chart draws, for anyone who cannot see it.
 *
 * Capped at 60 rows per trace: past that a table stops being an alternative
 * and becomes its own navigation problem, and no chart here plots more.
 */
function DataTable({ data, title, valueLabel }: Pick<ChartProps, "data" | "title" | "valueLabel">) {
  const traces = data.map(points).filter((rows) => rows.length > 0);
  if (traces.length === 0) return null;

  return (
    <figcaption className="sr-only">
      <table>
        <caption>{title} — data table</caption>
        <thead>
          <tr>
            <th scope="col">Label</th>
            <th scope="col">{valueLabel ?? "Value"}</th>
          </tr>
        </thead>
        <tbody>
          {traces.flatMap((rows, traceIndex) =>
            rows.map(([label, value]) => (
              <tr key={`${traceIndex}-${label}`}>
                <th scope="row">{label}</th>
                <td>{value}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </figcaption>
  );
}

export default function Chart({
  data,
  layout,
  height = 320,
  title,
  valueLabel,
}: ChartProps) {
  const dark = useIsDark();
  const ink = dark ? "#cbd5e1" : "#334155";
  const grid = dark ? "#334155" : "#e2e8f0";

  const merged: Partial<Layout> = {
    autosize: true,
    height,
    margin: { l: 60, r: 20, t: 10, b: 48 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: ink, size: 12 },
    xaxis: { gridcolor: grid, zerolinecolor: grid, ...layout?.xaxis },
    yaxis: { gridcolor: grid, zerolinecolor: grid, ...layout?.yaxis },
    showlegend: false,
    hoverlabel: { bgcolor: dark ? "#1e293b" : "#ffffff", font: { color: ink } },
    ...layout,
  };

  const config: Partial<Config> = {
    displayModeBar: false,
    responsive: true,
  };

  return (
    <figure className="w-full" aria-label={title}>
      {/* The SVG is decorative once the table below carries the same numbers;
          without this a screen reader reads a wall of unlabelled path data. */}
      <div aria-hidden="true">
        <Plot
          data={data}
          layout={merged}
          config={config}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>
      <DataTable data={data} title={title} valueLabel={valueLabel} />
    </figure>
  );
}
