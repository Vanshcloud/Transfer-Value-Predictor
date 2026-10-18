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
 *    so it cannot describe a different chart from the one on screen: series
 *    as columns, axis titles as headers, histogram bins as rows.
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

/** Rows past this stop being an alternative and become their own navigation problem. */
const MAX_ROWS = 60;

/** Bins a histogram is drawn with when the caller does not choose them. */
const HISTOGRAM_BINS = 20;

type Trace = {
  type?: string;
  name?: string;
  x?: unknown[];
  y?: unknown[];
  orientation?: string;
  customdata?: unknown[];
  xbins?: { start: number; end: number; size: number };
};

/**
 * Give every histogram explicit bins, so the bars Plotly draws and the counts
 * the table reads are computed from the same edges. Plotly's own automatic
 * binning cannot be reproduced outside it, and a table of raw samples — what
 * this used to produce — is not a text alternative to a histogram at all.
 */
function withBins(data: Data[]): Data[] {
  return data.map((raw) => {
    const trace = raw as unknown as Trace;
    if (trace.type !== "histogram" || trace.xbins || !trace.x?.length) return raw;
    const values = trace.x.map(Number).filter(Number.isFinite);
    if (values.length === 0) return raw;
    const low = Math.min(...values);
    const high = Math.max(...values);
    const size = high > low ? (high - low) / HISTOGRAM_BINS : 1;
    return { ...raw, xbins: { start: low, end: high + size * 1e-9, size } } as Data;
  });
}

function binned(trace: Trace): [string, string][] {
  const { start, end, size } = trace.xbins!;
  const values = (trace.x ?? []).map(Number).filter(Number.isFinite);
  const count = Math.max(1, Math.round((end - start) / size));
  const counts = new Array<number>(count).fill(0);
  for (const value of values) {
    counts[Math.min(count - 1, Math.max(0, Math.floor((value - start) / size)))] += 1;
  }
  const edge = (value: number) => Number(value.toPrecision(3));
  return counts.map((n, i) => [`${edge(start + i * size)} to ${edge(start + (i + 1) * size)}`, String(n)]);
}

/** One trace's points as (category, value, note) — note from string `customdata`. */
function points(trace: Trace): [string, string, string | null][] {
  if (trace.type === "histogram" && trace.xbins) {
    return binned(trace).map(([label, value]) => [label, value, null]);
  }
  const horizontal = trace.orientation === "h";
  const labels = (horizontal ? trace.y : trace.x) ?? [];
  const values = (horizontal ? trace.x : trace.y) ?? [];
  const notes = trace.customdata?.every((item) => typeof item === "string")
    ? (trace.customdata as string[])
    : null;
  return labels.map((label, index) => [
    String(label),
    String(values[index] ?? ""),
    notes?.[index] ?? null,
  ]);
}

function axisTitle(axis: unknown): string | undefined {
  const title = (axis as { title?: string | { text?: string } } | undefined)?.title;
  return typeof title === "string" ? title : title?.text;
}

/**
 * The same numbers the chart draws, for anyone who cannot see it (audit F18).
 *
 * One row per category and one column per series, headed by the chart's own
 * axis titles and series names, so a reader learns which player, which line
 * and which unit each number belongs to — the first version flattened every
 * trace into one two-column list. A series whose points carry text
 * (`customdata` strings, e.g. "held out") gets a note column, because that is
 * how those charts mark what the picture marks with a hollow circle or a
 * colour. Capped at 60 rows.
 */
function DataTable({
  data,
  layout,
  title,
  valueLabel,
}: Pick<ChartProps, "data" | "layout" | "title" | "valueLabel">) {
  const traces = data.map((trace) => trace as unknown as Trace);
  const series = traces.map(points).filter((rows) => rows.length > 0);
  if (series.length === 0) return null;

  const horizontal = traces[0]?.orientation === "h";
  const histogram = traces[0]?.type === "histogram";
  const categoryHeader = axisTitle(horizontal ? layout?.yaxis : layout?.xaxis) ?? "Label";
  const valueAxis = axisTitle(horizontal ? layout?.xaxis : layout?.yaxis);
  const single = series.length === 1;

  const columns = traces
    .map((trace, index) => ({ trace, rows: points(trace), index }))
    .filter(({ rows }) => rows.length > 0)
    .map(({ trace, rows, index }) => ({
      header: single
        ? (valueLabel ?? valueAxis ?? (histogram ? "count" : "Value"))
        : (trace.name ?? `Series ${index + 1}`),
      values: new Map(rows.map(([label, value]) => [label, value])),
      notes: rows.some(([, , note]) => note != null)
        ? new Map(rows.map(([label, , note]) => [label, note ?? ""]))
        : null,
    }));

  const categories = [...new Set(series.flatMap((rows) => rows.map(([label]) => label)))].slice(
    0,
    MAX_ROWS,
  );

  return (
    <figcaption className="sr-only">
      <table>
        <caption>
          {title} — data table
          {!single && valueAxis ? `. Values are ${valueAxis}.` : ""}
          {histogram ? " Each row is one bin and the number of values in it." : ""}
        </caption>
        <thead>
          <tr>
            <th scope="col">{categoryHeader}</th>
            {columns.flatMap((column) => [
              <th scope="col" key={column.header}>
                {column.header}
              </th>,
              ...(column.notes
                ? [
                    <th scope="col" key={`${column.header}-note`}>
                      {column.header} — note
                    </th>,
                  ]
                : []),
            ])}
          </tr>
        </thead>
        <tbody>
          {categories.map((category) => (
            <tr key={category}>
              <th scope="row">{category}</th>
              {columns.flatMap((column) => [
                <td key={column.header}>{column.values.get(category) ?? ""}</td>,
                ...(column.notes
                  ? [<td key={`${column.header}-note`}>{column.notes.get(category) ?? ""}</td>]
                  : []),
              ])}
            </tr>
          ))}
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
  const binnedData = withBins(data);
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
          data={binnedData}
          layout={merged}
          config={config}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>
      <DataTable data={binnedData} layout={layout} title={title} valueLabel={valueLabel} />
    </figure>
  );
}
