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
 * Every chart imports this wrapper, so the guard and the bundle choice exist
 * in exactly one file.
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
}

export default function Chart({ data, layout, height = 320, title }: ChartProps) {
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
      <Plot
        data={data}
        layout={merged}
        config={config}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </figure>
  );
}
