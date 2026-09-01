"use client";

/**
 * Feature contributions for one prediction.
 *
 * The multiplicative reading is what is shown, not the raw SHAP value. SHAP
 * values here are additive in log space because the model fits log1p(EUR), so
 * "age contributed −€2M" would be arithmetically false — the same contribution
 * is worth a different number of euros for a €500k player and a €90M one.
 * "×0.72" is exact at any value.
 */

import type { Contribution } from "@/lib/api";
import { featureLabel } from "@/lib/format";
import Chart from "./Chart";
import { Card, Empty } from "./ui";

function Row({ contribution }: { contribution: Contribution }) {
  const up = contribution.shap_value > 0;
  const pct = (contribution.effect_multiplier - 1) * 100;

  return (
    <li className="flex items-baseline justify-between gap-3 border-b border-slate-100 py-2 last:border-0 dark:border-slate-800">
      <span className="text-sm text-slate-700 dark:text-slate-300">
        {featureLabel(contribution.feature)}
      </span>
      <span
        className={`font-mono text-sm tabular-nums ${
          up
            ? "text-emerald-600 dark:text-emerald-400"
            : "text-rose-600 dark:text-rose-400"
        }`}
      >
        ×{contribution.effect_multiplier.toFixed(2)}
        <span className="ml-2 text-xs opacity-70">
          {pct >= 0 ? "+" : ""}
          {pct.toFixed(0)}%
        </span>
      </span>
    </li>
  );
}

export default function Contributions({
  positive,
  negative,
}: {
  positive: Contribution[];
  negative: Contribution[];
}) {
  const all = [...positive, ...negative].sort(
    (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value),
  );

  if (all.length === 0) {
    return (
      <Card title="Feature contributions">
        <Empty>This model family cannot be explained with SHAP.</Empty>
      </Card>
    );
  }

  const ordered = [...all].reverse();

  return (
    <Card
      title="Feature contributions"
      subtitle="What moved this prediction, and by how much"
    >
      <Chart
        title="Feature contributions to this prediction"
        height={Math.max(220, ordered.length * 32)}
        data={[
          {
            type: "bar",
            orientation: "h",
            x: ordered.map((c) => c.shap_value),
            y: ordered.map((c) => featureLabel(c.feature)),
            marker: {
              color: ordered.map((c) =>
                c.shap_value > 0 ? "#059669" : "#e11d48",
              ),
            },
            hovertemplate:
              "%{y}<br>multiplies prediction by %{customdata:.2f}<extra></extra>",
            customdata: ordered.map((c) => c.effect_multiplier),
          },
        ]}
        layout={{
          margin: { l: 190, r: 20, t: 8, b: 40 },
          xaxis: {
            title: { text: "contribution (log space)" },
            zeroline: true,
          },
        }}
      />

      <div className="mt-4 grid gap-6 sm:grid-cols-2">
        <div>
          <h3 className="mb-1 text-xs font-semibold tracking-wide text-emerald-700 uppercase dark:text-emerald-400">
            Raises the value
          </h3>
          <ul>
            {positive.map((c) => (
              <Row key={c.feature} contribution={c} />
            ))}
          </ul>
        </div>
        <div>
          <h3 className="mb-1 text-xs font-semibold tracking-wide text-rose-700 uppercase dark:text-rose-400">
            Lowers the value
          </h3>
          <ul>
            {negative.map((c) => (
              <Row key={c.feature} contribution={c} />
            ))}
          </ul>
        </div>
      </div>

      <p className="mt-4 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        Contributions are multiplicative, not additive in euros. The model
        predicts log value, so ×1.51 means this feature raised the prediction by
        51% — whatever the player is worth. They cannot be summed into a euro
        figure.
      </p>
    </Card>
  );
}
