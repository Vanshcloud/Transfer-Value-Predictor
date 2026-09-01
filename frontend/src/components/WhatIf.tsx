"use client";

/**
 * What-if: change the inputs, watch the prediction move.
 *
 * The form is seeded from the player's real feature values (the API returns
 * them for exactly this reason), so "goals 10 → 18" starts from something
 * true rather than a guess. Only the handful of features a person can reason
 * about are editable; the rest travel unchanged so the counterfactual differs
 * from the real player in exactly the ways the user chose.
 *
 * Derived features are deliberately NOT recomputed here. goals_per_90 is a
 * function of goals and minutes, and quietly recomputing it in the browser
 * would mean two implementations of the feature definition — the real one in
 * src/feature_engineering and a JavaScript copy that drifts. The panel says
 * which fields move together instead.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError, type PredictResponse, type Variant } from "@/lib/api";
import { eur, eurExact } from "@/lib/format";
import { Card, ErrorPanel } from "./ui";

/** Editable inputs, with the bounds the training data actually spans. */
const CONTROLS: { key: string; label: string; min: number; max: number; step: number }[] = [
  { key: "age", label: "Age", min: 16, max: 40, step: 0.5 },
  { key: "goals", label: "Goals", min: 0, max: 50, step: 1 },
  { key: "assists", label: "Assists", min: 0, max: 30, step: 1 },
  { key: "appearances", label: "Appearances", min: 0, max: 60, step: 1 },
  { key: "minutes_played", label: "Minutes played", min: 0, max: 4500, step: 50 },
  { key: "yellow_cards", label: "Yellow cards", min: 0, max: 20, step: 1 },
];

function toNumber(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export default function WhatIf({
  baseline,
  features,
  variant,
}: {
  baseline: number;
  features: Record<string, unknown>;
  variant: Variant;
}) {
  const editable = useMemo(
    () => CONTROLS.filter((control) => control.key in features),
    [features],
  );

  const initial = useMemo(() => {
    const seed: Record<string, number> = {};
    for (const control of editable) seed[control.key] = toNumber(features[control.key]);
    return seed;
  }, [editable, features]);

  const [values, setValues] = useState<Record<string, number>>(initial);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [pending, setPending] = useState(false);

  const run = useCallback(async () => {
    setPending(true);
    setError(null);
    try {
      setResult(await api.predictFromFeatures({ ...features, ...values }, variant));
    } catch (caught) {
      setError(caught);
    } finally {
      setPending(false);
    }
  }, [features, values, variant]);

  const touched = editable.some((c) => values[c.key] !== initial[c.key]);

  // Debounced: a slider drag fires dozens of changes and each one is a model
  // call. 300ms turns a drag into one request without feeling laggy.
  useEffect(() => {
    if (!touched) return;
    const timer = setTimeout(run, 300);
    return () => clearTimeout(timer);
  }, [touched, run]);

  const predicted = touched ? (result?.prediction_eur ?? baseline) : baseline;
  const delta = predicted - baseline;
  const ratio = baseline > 0 ? predicted / baseline : 1;

  return (
    <Card
      title="What if?"
      subtitle="Change the season, and see how the valuation responds"
    >
      <div className="grid gap-6 lg:grid-cols-[1fr_16rem]">
        <div className="space-y-4">
          {editable.map((control) => {
            const value = values[control.key] ?? 0;
            const original = initial[control.key] ?? 0;
            const moved = value !== original;
            return (
              <div key={control.key}>
                <label
                  htmlFor={`whatif-${control.key}`}
                  className="flex items-baseline justify-between text-sm"
                >
                  <span className="text-slate-700 dark:text-slate-300">{control.label}</span>
                  <span className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                    {value}
                    {moved && (
                      <span className="ml-2 text-xs text-slate-400">was {original}</span>
                    )}
                  </span>
                </label>
                <input
                  id={`whatif-${control.key}`}
                  type="range"
                  min={control.min}
                  max={control.max}
                  step={control.step}
                  value={value}
                  onChange={(event) =>
                    setValues((previous) => ({
                      ...previous,
                      [control.key]: Number(event.target.value),
                    }))
                  }
                  className="mt-1 w-full accent-sky-600"
                />
              </div>
            );
          })}

          <button
            onClick={() => setValues(initial)}
            disabled={!touched}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm disabled:opacity-40 dark:border-slate-700"
          >
            Reset to actual season
          </button>
        </div>

        <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
          <div className="text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
            {touched ? "Counterfactual" : "Actual season"}
          </div>
          <div
            className="mt-1 text-3xl font-semibold tabular-nums"
            aria-live="polite"
            aria-busy={pending}
          >
            {eur(predicted)}
          </div>
          <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            {eurExact(predicted)}
          </div>

          {touched && (
            <div className="mt-4 border-t border-slate-100 pt-3 dark:border-slate-800">
              <div className="text-xs text-slate-500 dark:text-slate-400">
                against {eur(baseline)}
              </div>
              <div
                className={`mt-1 font-mono text-lg tabular-nums ${
                  delta >= 0
                    ? "text-emerald-600 dark:text-emerald-400"
                    : "text-rose-600 dark:text-rose-400"
                }`}
              >
                {delta >= 0 ? "+" : ""}
                {eur(delta)}
                <span className="ml-2 text-sm opacity-70">×{ratio.toFixed(2)}</span>
              </div>
            </div>
          )}

          {pending && (
            <div className="mt-3 text-xs text-slate-400">predicting…</div>
          )}
        </div>
      </div>

      {error != null && (
        <div className="mt-4">
          <ErrorPanel
            error={
              error instanceof ApiError
                ? error
                : new Error("What-if prediction failed")
            }
            onRetry={run}
          />
        </div>
      )}

      <p className="mt-4 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        Derived features such as goals per 90 are <strong>not</strong> recomputed as
        you drag: they are defined once in the feature pipeline, and a second
        definition living in the browser would eventually disagree with it. So
        raising goals here shows the effect of goals alone, holding the per-90
        rates at the real season&apos;s values.
      </p>
    </Card>
  );
}
