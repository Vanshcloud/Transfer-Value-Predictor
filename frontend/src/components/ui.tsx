"use client";

/**
 * The small shared pieces: cards, stats, loading and error states.
 *
 * Loading and error states live here rather than being improvised per page,
 * because the improvised version is the one that gets skipped — and a
 * dashboard that shows nothing while it waits, then a blank panel when the
 * backend is down, is indistinguishable from a broken one.
 */

import type { ReactNode } from "react";
import { ApiError } from "@/lib/api";

export function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900/60 ${className}`}
    >
      {title && (
        <header className="mb-4">
          <h2 className="text-sm font-semibold tracking-wide text-slate-900 uppercase dark:text-slate-100">
            {title}
          </h2>
          {subtitle && (
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{subtitle}</p>
          )}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
}) {
  return (
    <div className="min-w-[8rem]">
      <div className="text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-slate-900 dark:text-slate-50">
        {value}
      </div>
      {hint && <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{hint}</div>}
    </div>
  );
}

export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded bg-slate-200 dark:bg-slate-800 ${className}`}
      aria-hidden
    />
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="space-y-3" role="status" aria-live="polite">
      <span className="sr-only">{label}</span>
      <Skeleton className="h-4 w-1/3" />
      <Skeleton className="h-4 w-2/3" />
      <Skeleton className="h-4 w-1/2" />
    </div>
  );
}

/**
 * An error a reader can act on.
 *
 * A dead backend is the common case in development, so it gets its own
 * message naming the command that fixes it rather than "Failed to fetch".
 */
export function ErrorPanel({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const isApi = error instanceof ApiError;
  const code = isApi ? error.code : "unexpected_error";
  const message = error instanceof Error ? error.message : String(error);

  return (
    <div
      role="alert"
      className="rounded-xl border border-red-200 bg-red-50 p-5 dark:border-red-900/50 dark:bg-red-950/30"
    >
      <p className="font-medium text-red-900 dark:text-red-200">{message}</p>
      <p className="mt-1 font-mono text-xs text-red-700/80 dark:text-red-300/70">{code}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 rounded-lg border border-red-300 px-3 py-1.5 text-sm font-medium text-red-900 hover:bg-red-100 dark:border-red-800 dark:text-red-200 dark:hover:bg-red-900/40"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <p className="py-8 text-center text-sm text-slate-500 dark:text-slate-400">{children}</p>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "positive" | "negative" | "warn";
}) {
  const tones = {
    neutral: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    positive: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
    negative: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
    warn: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-300",
  } as const;
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

/**
 * An initials disc standing in for a player portrait.
 *
 * The CC0 dataset carries an `image_url`, but it points at Transfermarkt's own
 * CDN — the licence covers the table, not the photographs behind it, and this
 * project does not call Transfermarkt. Initials cost one request fewer than a
 * portrait and make no claim about a licence nobody granted.
 */
export function Avatar({ name, seed }: { name: string; seed: number }) {
  const initials =
    name
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part.charAt(0).toUpperCase())
      .join("") || "?";
  // Golden angle, so adjacent ids do not land on near-identical hues.
  const hue = Math.round((seed * 137.508) % 360);
  // 55%/30% is the point where white text clears WCAG AA (4.5:1) on *every*
  // hue — 45%/38% left the yellow-greens at 3.53:1, which is 40% of players.
  return (
    <span
      aria-hidden
      style={{ backgroundColor: `hsl(${hue} 55% 30%)` }}
      className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-xs font-semibold text-white"
    >
      {initials}
    </span>
  );
}
