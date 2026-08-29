"use client";

/**
 * Fetch-on-mount, without setting state synchronously inside the effect.
 *
 * React 19 flags `setState` called directly in an effect body, and it is right
 * to: a component that mounts with `loading: true` and then immediately sets
 * `loading: true` again has rendered twice to say the same thing. Every state
 * transition here happens in a promise continuation or an event handler
 * instead.
 *
 * One state object rather than three booleans, so "loading and also holding a
 * stale error" cannot be represented at all.
 */

import { useCallback, useEffect, useState } from "react";

export type AsyncState<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; error: unknown };

export function useAsync<T>(fetcher: () => Promise<T>) {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" });

  const run = useCallback(() => {
    let cancelled = false;
    fetcher().then(
      (data) => {
        // Guarded so a slow response for a player the user has already
        // navigated away from cannot overwrite the current one.
        if (!cancelled) setState({ status: "ready", data });
      },
      (error) => {
        if (!cancelled) setState({ status: "error", error });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [fetcher]);

  useEffect(() => run(), [run]);

  // An event handler, so setting state here is not an effect-time write.
  const reload = useCallback(() => {
    setState({ status: "loading" });
    run();
  }, [run]);

  return { state, reload };
}
