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
 *
 * **The fetcher must be stable.** It is the effect's only dependency, which is
 * what makes "the player changed, go and get the new one" work. Pass an inline
 * arrow and every render creates a new identity, the effect re-runs, and the
 * page fires requests at the API until the tab is closed — silently, and fast
 * enough that a test hit 27,000 of them. Wrap it in `useCallback` with the
 * values it depends on; every page here does. In development the hook now says
 * so out loud rather than leaving it to be discovered in a network tab.
 */

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * How many refetches in one mount before we assume the fetcher is unstable.
 * A page legitimately refetches when the player or the variant changes; it
 * does not do so twenty times without a user touching anything.
 */
const RUNAWAY_THRESHOLD = 20;

export type AsyncState<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; error: unknown };

export function useAsync<T>(fetcher: () => Promise<T>) {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" });
  const runaway = useRef(0);

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

  useEffect(() => {
    if (process.env.NODE_ENV !== "production") {
      runaway.current += 1;
      if (runaway.current === RUNAWAY_THRESHOLD) {
        // Once, not every iteration: a warning inside a runaway loop is itself
        // a runaway loop.
        console.error(
          `useAsync has refetched ${RUNAWAY_THRESHOLD} times. Its fetcher is ` +
            "changing identity on every render — wrap it in useCallback with " +
            "the values it depends on.",
        );
      }
    }
    return run();
  }, [run]);

  // An event handler, so setting state here is not an effect-time write.
  const reload = useCallback(() => {
    setState({ status: "loading" });
    run();
  }, [run]);

  return { state, reload };
}
