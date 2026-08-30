/**
 * The race guard is the reason this hook exists rather than a useEffect in each
 * page: navigate from one player to another while the first request is in
 * flight, and without the guard the slow response overwrites the fast one and
 * the page shows the wrong player's valuation with the right player's name.
 *
 * That is not a bug a type checker or a build can find, and it is invisible on
 * a fast connection — which is exactly why it is worth a test.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useAsync } from "./useAsync";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("useAsync", () => {
  it("starts loading and never renders data and an error together", async () => {
    const fetcher = () => Promise.resolve("value");
    const { result } = renderHook(() => useAsync(fetcher));
    expect(result.current.state.status).toBe("loading");
    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(result.current.state).toEqual({ status: "ready", data: "value" });
  });

  it("reports a rejection as an error state rather than throwing", async () => {
    const boom = new Error("boom");
    const fetcher = () => Promise.reject(boom);
    const { result } = renderHook(() => useAsync(fetcher));
    await waitFor(() => expect(result.current.state.status).toBe("error"));
    expect(result.current.state).toEqual({ status: "error", error: boom });
  });

  it("ignores a response for a fetcher it has already moved on from", async () => {
    // The whole point of the hook. The stale resolve must not win.
    const stale = deferred<string>();
    const fresh = deferred<string>();
    const first = () => stale.promise;
    const second = () => fresh.promise;

    const { result, rerender } = renderHook(({ fetcher }) => useAsync(fetcher), {
      initialProps: { fetcher: first },
    });

    rerender({ fetcher: second });
    await act(async () => {
      fresh.resolve("current player");
      stale.resolve("previous player");
    });

    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(result.current.state).toEqual({ status: "ready", data: "current player" });
  });

  it("does not report an error from a cancelled request", async () => {
    const stale = deferred<string>();
    const fresh = deferred<string>();

    const { result, rerender } = renderHook(({ fetcher }) => useAsync(fetcher), {
      initialProps: { fetcher: () => stale.promise },
    });

    rerender({ fetcher: () => fresh.promise });
    await act(async () => {
      fresh.resolve("ok");
      stale.reject(new Error("aborted navigation"));
    });

    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(result.current.state.status).not.toBe("error");
  });

  it("reload returns to loading and then to the new value", async () => {
    let call = 0;
    // Stable identity, as every real call site uses. An inline arrow here is
    // the runaway case the hook now warns about, and is tested below.
    const fetcher = () => Promise.resolve(`call ${++call}`);
    const { result } = renderHook(() => useAsync(fetcher));
    await waitFor(() => expect(result.current.state.status).toBe("ready"));

    act(() => result.current.reload());
    await waitFor(() => expect(result.current.state).toEqual({ status: "ready", data: "call 2" }));
  });

  it("names the mistake when the fetcher is not stable", async () => {
    // An unstable fetcher makes the effect re-run forever. Silently, and fast:
    // this loop reached 27,000 requests before anything noticed. One console
    // error naming useCallback is the difference between a five-minute fix and
    // an afternoon in a network tab.
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const { unmount } = renderHook(() => useAsync(() => Promise.resolve("value")));

    await waitFor(() =>
      expect(error).toHaveBeenCalledWith(expect.stringContaining("useCallback")),
    );
    // Said once, not once per iteration.
    const complaints = error.mock.calls.filter((call) =>
      String(call[0]).includes("useAsync has refetched"),
    );
    expect(complaints).toHaveLength(1);
    unmount();
  });

  it("refetches when the fetcher identity changes, and only then", async () => {
    const fetcher = vi.fn().mockResolvedValue("value");
    const { result, rerender } = renderHook(({ f }) => useAsync(f), {
      initialProps: { f: fetcher },
    });
    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(fetcher).toHaveBeenCalledTimes(1);

    // Same reference: a re-render must not fire a second request.
    rerender({ f: fetcher });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
