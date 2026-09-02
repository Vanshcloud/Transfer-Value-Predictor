"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api, type SearchResult } from "@/lib/api";
import { eur } from "@/lib/format";
import { Avatar, Badge, Card, ClubTag, Empty, ErrorPanel, Loading } from "@/components/ui";

const EXAMPLES = ["Haaland", "Bellingham", "Vinicius", "Rodri", "Saka"];

export default function PlayerSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [pending, setPending] = useState(false);

  const search = useCallback(async (term: string) => {
    if (!term.trim()) {
      setResults(null);
      return;
    }
    setPending(true);
    setError(null);
    try {
      setResults((await api.searchPlayers(term)).results);
    } catch (caught) {
      setError(caught);
      setResults(null);
    } finally {
      setPending(false);
    }
  }, []);

  // Debounced so typing does not fire a request per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => search(query), 250);
    return () => clearTimeout(timer);
  }, [query, search]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Find a player</h1>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          Search by name. Players the model can predict for are listed first.
        </p>
      </div>

      <div>
        <label htmlFor="player-search" className="sr-only">
          Player name
        </label>
        <input
          id="player-search"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="e.g. Haaland"
          autoComplete="off"
          className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-lg shadow-sm placeholder:text-slate-400 focus:border-sky-500 dark:border-slate-700 dark:bg-slate-900"
        />
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="text-xs text-slate-500 dark:text-slate-400">Try:</span>
          {EXAMPLES.map((example) => (
            <button
              key={example}
              onClick={() => setQuery(example)}
              className="rounded-full border border-slate-200 px-2.5 py-0.5 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {error != null && <ErrorPanel error={error} onRetry={() => search(query)} />}

      {pending && <Card><Loading label="Searching" /></Card>}

      {!pending && results != null && (
        <Card>
          {results.length === 0 ? (
            <Empty>No player matches “{query}”.</Empty>
          ) : (
            <ul className="divide-y divide-slate-100 dark:divide-slate-800">
              {results.map((result) => {
                const inner = (
                  <div className="flex items-center justify-between gap-4 py-3">
                    <div className="flex min-w-0 items-center gap-3">
                      <Avatar name={result.name} seed={result.player_id} />
                      <div className="min-w-0">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="truncate font-medium">{result.name}</span>
                          <ClubTag
                            club={result.club}
                            league={result.league}
                            country={result.league_country}
                          />
                          {!result.predictable && <Badge tone="warn">no modelled season</Badge>}
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400">
                          {result.position ?? "—"}
                          {result.latest_season && ` · latest ${result.latest_season}`}
                        </div>
                      </div>
                    </div>
                    <span className="shrink-0 text-sm tabular-nums text-slate-600 dark:text-slate-300">
                      {eur(result.market_value_in_eur)}
                    </span>
                  </div>
                );

                return (
                  <li key={result.player_id}>
                    {result.predictable ? (
                      <Link
                        href={`/players/${result.player_id}`}
                        className="block px-2 hover:bg-slate-50 dark:hover:bg-slate-800/50"
                      >
                        {inner}
                      </Link>
                    ) : (
                      <div className="px-2 opacity-60">{inner}</div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </Card>
      )}
    </div>
  );
}
