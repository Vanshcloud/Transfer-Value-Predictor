"use client";

import Link from "next/link";
import type { SimilarPlayer } from "@/lib/api";
import { eur } from "@/lib/format";
import { Card, Empty } from "./ui";

/**
 * Comparable seasons, measured in the model's own feature space rather than on
 * a hand-picked pair of columns — so "similar" means similar to the thing
 * making the prediction.
 */
export default function SimilarPlayers({
  players,
  season,
}: {
  players: SimilarPlayer[];
  season: number | null;
}) {
  return (
    <Card
      title="Similar seasons"
      subtitle={
        season
          ? `Closest ${season} seasons in the model's feature space`
          : "Closest seasons in the model's feature space"
      }
    >
      {players.length === 0 ? (
        <Empty>No comparable seasons found.</Empty>
      ) : (
        <ul className="divide-y divide-slate-100 dark:divide-slate-800">
          {players.map((player) => (
            <li key={`${player.player_id}-${player.season}`}>
              <Link
                href={`/players/${player.player_id}`}
                className="flex items-center justify-between gap-4 py-2.5 hover:bg-slate-50 dark:hover:bg-slate-800/50"
              >
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium">
                    {player.name ?? `Player ${player.player_id}`}
                  </span>
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    {player.position ?? "—"} · age {player.age.toFixed(0)}
                  </span>
                </span>
                <span className="text-right">
                  <span className="block text-sm tabular-nums">
                    {eur(player.market_value_in_eur)}
                  </span>
                  <span className="text-xs text-slate-400">
                    d {player.distance.toFixed(2)}
                  </span>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
        Distance is measured on the preprocessed features the model sees.
        Comparisons stay within one season — market conditions differ across
        years.
      </p>
    </Card>
  );
}
