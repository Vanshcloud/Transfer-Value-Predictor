/**
 * Audit F18: the radar is inline SVG, so its numbers need a text alternative
 * of their own — the aria-label said only how many metrics there were.
 */
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Radar, { RADAR_FEATURES } from "./Radar";
import type { FeatureDistribution } from "@/lib/api";

const grid = [0, 0.25, 0.5, 0.75, 1];
const distribution: FeatureDistribution = {
  variant: "performance_only",
  grid,
  features: Object.fromEntries(
    RADAR_FEATURES.map((name) => [name, { quantiles: [0, 1, 2, 3, 4], median: 2, n: 1000 }]),
  ),
} as FeatureDistribution;

describe("Radar", () => {
  it("states each player's percentile and value per metric", () => {
    render(
      <Radar
        distribution={distribution}
        series={[
          { label: "Player A", colour: "#000", features: { goals_per_90: 3 } },
          { label: "Player B", colour: "#fff", features: { goals_per_90: 1 } },
        ]}
      />,
    );
    const table = screen.getByRole("table");
    expect(within(table).getByRole("columnheader", { name: "Player A" })).toBeInTheDocument();
    const row = within(table).getByRole("row", { name: /goals per 90/i });
    expect(row).toHaveTextContent("75th percentile (3)");
    expect(row).toHaveTextContent("25th percentile (1)");
  });

  it("says a missing value is missing rather than the bottom of the panel", () => {
    render(
      <Radar
        distribution={distribution}
        series={[{ label: "Player A", colour: "#000", features: {} }]}
      />,
    );
    const row = within(screen.getByRole("table")).getByRole("row", { name: /goals per 90/i });
    expect(row).toHaveTextContent("not recorded");
    expect(row).not.toHaveTextContent("percentile");
  });
});
