/**
 * The chart's text alternative.
 *
 * Plotly draws an SVG of positioned shapes: a screen reader gets the
 * aria-label and no numbers at all, and the numbers are the entire content.
 * The hidden table is derived from the same `data` prop the chart renders, so
 * these tests are really asserting that the two cannot diverge.
 */
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// next/dynamic resolves asynchronously and never resolves in jsdom, so the
// Plot itself is stubbed. The SSR guard it exists for is caught by `next
// build` in CI, which is where that failure actually appears.
vi.mock("next/dynamic", () => ({
  default: () => function StubbedPlot() {
    return <div data-testid="plot" />;
  },
}));

const { default: Chart } = await import("./Chart");

const bars = [
  {
    type: "bar" as const,
    orientation: "h" as const,
    y: ["appearances", "goals"],
    x: [0.657, 0.3],
  },
];

describe("Chart", () => {
  it("names the figure for a screen reader", () => {
    render(<Chart data={bars} title="Feature contributions" />);
    expect(screen.getByRole("figure", { name: "Feature contributions" })).toBeInTheDocument();
  });

  it("exposes the same numbers as a table, not just as pixels", () => {
    render(<Chart data={bars} title="Feature contributions" />);
    const table = screen.getByRole("table");
    expect(within(table).getByRole("rowheader", { name: "appearances" })).toBeInTheDocument();
    expect(within(table).getByText("0.657")).toBeInTheDocument();
    expect(within(table).getByText("0.3")).toBeInTheDocument();
  });

  it("reads a vertical trace off x/y the other way round", () => {
    render(
      <Chart data={[{ type: "scatter", x: [2023, 2024], y: [12, 19] }]} title="Value history" />,
    );
    const table = screen.getByRole("table");
    expect(within(table).getByRole("rowheader", { name: "2023" })).toBeInTheDocument();
    expect(within(table).getByText("19")).toBeInTheDocument();
  });

  it("labels the value column when units are given", () => {
    render(<Chart data={bars} title="Value" valueLabel="EUR" />);
    expect(screen.getByRole("columnheader", { name: "EUR" })).toBeInTheDocument();
  });

  it("hides the SVG from assistive tech, since the table carries the content", () => {
    const { container } = render(<Chart data={bars} title="Feature contributions" />);
    expect(container.querySelector('[aria-hidden="true"]')).toContainElement(
      screen.getByTestId("plot"),
    );
  });

  it("renders no empty table when a trace has no points", () => {
    render(<Chart data={[{ type: "bar", x: [], y: [] }]} title="Empty" />);
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("caps the table so it does not become its own navigation problem", () => {
    const many = Array.from({ length: 200 }, (_, i) => i);
    render(<Chart data={[{ type: "bar", x: many, y: many }]} title="Many" />);
    expect(within(screen.getByRole("table")).getAllByRole("rowheader")).toHaveLength(60);
  });
});
