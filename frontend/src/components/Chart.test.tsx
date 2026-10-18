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
const drawn = vi.hoisted(() => ({ data: [] as unknown[] }));

vi.mock("next/dynamic", () => ({
  default: () =>
    function StubbedPlot({ data }: { data: unknown[] }) {
      drawn.data = data;
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

  // F18: what the first table could not say.

  it("gives each series its own column, named as in the legend", () => {
    render(
      <Chart
        title="Predicted against actual"
        data={[
          { type: "scatter", name: "actual", x: [2023, 2024], y: [10, 12] },
          { type: "scatter", name: "predicted", x: [2023, 2024], y: [11, 15] },
        ]}
        layout={{ xaxis: { title: { text: "season" } }, yaxis: { title: { text: "EUR" } } }}
      />,
    );
    const table = screen.getByRole("table");
    expect(within(table).getByRole("columnheader", { name: "actual" })).toBeInTheDocument();
    expect(within(table).getByRole("columnheader", { name: "predicted" })).toBeInTheDocument();
    const row = within(table).getByRole("row", { name: /2024/ });
    expect(row).toHaveTextContent("12");
    expect(row).toHaveTextContent("15");
    expect(within(table).getAllByRole("rowheader")).toHaveLength(2); // not one per trace
    expect(table.querySelector("caption")).toHaveTextContent("Values are EUR");
  });

  it("heads the label column with the chart's own axis title", () => {
    render(
      <Chart data={bars} title="Impact" layout={{ yaxis: { title: { text: "feature" } } }} />,
    );
    expect(screen.getByRole("columnheader", { name: "feature" })).toBeInTheDocument();
  });

  it("states per-point notes, such as which seasons were held out", () => {
    render(
      <Chart
        title="History"
        data={[
          {
            type: "scatter",
            name: "predicted",
            x: [2021, 2023],
            y: [5, 7],
            customdata: ["in training range", "held out"],
          },
        ]}
      />,
    );
    const table = screen.getByRole("table");
    expect(within(table).getByRole("columnheader", { name: /note/ })).toBeInTheDocument();
    expect(within(table).getByRole("row", { name: /2023/ })).toHaveTextContent("held out");
    expect(within(table).getByRole("row", { name: /2021/ })).toHaveTextContent(
      "in training range",
    );
  });

  it("reads a histogram as bins and counts, the bins Plotly is given", () => {
    const values = [1, 1.5, 2, 2, 2.5, 3, 3, 3, 4, 5];
    render(
      <Chart
        title="Distribution"
        data={[{ type: "histogram", x: values }]}
        layout={{ xaxis: { title: { text: "log value" } }, yaxis: { title: { text: "players" } } }}
      />,
    );
    const table = screen.getByRole("table");
    const counts = within(table)
      .getAllByRole("cell")
      .map((cell) => Number(cell.textContent));
    expect(counts.reduce((a, b) => a + b, 0)).toBe(values.length);
    expect(within(table).getAllByRole("rowheader")[0]).toHaveTextContent(/^1 to /);
    expect(within(table).getByRole("columnheader", { name: "players" })).toBeInTheDocument();
    // The drawn chart was handed the same explicit bins the table counted.
    const trace = drawn.data[0] as { xbins?: { start: number; size: number } };
    expect(trace.xbins?.start).toBe(1);
    expect(trace.xbins?.size).toBeCloseTo(0.2);
  });
});
