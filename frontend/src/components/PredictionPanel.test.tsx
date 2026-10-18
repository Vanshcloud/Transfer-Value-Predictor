/**
 * Audit F9: an interval's record is a frequency over a group of predictions,
 * and it must never read as this player's probability — nor quote the overall
 * figure when the prediction's own band was measured.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { IntervalRecord } from "./PredictionPanel";
import type { Confidence } from "@/lib/api";

function confidence(over: Partial<Confidence> = {}): Confidence {
  return {
    level: 0.8,
    lower_eur: 1_000_000,
    upper_eur: 4_000_000,
    basis: "empirical residual quantiles for the 1M-5M band on held-out seasons",
    reference_rows: 2034,
    measured_coverage: 0.751,
    band: "1M-5M",
    band_measured_coverage: 0.746,
    band_coverage_rows: 4382,
    ...over,
  };
}

describe("IntervalRecord", () => {
  it("quotes the prediction's own band, with the rows behind it", () => {
    render(<IntervalRecord confidence={confidence()} />);
    const text = screen.getByText(/On the test seasons/).textContent ?? "";
    expect(text).toContain("predictions in the 1M-5M band");
    expect(text).toContain("4,382 rows");
    expect(text).toContain("75%");
    expect(text).toContain("80% targeted");
    expect(text).toContain("75% across all predictions");
  });

  it("says it is not a probability for this player", () => {
    render(<IntervalRecord confidence={confidence()} />);
    expect(screen.getByText(/not a probability for this player/)).toBeInTheDocument();
  });

  it("falls back to the overall figure when the band was not measured", () => {
    render(
      <IntervalRecord
        confidence={confidence({ band_measured_coverage: null, band_coverage_rows: null })}
      />,
    );
    const text = screen.getByText(/On the test seasons/).textContent ?? "";
    expect(text).toContain("all predictions");
    expect(text).not.toContain("band");
  });

  it("shows nothing when coverage was never measured", () => {
    const { container } = render(
      <IntervalRecord
        confidence={confidence({ measured_coverage: null, band_measured_coverage: null })}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
