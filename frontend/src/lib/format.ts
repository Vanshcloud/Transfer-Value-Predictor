/** Display helpers. Money is formatted in one place so it reads the same everywhere. */

/** EUR, abbreviated. Full precision on a €47,300,000 figure is noise on a card. */
export function eur(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `€${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `€${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `€${(value / 1e3).toFixed(0)}k`;
  return `€${value.toFixed(0)}`;
}

export function eurExact(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(value);
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function number(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-GB", { maximumFractionDigits: digits });
}

/** Strip the ColumnTransformer prefix so "numeric__goals_per_90" reads as "goals per 90". */
export function featureLabel(name: string): string {
  return name
    .replace(/^(numeric|categorical)__/, "")
    .replace(/_/g, " ")
    .replace(/\bper 90\b/, "per 90")
    .replace(/\beur\b/i, "EUR");
}

export function shortDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const parsed = new Date(iso);
  return Number.isNaN(parsed.valueOf()) ? "—" : parsed.toISOString().slice(0, 10);
}
