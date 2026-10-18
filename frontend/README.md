# Dashboard

The frontend for the Transfer Value Predictor. Next.js 16, React 19,
Tailwind v4, Plotly.

```bash
npm ci
npm run dev     # expects the API on http://localhost:8000
```

Point it elsewhere with `NEXT_PUBLIC_API_BASE`:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8010 npm run dev
```

## Three things to know before editing

- **Plotly lives in exactly one file.** `src/components/Chart.tsx` does the
  `next/dynamic` import with `ssr: false`, because a plain import fails
  `next build` with `ReferenceError: self is not defined` even inside
  `"use client"` — client components are prerendered on the server. It also
  pins the slim cartesian bundle (1.4 MB) instead of full Plotly (4 MB).
  Import that wrapper; never import Plotly directly.
- **There is no `tailwind.config.js`.** Tailwind v4 is CSS-first; theme config
  lives in `@theme` blocks in `src/app/globals.css`. Creating the config file
  would silently do nothing.
- **`@types/react-plotly.js` is deliberately not installed.** react-plotly.js
  v4 ships its own types and the DefinitelyTyped package conflicts with them.
  The slim bundle has no types, so `src/types/plotly-cartesian.d.ts` aliases it
  to `plotly.js`.

## Tests

```bash
npm test        # 118 tests, a few seconds
```

The suite covers the parts of the dashboard where being wrong is silent: the
error mapping in `lib/api.ts` (every failure becomes one `ApiError` carrying
the server's own `code`), the race guards (`useLatest`: only the newest request
for a search box, a comparison side, a what-if slider or a `useAsync` fetch may
change the screen — `src/test/races.test.tsx` resolves the earlier request last
in each case), the season warning on `/compare`, the money formatting,
and the state each component renders — including that a per-feature euro figure
never appears next to a SHAP contribution, because that number would be false.

It does not test what Plotly draws. jsdom cannot answer that, and the frontend
failure that actually breaks this app is the SSR trap, which `next build`
catches in CI.

## Accessibility

Charts are not only pictures. Plotly emits an SVG of positioned shapes, so a
screen reader would otherwise get a label and no numbers — and the numbers are
the content. `Chart.tsx` marks the SVG `aria-hidden` and renders the same
data as a visually-hidden table, derived from the one `data` prop so the two
cannot describe different charts: one row per category and one column per named
series, headed by the axis titles; per-point notes from string `customdata`
(which seasons were held out, why a model family did not ship); and histograms
as bin ranges and counts, from explicit bins Plotly is given too. The inline-SVG
radar on `/compare` has its own table of each player's percentile and value.

## Checks

```bash
npx tsc --noEmit    # types
npx eslint src      # lint
npm run build       # the one that catches the SSR trap
```
