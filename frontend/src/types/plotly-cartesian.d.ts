/**
 * Types for the slim Plotly distribution.
 *
 * `plotly.js-cartesian-dist-min` is a prebuilt bundle and ships no types of its
 * own, but its API is a subset of `plotly.js`, whose types are already present.
 * Declaring the alias here costs three lines and avoids another dependency —
 * and `@types/react-plotly.js` stays uninstalled, because react-plotly.js v4
 * ships its own and the DefinitelyTyped package conflicts with them.
 */
declare module "plotly.js-cartesian-dist-min" {
  import type * as Plotly from "plotly.js";
  export = Plotly;
}
