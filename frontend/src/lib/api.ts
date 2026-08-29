/**
 * Typed client for the Transfer Value Predictor API.
 *
 * The types here mirror docs/API_CONTRACT.md. They are written by hand rather
 * than generated, because the contract is small and stable and a generator
 * would be another build step to keep working; if the API grows, generate them
 * from /api/v1/openapi.json instead of letting these drift.
 *
 * Every failure comes back as ApiError carrying the server's own `code`, so a
 * page can branch on "player_not_found" rather than parsing a message.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type Variant = "performance_only" | "with_prior_value";

export interface Contribution {
  feature: string;
  value: unknown;
  /** Additive in log space. Never sum these into euros. */
  shap_value: number;
  /** exp(shap_value): what this feature multiplied the prediction by. Exact. */
  effect_multiplier: number;
  direction: "increases" | "decreases";
}

export interface Confidence {
  level: number;
  lower_eur: number;
  upper_eur: number;
  basis: string;
  reference_rows: number;
}

export interface PredictResponse {
  prediction_eur: number;
  variant: string;
  model: { name: string; variant: string; trained_at: string };
  player_id: number | null;
  season: number | null;
  confidence: Confidence | null;
  explanation: {
    base_value_eur: number;
    top_positive_features: Contribution[];
    top_negative_features: Contribution[];
  } | null;
}

export interface SeasonRow {
  season: number;
  age: number;
  appearances: number;
  goals: number;
  assists: number;
  minutes_played: number;
  market_value_in_eur: number;
}

export interface Player {
  player_id: number;
  name: string | null;
  position: string | null;
  sub_position: string | null;
  foot: string | null;
  height_in_cm: number | null;
  country_of_citizenship: string | null;
  seasons: SeasonRow[];
  features: Record<string, unknown>;
}

export interface SearchResult {
  player_id: number;
  name: string;
  position: string | null;
  latest_season: number | null;
  market_value_in_eur: number | null;
  predictable: boolean;
}

export interface SimilarPlayer {
  player_id: number;
  name: string | null;
  season: number;
  position: string | null;
  age: number;
  market_value_in_eur: number;
  distance: number;
}

export interface Metrics {
  mae_eur: number | null;
  rmse_eur: number | null;
  r2: number | null;
  mape: number | null;
  n: number | null;
}

export interface ModelMetrics {
  variant: string;
  validation: Metrics;
  test: Metrics;
  leaderboard: Record<string, unknown>[];
}

export interface ModelInfo {
  variant: string;
  model_name: string;
  params: Record<string, unknown>;
  feature_columns: string[];
  target_column: string;
  trained_at: string;
  seed: number;
  split: Record<string, unknown>;
  dataset: Record<string, unknown>;
  artifact_version: number;
  explainable: boolean;
}

export interface FeatureImportance {
  variant: string;
  model_name: string;
  features: { feature: string; importance: number }[];
  shap: {
    sample_size?: number;
    features?: { feature: string; mean_abs_shap: number; mean_shap: number }[];
  };
}

export interface Health {
  status: "ok" | "degraded";
  ready: boolean;
  models_loaded: string[];
  version: string;
}

/** A failure the server described, with the code a caller can branch on. */
export class ApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      cache: "no-store",
    });
  } catch {
    // A dead backend is the most common failure in development, and
    // "Failed to fetch" tells a reader nothing about what to do next.
    throw new ApiError(
      "network_error",
      `Cannot reach the API at ${API_BASE}. Is it running? (make serve)`,
      0,
    );
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const error = body?.error;
    throw new ApiError(
      error?.code ?? "unknown_error",
      error?.message ?? `Request failed with ${response.status}`,
      response.status,
      error?.detail,
    );
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

  searchPlayers: (query: string, limit = 20) =>
    request<{ query: string; results: SearchResult[] }>(
      `/api/v1/players?q=${encodeURIComponent(query)}&limit=${limit}`,
    ),

  player: (id: number) => request<Player>(`/api/v1/players/${id}`),

  similarPlayers: (id: number, k = 8, variant?: Variant) =>
    request<{ player_id: number; season: number | null; results: SimilarPlayer[] }>(
      `/api/v1/players/${id}/similar?k=${k}${variant ? `&variant=${variant}` : ""}`,
    ),

  predictForPlayer: (playerId: number, variant?: Variant, season?: number) =>
    request<PredictResponse>("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify({ player_id: playerId, variant, season, top_n: 8 }),
    }),

  predictFromFeatures: (features: Record<string, unknown>, variant?: Variant) =>
    request<PredictResponse>("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify({ features, variant, top_n: 8 }),
    }),

  models: () => request<{ variants: string[]; default: string | null }>("/api/v1/models"),

  modelInfo: (variant: string) => request<ModelInfo>(`/api/v1/models/${variant}`),

  modelMetrics: (variant: string) =>
    request<ModelMetrics>(`/api/v1/models/${variant}/metrics`),

  featureImportance: (variant: string, topN = 20, includeShap = false) =>
    request<FeatureImportance>(
      `/api/v1/models/${variant}/feature-importance?top_n=${topN}&include_shap=${includeShap}`,
    ),
};
