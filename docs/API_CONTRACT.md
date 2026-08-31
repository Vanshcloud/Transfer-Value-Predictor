# API contract

The API is a **stable interface**, not a view onto whatever the implementation
currently returns. This document is written before the service and the service
conforms to it; where the two disagree, this document is the defect report.

Phase 10's dashboard is the first consumer, but it will not be the last, and a
consumer cannot be asked to re-read the source every time an internal detail
moves.

---

## 1. Versioning

Every data endpoint lives under `/api/v1/`. Version 1 is frozen once the
dashboard consumes it: fields may be **added**, never removed, renamed, or
changed in type or meaning. A breaking change is `/api/v2/`, served alongside.

`/health` is deliberately **unversioned**. It is for load balancers and
orchestrators, which should not have to know about API versions to decide
whether a process is alive.

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and readiness. Never versioned. |
| `POST` | `/api/v1/predict` | Predict a market value, with an explanation. |
| `GET` | `/api/v1/players?q=` | Find players by name. |
| `GET` | `/api/v1/players/{player_id}` | A player's record, history and feature seed. |
| `GET` | `/api/v1/players/{player_id}/similar` | Comparable seasons, in the model's feature space. |
| `GET` | `/api/v1/players/{player_id}/history` | Predicted against actual, season by season. |
| `GET` | `/api/v1/features/distribution` | Population quantiles per feature, for percentile framing. |
| `GET` | `/api/v1/models` | Every loaded model variant. |
| `GET` | `/api/v1/models/{variant}` | One model's card: family, params, features, training data. |
| `GET` | `/api/v1/models/{variant}/metrics` | Held-out metrics, in EUR. |
| `GET` | `/api/v1/models/{variant}/feature-importance` | Ranked importances and global SHAP. |

`variant` is one of `performance_only` or `with_prior_value`. It is a path
parameter rather than a query flag because the two are different products
answering different questions, not two settings of one model.

## 3. `POST /api/v1/predict`

Two request shapes, exactly one of which must be supplied.

### By player

```json
{ "player_id": 28003, "season": 2024, "variant": "performance_only" }
```

`season` is optional; omitted, the most recent season on record for that player
is used. The response says which season was actually used — a prediction whose
input period is ambiguous is not auditable.

### By explicit features

```json
{ "features": { "age": 24.5, "goals": 12, "minutes_played": 2400, "position": "Attack" },
  "variant": "performance_only" }
```

Any feature the model expects and the caller omits is imputed exactly as it is
during training, by the fitted pipeline. Unknown keys are **rejected**, not
ignored: silently dropping a misspelt `minutes_playd` gives a confident answer
to a question nobody asked.

Supplying both `player_id` and `features`, or neither, is a `422`.

### Response

```json
{
  "prediction_eur": 12450000.0,
  "variant": "performance_only",
  "model": { "name": "lightgbm", "variant": "performance_only", "trained_at": "..." },
  "player_id": 28003,
  "season": 2024,
  "confidence": {
    "level": 0.8,
    "lower_eur": 6100000.0,
    "upper_eur": 24900000.0,
    "basis": "empirical residual quantiles for the 5M-20M band on held-out seasons",
    "reference_rows": 812,
    "measured_coverage": 0.77
  },
  "explanation": {
    "base_value_eur": 2074881.0,
    "top_positive_features": [
      { "feature": "numeric__goals_per_90", "value": 0.62,
        "shap_value": 0.41, "effect_multiplier": 1.51, "direction": "increases" }
    ],
    "top_negative_features": [
      { "feature": "numeric__age", "value": 31.2,
        "shap_value": -0.33, "effect_multiplier": 0.72, "direction": "decreases" }
    ]
  }
}
```

### On `confidence` — read this before using it

There is no probability here, and the API will never invent one. A gradient
boosting regressor has no calibrated uncertainty, and a number like
`"confidence": 0.87` printed next to a prediction would be fabricated.

What `confidence` reports is an **empirical prediction interval**: the model's
own residual quantiles, measured on the validation season, for the value band
the prediction falls into. `reference_rows` is how many rows that was measured
over — read it, because a band with 40 reference rows deserves less trust than
one with 800.

**Read `measured_coverage`, not `level`.** `level` is what the interval asks
for; `measured_coverage` is what it achieved on the test seasons, which come
*after* the season its quantiles were taken from and which neither the model
nor the interval had seen. They are not the same number and the second is
smaller — around 0.77 against a nominal 0.80 for the performance-only model.

The gap is not a bug to close. Split conformal guarantees coverage under
exchangeability, and consecutive football seasons are not exchangeable: the
market inflates and the panel's composition moves. Widening the bounds until
the measured figure read 0.80 would mean fitting them to the test seasons,
which is how this field came to overstate itself in the first place — until
the final audit the quantiles were measured on the very rows their coverage
was then quoted on, where 80% is arithmetic rather than evidence.

`measured_coverage` is `null` on a model whose calibration was never checked
against a later season. Null means unmeasured, never 0.

The interval is wide. That is the honest finding, not a defect to tune away:
market value spans four orders of magnitude and the temporal split is hard.

### On explanation units

`shap_value` is additive in **log space**, because the model fits `log1p(EUR)`.
It is *not* additive in euros — the same contribution is worth a different
number of euros for a €500k player and a €90M one — so a client must never sum
them into euros. `effect_multiplier` is `exp(shap_value)` and is exact at any
value: 1.51 means this feature multiplied the prediction by 1.51.

`top_positive_features` and `top_negative_features` are each ordered by
magnitude, largest first, and capped at `top_n` (default 5, max 25).

## 3a. Player endpoints

`GET /api/v1/players?q=<name>&limit=<n>` is a **case-insensitive substring**
match, not fuzzy search. Fuzzy matching needs a measured threshold and a way to
judge a bad match, and neither exists yet; substring is honest about being
exactly what it is. Players with a modellable season are ranked first, and each
result carries `predictable` — a search result that leads to a 404 is worse
than no result.

`GET /api/v1/players/{player_id}` returns the player's seasons **and** a
`features` map: the model-ready feature values for the latest season. It is
there so a what-if form can start from the real values instead of
reconstructing them, and so "change goals from 10 to 18" is a change from
something true. Feeding that map straight back to `/predict` reproduces the
stored prediction exactly, and a test asserts it.

### The season in progress

A player's season list now includes the season **being played**, which has no
published valuation yet. Those rows carry `has_label: false` and
`market_value_in_eur: null` — null meaning "not yet published", never zero.

They are fully predictable. Every feature is complete the moment the matches
are played; only the label waits for Transfermarkt. `POST /api/v1/predict` with
just a `player_id` uses the player's most recent season, so it now answers for
the current one by default rather than for the last one that happened to be
labelled — which used to be roughly a year stale.

Three endpoints deliberately exclude these rows, because each of them reads a
recorded value that does not exist yet:

- `/players/{id}/history` — every point carries `actual_eur` to compare against.
- `/players/{id}/similar` — a neighbour is shown with its market value.
- `/features/distribution` — percentiles are computed over recorded values.

A partially played season is not refused. Half a season of matches is half a
season of evidence: `appearances`, `squad_match_share` and `months_active` all
fall, and the prediction reflects that there is less to go on.

`GET /api/v1/players/{player_id}/similar` measures distance on the
**preprocessed** matrix — the same scaled, encoded features the model sees — so
"similar" means similar to the model rather than similar on a hand-picked pair
of columns. The pool is restricted to the same season: market conditions differ
across years, and a 2014 striker is not a comparison for a 2024 one.

`GET /api/v1/players/{player_id}/history` returns the model's prediction beside
the recorded value for every season the player has. Each point carries
`in_training_range`: agreement on a season the model trained on is not evidence
about the model, and a chart that does not say which is which invites the reader
to count it as such.

`GET /api/v1/features/distribution` returns population quantiles per feature on
a fixed grid, so a client can say "92nd percentile for goals per 90" rather than
normalising against whichever two players are on screen. Both endpoints accept
`?variant=`; both default to the same variant as `/predict`. With no model
loaded, the distribution endpoint is `503` like every other model-backed route.
With a model whose training frame yields no usable quantiles it returns an empty
`grid` and `features` instead — an absent population is not a failed request.

## 3b. CORS

The dashboard runs on a different origin in development (`localhost:3000`
against `localhost:8000`), so browsers preflight every request. Allowed origins
are listed rather than wildcarded, set with `CORS_ORIGINS` in a deployment.
This API is read-only and unauthenticated today; `*` would be a habit that
becomes wrong the moment either of those changes.

## 4. Errors

Every non-2xx response uses one envelope, so a client writes one error path:

```json
{ "error": { "code": "player_not_found", "message": "no player with id 999999999",
             "detail": null } }
```

| Status | `code` | When |
|---|---|---|
| 404 | `player_not_found` | The player id is not in the dataset. |
| 404 | `season_not_found` | The player exists but has no row for that season. |
| 404 | `model_not_found` | No such variant is loaded. |
| 404 | `not_found` | No route matches the path. |
| 405 | `method_not_allowed` | The route exists but not for that method. |
| 422 | `validation_error` | The body failed schema validation, named an unknown feature, or gave a feature a value the model cannot be asked about. `detail` carries the field-level errors. |
| 500 | `internal_error` | The service failed unexpectedly. The cause is logged; the response deliberately carries no detail. |
| 503 | `model_unavailable` | No model artifact is loaded; the service is up but cannot predict. |

`422` bodies keep FastAPI's field-level detail inside `detail`, because "which
field, and why" is the entire value of a validation error. A bare
`{"detail": "Unprocessable Entity"}` forces a client to guess.

**Every** row above is the same envelope, including the ones this application
does not raise itself. Starlette's own `404` and `405`, and any unhandled
exception, are re-shaped by handlers in `api/errors.py`; without them a client
would need three parsers — the envelope, `{"detail": ...}`, and a `text/plain`
body. `tests/unit/test_api.py::TestContract::test_every_error_uses_one_envelope`
asserts it against a client configured the way a real deployment behaves.

`500` is the one status whose `message` is fixed and whose `detail` is always
`null`. An exception's text can carry a path, a query or a fragment of the data
it choked on, and this API is unauthenticated; the detail goes to the log.

### Feature values

`features` rejects more than unknown names. A value is refused, with the
feature named in the message, when it is:

- not a single value (an object, array or set),
- not a number where the model expects one — including `true`/`false`, since
  `float(True)` is `1.0` and would otherwise be answered confidently,
- not finite (`NaN`, `Infinity`, `1e999` — reachable in a raw body even though
  most JSON encoders refuse to write them),
- negative for a feature that cannot be (`src.feature_engineering.build.NON_NEGATIVE_FEATURES`:
  counts, durations, rates and physical measurements), or
- a category longer than 200 characters.

Every offending value is reported in one response rather than one per round
trip. An explicit `null` is **not** an error: it means "absent", and the fitted
imputer fills it exactly as it did during training.

## 5. OpenAPI

Served at `/api/v1/openapi.json`, with Swagger UI at `/docs` and ReDoc at
`/redoc`. Every schema carries field descriptions and a worked example, because
an OpenAPI document with bare types is a type checker, not documentation.

## 6. Layering

FastAPI is the transport, and nothing else:

```
FastAPI router      request/response schemas, HTTP status codes
      ↓
PredictionService   framework-agnostic: no Request, no HTTPException
      ↓
Artifact + pipeline preprocessing and model, as one fitted object
      ↓
Explainer           SHAP contributions as data
```

`PredictionService` imports nothing from FastAPI. That is what lets the same
prediction path serve batch inference, a CLI, or a test, without a running
server — and it is why the tests for prediction logic do not need a TestClient.

## 7. Model loading

Artifacts load **once**, in a `lifespan` handler, and reach handlers through
`Depends`. Not at import time: an import-time load makes the module unimportable
without a trained model, which breaks collection of every test in the suite.

`@app.on_event("startup")` is deprecated in FastAPI 0.141 and is not used.

## 8. What this API does not do

- **No authentication.** Nothing here is user data and there is nothing to
  spend. Adding auth without a threat model would be theatre.
- **No write endpoints.** Predictions are not persisted. The service is a pure
  function of the artifact plus the request.
- **No training.** Training is `scripts/train_models.py`, deliberately not
  reachable over HTTP.
- **No batch endpoint yet.** It will be `POST /api/v1/predict:batch` when a
  consumer needs one; guessing its shape now would freeze a wrong guess into a
  frozen v1.
- **No fuzzy name search.** Substring only, until there is a way to measure
  whether a fuzzy match is any good.
- **No league-wide aggregates.** The dashboard's analytics page samples through
  the search endpoint. If it ever needs the whole panel it wants a real
  aggregate endpoint, not a hundred client-side requests.
