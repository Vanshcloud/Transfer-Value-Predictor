# Security policy

## Reporting a vulnerability

Open a [private security advisory](https://github.com/Vanshcloud/Transfer-Value-Predictor/security/advisories/new)
rather than a public issue. If that is unavailable to you, email
vanshwar@gmail.com with `SECURITY` in the subject.

Expect a first reply within seven days. This is a single-author portfolio
project, not a funded product — the honest commitment is that a report will be
read and answered, not that a patch will ship on a schedule.

## What this project is, for threat-modelling purposes

A read-only inference service over a public football dataset (a Kaggle mirror of
Transfermarkt data; its usage rights are discussed in
[`docs/DATASET_CARD.md`](docs/DATASET_CARD.md#data-rights), not here).

- **No authentication, and none intended.** Every endpoint is public and every
  response is derived from public data. There is no session, no cookie and no
  account. Deploy it behind your own gateway if you need access control.
- **No personal data beyond the public record.** Player biography, appearances
  and published market valuations. No contact details, no private information,
  nothing scraped. See [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md).
- **No writes.** No endpoint mutates state, uploads a file, or accepts a model.
  `docker-compose.yml` mounts `models/` read-only.
- **No secrets in the image.** Kaggle credentials are needed only by
  `scripts/fetch_data.py` at build-your-own-dataset time, never by the service.
  `.env` is gitignored and `.env.example` carries no real values.

## Known and accepted risks

**Model deserialisation.** `src/models/artifact.py` loads artifacts with
`joblib.load`, which unpickles and therefore executes whatever the file
contains. The `artifact_version` check runs *after* that, so it is a
compatibility guard and never a security one.

This is accepted rather than overlooked, and the reasoning is in the function's
own docstring: nothing downloads an artifact, no endpoint accepts one, and the
deployment mounts `models/` read-only. The threat requires an attacker who can
already write to the model directory, at which point the pickle is not the
weakest thing they control.

**Load only artifacts you produced.** Replacing joblib with ONNX or the
booster's own text dump would remove the assumption instead of documenting it,
at the cost of the fitted preprocessing travelling inside the same object —
which is the property that stops training and serving drifting apart. It is on
the roadmap as a deliberate trade, not a pending fix.

## Dependency advisories

Audited 2026-10-19 with `npm audit` and `pip-audit` against the committed locks.

| Package | Was | Now | Advisories | Reachable here? |
|---|---|---|---|---|
| `next` | 16.3.3 | **16.3.8** | RCE in `next/og`; SSRF in image optimisation; cache poisoning of SSG/ISR pages; draft-mode leak via `use cache`; metadata-image disclosure; dev-server MCP disclosure | The app uses none of `next/og`, `next/image`, `use cache`, draft mode or ISR revalidation, but the pages *are* statically generated and served by a self-hosted `next start`, so the SSG cache-poisoning advisories were not ruled out. Patched within the same minor. |
| `sharp` | 0.35.4 | **0.35.5** | librsvg CVE in SVG decoding | Pulled in by `next` for image optimisation, which is unused. Patched. |
| `maplibre-gl` (via `plotly.js`) | 5.24.0 | 6.11.2 | XSS sanitiser bypass | Not shipped: the dashboard bundles `plotly.js-cartesian-dist-min`, which contains MapLibre's CSS strings only, no map code. The full `plotly.js` is installed only as `react-plotly.js`'s peer and used for types. Upgraded to `plotly.js` 4.1.2 anyway, within its declared range. |
| `source-map-js` | 1.2.1 | **1.2.2** | event-loop DoS from crafted source maps | Build-time only (PostCSS, Tailwind). Patched. |
| `undici` (via `jsdom`) | 8.10.0 | **8.12.2** | several DoS, cache and TLS issues | Test-only. Patched. |
| `brace-expansion` | 1.1.18 / 5.0.9 | **1.1.21 / 5.0.12** | ReDoS-style CPU and stack exhaustion | Lint-only. Patched. |
| `braces` ← `micromatch` ← `fast-glob` ← `eslint-config-next` | 3.0.3 | 3.0.3 | stack exhaustion on nested patterns | **Open.** No patched `braces` exists; npm's only "fix" is downgrading to the Next 14 lint config. Lint-time only, over this repository's own glob patterns; never installed in an image or reached by a request. |
| `urllib3` | 2.7.0 | **2.8.0** | streaming decompression and chunked-decoding DoS; proxy TLS settings misapplied | Only `scripts/fetch_data.py` makes HTTP requests (to Kaggle, without a proxy); the serving lock does not contain `urllib3`. Patched in `requirements-lock.txt`. |

After the upgrades: `npm audit --omit=dev` reports 0 vulnerabilities, `npm audit`
reports the 5 `braces`-chain entries above, `pip-audit` reports none for either
lock. Advisory databases change; rerun both before relying on this table.

## What has been checked

Measured in [`plans/07-adversarial-audit.md`](plans/07-adversarial-audit.md)
and [`plans/08-staff-engineering-pass.md`](plans/08-staff-engineering-pass.md):

| Vector | Result |
|---|---|
| YAML parsing | `yaml.safe_load` throughout; no `yaml.load` anywhere |
| Malformed request bodies | 5,000 unknown keys, 1 MB body, 60-deep nesting — all `422` before any model work |
| Unbounded result sets | Search `limit` is range-checked; an oversized or negative value is `422` |
| Out-of-range features | Rejected via `PLAUSIBLE_RANGES`; an impossible age is `422`, not a confident answer |
| Unknown categories | Encoded as infrequent rather than crashing; verified on 402 real rows |
| Concurrency | 320 threaded requests, all `200`, caches bounded, no corruption |
| Secrets in history | CI greps the full history on every push |

CI additionally fails the build if any local filesystem path or host inventory
appears anywhere in the commit history.

## Deployment safeguards

The intended deployment is a public, read-only demonstration: no accounts, no
writes, public data. Authentication is deliberately absent, and adding it would
protect nothing. What a public deployment needs instead is protection from
being run out of resources. Each item is either enforced by the API, set in the
container, or left to the reverse proxy, which is the only place it can be
enforced correctly.

| Safeguard | Where | Setting |
|---|---|---|
| Request body size | API, `api/limits.BodySizeLimit` | 64 KB, `413 payload_too_large`. Chunked bodies are counted, not trusted. |
| Concurrent model work | API, `api/limits.inference_slot` | One per core, up to 8. A request waiting over 2 s gets `503 service_busy` with `Retry-After: 1`. Applies to predict, history, similar players and feature importance. |
| Global SHAP | API, `PredictionService.global_explanation` | Computed once per loaded model and cached; the sample is seeded, so the cache is exact. |
| Per-prediction SHAP | Selection policy | A model ships only if predict plus explanation is ≤ 100 ms p50 on validation rows (`SERVING_LATENCY_BUDGET_MS`). |
| Open connections | API entry point, `api/__main__.py` | uvicorn `limit_concurrency=64` (503 beyond), `timeout_keep_alive=5`. |
| Memory and CPU | Container, `docker-compose.yml` | API 2 GB / 2 CPUs (≈710 MB resident at start), dashboard 512 MB / 1 CPU. |
| Liveness | API, `GET /health` | 200 whenever the process is up, so a missing model never restart-loops it. |
| Readiness | API, `GET /ready` | 503 until a model is loaded, for a load balancer to route on. |
| Error reporting | API, `api/errors.py` | Every unhandled exception is logged with its traceback to stdout and returned as a fixed-message `500`. No error-tracking service is integrated; ship the logs to one. |
| **Rate limiting** | **Reverse proxy** | Per client IP. In-process counters are wrong behind a proxy (every request comes from it) and with more than one worker. |
| **Request timeout** | **Reverse proxy** | A Python thread running SHAP cannot be interrupted, so a timeout inside the process would only abandon the response, not the work. |
| **TLS** | **Reverse proxy** | The containers speak plain HTTP. |

A minimal nginx front for the API, as a starting point rather than a tuned
configuration:

```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    listen 443 ssl;
    client_max_body_size 64k;

    location / {
        limit_req zone=api burst=20 nodelay;
        proxy_read_timeout 10s;
        proxy_pass http://api:8000;
    }
}
```

These were tested without load testing: the body limit, load shedding, slot
release, readiness and the SHAP cache each have a unit test
(`tests/unit/test_api.py::TestDeploymentSafeguards`). No sustained-traffic
measurement has been made, and none is claimed.

## Out of scope

Denial of service through sheer request volume beyond the safeguards above.
That is what the reverse proxy's rate limit is for; without one, this service
should not be exposed directly to the internet.
