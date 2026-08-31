# Security policy

## Reporting a vulnerability

Open a [private security advisory](https://github.com/Vanshcloud/Transfer-Value-Predictor/security/advisories/new)
rather than a public issue. If that is unavailable to you, email
vanshwar@gmail.com with `SECURITY` in the subject.

Expect a first reply within seven days. This is a single-author portfolio
project, not a funded product — the honest commitment is that a report will be
read and answered, not that a patch will ship on a schedule.

## What this project is, for threat-modelling purposes

A read-only inference service over a public, CC0-licensed football dataset.

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

## Out of scope

Denial of service through sheer request volume. The service is CPU-bound on
SHAP (~30 ms per prediction) and ships no rate limiting; put a reverse proxy in
front of it. This is a documented deployment requirement, not a defect to
report.
