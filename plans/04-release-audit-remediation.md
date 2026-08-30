# Phase 14 — external release audit, and what it cost to close

An independent audit of the repository at `v1.0.0`, conducted as due diligence
rather than code review: nothing was taken on trust, every claim was re-derived
from the repository, and the explicit goal was to find a reason not to publish.

It did not find one. It found three HIGH issues, five MEDIUM, ten LOW and three
NITs. All are closed. This records what was checked, what held, what did not,
and what was deliberately left alone.

## What held, re-derived rather than re-read

Every figure below was recomputed from the repository, not copied from the
previous phase.

| Claim | How it was checked | Result |
|---|---|---|
| Six baseline table cells | re-ran `scripts/train_baseline.py` | exact, all six |
| `performance_only` €4.44M / R² 0.441 | `models/*.json` | 4,442,427 / 0.4410 |
| `with_prior_value` €3.71M / R² 0.775 | `models/*.json` | 3,710,897 / 0.7747 |
| 36,880 rows, 19,827 with a prior value | artifact metadata | exact |
| "RF serialises to 458 MB … LightGBM 1.7 MB" | leaderboard `size_bytes` | 457.5 MiB / 1.72 MB |
| "CatBoost within 0.75%, 5.7× faster, 4.8× less disk" | leaderboard | 0.748%, 5.73×, 4.80× |
| Screenshot's 80% interval | recomputed from `calibration.bands` | €14.28M–€80.0M, n=145 |
| Clean clone, no credentials | fresh `git clone` + venv | 538 pass, 47 skip, 0 fail |
| "roughly fifteen seconds" | timed on that clone | 14.9s |
| Containers run as non-root | `docker compose exec id` | uid 10001, both |
| Mounts are read-only | `docker inspect` | `RW=false`, both |
| Degraded start | ran the image with no mounts | 200, `ready: false`, 503 on predict |
| Service imports no web framework | read the test | genuinely `ast.parse`, not a grep |
| Single authorship | `git log` | 0 trailers, one address |
| `npm audit`, `pip-audit` | run | 0 vulnerabilities each |
| Secrets in tree and history | scanned 45,689 diff lines | none |

Twenty-odd numeric claims; one was wrong. That one is below.

## The three that mattered

**1. `POST /api/v1/predict` returned a bare 500 on a malformed feature value.**
Names were validated, values were not. `{"features": {"age": "twenty-five"}}`
reached the fitted pipeline, scikit-learn raised from inside a transformer, and
the caller got `text/plain` "Internal Server Error" — for something they typed.
`api/errors.py` opens by claiming it collapses *every* failure into one
envelope; `docs/API_CONTRACT.md` §4 says the same. Both were false.

No test caught it because `TestClient` re-raises server exceptions by default,
so the suite never saw the response a deployment actually sends. The regression
tests now use `raise_server_exceptions=False`. That single detail is the reason
this survived a full phase of testing.

**2. The redaction had not redacted anything.** `plans/00-discovery.md` said
the local paths, account name and machine inventory "has been removed". They
were in 16 of 17 commits, and the commit that removed them printed every line
again in its own diff. Publishing would have published all of it. Rewritten
with `git filter-repo` while the repository was still private with 0 forks —
the only cheap moment — and verified gone by `git log -S` per string and by a
full `git log -p` scan.

**3. The coverage figure was not the one its command printed.** README said
97% and cited `make test-cov`, which prints 89%. Both are true of different
suites; only one was stated. This is the claim a sceptical reader checks first.

## What the fixes cost, and what they bought

- **Image size.** Measuring the image rather than accepting its size found 291 MB
  of CUDA libraries that xgboost ships for a GPU the inference path never
  touches, plus 269 MB of catboost and 59 MB of the plotly it drags. None is
  needed to unpickle a LightGBM artifact. A serving-only lock took the API image
  **2.62 GB → 1.32 GB**, verified by running the full serving path — prediction,
  interval, SHAP, neighbours — on the slim set and getting a byte-identical
  answer.
- **The frontend had no tests at all.** Writing 77 of them found two real
  defects immediately: a money-formatting bug (`€1000k`) and a `useAsync` that
  refetches forever on an unstable fetcher — 27,000 requests before the test
  noticed. Every existing call site was correct, so nothing was broken; the next
  page added would have been.
- **Documents that nothing checks are checked once.** Phase 13 fixed the API
  contract and added a test for it. Phase 14 found the same two endpoints
  missing from the README, because only one document had a test. Five new test
  modules now check the README's counts against pytest's collector, both
  endpoint tables against the live OpenAPI document, the lock against the
  declaration, the serve set against the shipped artifacts, and the release tag
  against the declared version.

## What the new guards caught, on their first run

Two things, neither of which any pre-existing gate would have seen.

**A history rewrite is not finished when `main` is pushed.** Between
`git push --force origin main` and `git push --force --tags`, the remote tag
`v1.0.0` still pointed at a pre-rewrite commit — so the old history, PII
included, was reachable from the remote for about ninety seconds. `git log -p
--all` reads tags, so the new CI grep failed on exactly that window and named
the line. A local clone showed nothing, because locally the tag had already
moved. Push the tags in the same breath as the branch, or the rewrite is only
half done.

Worth stating plainly: GitHub retains unreferenced objects after a force-push,
and a commit SHA remains fetchable for a while even with no ref pointing at it.
The rewrite was done while the repository was private and never forked, so
nobody outside the account has ever had a SHA to ask for. That is the reason
this was worth doing before publishing rather than after.

**A lockfile written on macOS is one `npm ci` rejects on Linux.** `@emnapi/*`
resolves per platform, and CI's frontend job and the Docker build both run
`npm ci` on Linux. Adding vitest on a Mac produced a lock that would have
failed CI on the very first push. The lock is now generated inside
`node:24-slim`, and CONTRIBUTING says so.

## Deliberately not done

- **Bounding feature values from above.** `age: 1e308` is finite and returns a
  prediction. Refusing it means inventing a plausible range per feature, and the
  model card already says the model should not be trusted outside its training
  support. A measured out-of-support warning would be worth more than an
  invented ceiling.
- **Replacing joblib.** `load()` unpickles, and the version check runs after the
  code has already executed. Moving to ONNX would remove the assumption rather
  than document it, but it would also cost the fitted preprocessing travelling
  inside the artifact — the property that stops training and serving drifting
  apart. Documented instead, with the deployment shape that makes it acceptable.
- **Signing commits.** The hook and CI both enforce a single author address, but
  an address is a claim, not a proof. Signing is the real fix and is a key-
  management decision, not a code change.
- **Editing `plans/03-final-verification.md`'s "517 tests".** It is a dated
  record of what was true that day. Updating it would falsify the record.

## Release checklist

Green before tagging `v1.1.0`. Everything here is a command.

```sh
make quality                          # ruff, black, mypy
make test-cov                         # 538 pass, 47 deselected, >= 88%
pytest                                # 585, integration included
cd frontend && npm test && npx tsc --noEmit && npx eslint src && npm run build
docker compose build && docker compose up -d
curl -sf localhost:8000/health && docker compose down -v
git log --format='%ae%n%ce' | sort -u                 # one address
git log -p --all | grep -cE '/(Users|home)/[a-z]+/'   # 0
pip-audit && (cd frontend && npm audit)
```
