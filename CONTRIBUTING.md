# Contributing

Read this before opening a pull request, because the answer is unusual.

## Pull requests are not accepted

This is a single-author portfolio project, and that is enforced mechanically
rather than by convention. `scripts/hooks/commit-msg` rejects any commit whose
message carries a `Co-Authored-By` trailer or an AI-generation marker, and CI
re-checks the whole history on every push. A pull request cannot be merged here
without breaking a property the repository exists to demonstrate.

Saying so plainly seems better than accepting a contribution and then explaining
why it cannot land.

## Issues are welcome

Genuinely — they are the useful contribution. Especially:

- **A number that does not reproduce.** Every figure in the README is regenerated
  by a command named beside it. If one disagrees with your run, that is a real
  bug and I want to know.
- **A documented behaviour that is not the actual behaviour.** The API contract
  in `docs/API_CONTRACT.md` is checked against the live OpenAPI document by
  `tests/unit/test_api_contract_sync.py`, but prose can still drift from code in
  ways a test cannot see.
- **Setup failing on a platform I have not tried.** Developed on macOS 15 with
  Python 3.13 and Node 24; CI runs Ubuntu.

## If you want to build on it

Fork it. The licence is MIT and no attribution ritual is required beyond the
licence text. The data is not MIT — see the Licence section of the README before
redistributing anything derived from it.

## Running the checks

```bash
make setup          # venv, dev dependencies, git hooks
make test           # unit suite; needs no data and no credentials
make quality        # ruff, black --check, mypy
make test-cov       # the same suite with a coverage report
```

The full suite including integration tests needs a trained model:

```bash
python scripts/fetch_data.py      # needs Kaggle credentials, see README
python scripts/build_features.py
python scripts/train_models.py
pytest                            # integration tests now run instead of skipping
```

Integration tests **skip** rather than fail when data and models are absent.
That is deliberate: a suite that is only green on one machine is not a suite.

## Two traps worth knowing before you touch dependencies

**Regenerate `frontend/package-lock.json` on Linux, not on macOS.** Some
transitive packages (`@emnapi/*`, and anything else resolved per-platform)
resolve differently, and a lock written on macOS is one `npm ci` on Linux
rejects — which is both CI's frontend job and the Docker build. The lock is
generated in the same image they use:

```bash
docker run --rm -v "$PWD/frontend:/w" -w /w node:24-slim \
    sh -c "npm install --package-lock-only && npm ci"
```

**Regenerate `requirements-lock.txt` after editing `requirements.txt`**, and
`requirements-serve-lock.txt` after editing `requirements-serve.txt`:

```bash
uv pip compile requirements.txt --python-version 3.13 --output-file requirements-lock.txt
uv pip compile requirements-serve.txt --python-version 3.13 --output-file requirements-serve-lock.txt
```

`tests/unit/test_dependencies.py` fails if a declaration and its lock disagree.
