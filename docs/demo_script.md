# Demo script

A twelve-minute walkthrough. Every command below was run against this
repository and the outputs are what it actually printed — figures will differ
slightly once you retrain, and that is the point of the reproducibility segment.

**Before you start:** `make setup && make test`. If that is green you can
deliver the whole demo with no network, no Kaggle account and no trained model.
Everything requiring data is marked **[needs data]** and has a fallback.

---

## 0 — Setup (before the audience arrives)

```bash
make setup                  # venv + dev deps + git hooks
make test                   # 679 pass, 53 skip, ~15s, no credentials
docker compose up --build   # optional; takes minutes on a cold cache
```

Have open: a terminal, the dashboard at `http://localhost:3000`, and
`docs/MODEL_CARD_performance_only.md`.

---

## 1 — The problem (1 min, no commands)

> "What is a footballer worth, and *why*? Transfermarkt publishes a number for
> every player. It is a crowd consensus, and nobody can tell you which parts of
> a player's season produced it. This predicts that number from performance and
> biography, and decomposes every prediction into the features that moved it."

**Talking point — the constraint that shaped everything.** Transfermarkt's
Terms of Use §11.1 prohibit automated access *and* separately prohibit training
on the content. So nothing is scraped: labels come from a CC0 Kaggle mirror.
That is the first slide because it explains an architecture decision, not a
legal footnote.

---

## 2 — It runs with nothing installed but Python (1 min)

```bash
make test
```

Expected: `679 passed, 53 skipped`, about fifteen seconds.

> "The 53 skips are integration tests that need a trained model. They skip
> rather than fail, because a suite that is only green on my laptop is not a
> suite."

---

## 3 — The pipeline **[needs data]** (2 min)

```bash
python scripts/build_features.py
```

Expected tail:

```
  leakage: 0 error(s), 0 warning(s)
```

> "Leakage detection is a *pipeline stage*, not a test. Seven checks run here
> and re-run after every split. A leak stops the run — because leakage doesn't
> raise an exception, it produces a *better* number, and nobody investigates a
> model that beat expectations."

**Fallback (no data):** show `src/validation/leakage.py` and read the module
docstring, which lists the seven failure modes and names the one that was found
in production.

**Best single talking point in the demo.** Club form used to be a whole-season
average. A season runs August–July, so a fixture on 20 July belongs to it while
falling *after* the 1 July cutoff — 5.56% of all fixtures are dated in July.
That folded matches played after the *label* into 1,049 rows. Seven checks
missed it because the leak lived inside a number, not a column name or a date.
Removing it cost nothing measurable (p = 0.49). It was not buying accuracy,
only overstating provenance.

---

## 4 — The API (2 min)

```bash
curl -s localhost:8000/health | jq
```

```json
{ "status": "ok", "ready": true,
  "models_loaded": ["performance_only", "with_prior_value"],
  "version": "1.3.0" }
```

```bash
curl -s localhost:8000/api/v1/predict \
  -H 'content-type: application/json' \
  -d '{"player_id": 418560}' | jq '{prediction_eur, season, confidence}'
```

> "Two things to notice. `season` is 2025 — the season being *played*, which
> has no label yet but has complete features. And `confidence` carries two
> coverage numbers."

**The honest-uncertainty talking point.** `level` is 0.80 — what the interval
asks for. `measured_coverage` is ~0.77 — what it achieved on test seasons it
had never seen. Until the final audit those were the same number, because the
interval was calibrated on the very rows its coverage was quoted on, where 80%
is arithmetic rather than evidence.

**Show a rejection:**

```bash
curl -s localhost:8000/api/v1/predict -H 'content-type: application/json' \
  -d '{"features": {"age": 1000000000000}}' | jq '.error.code'
```

Expected: `"validation_error"`. > "It used to return 200 and a confident
number. The config declared a valid age range and nothing read it."

---

## 5 — Explainability, in the dashboard (2 min)

Search **Haaland** → open the player.

Point at, in order: the prediction, the 80% interval with its measured
coverage, then the SHAP contributions.

> "These are additive in **log** space, because the model fits log value. So
> read them multiplicatively — this feature multiplies the prediction by 1.5,
> it doesn't add €1.5M. The API returns `effect_multiplier` precomputed so
> nobody has to remember that."

Then drag a **what-if** slider and watch the number move.

> "That round trip is 30 ms. It was 385 ms a week ago — 97% of it rebuilding a
> SHAP explainer for a model that never changes. Inference was 1.6% of the
> endpoint."

---

## 6 — Model selection (2 min)

```bash
cat docs/model_comparison.md
```

> "Eleven families. LightGBM ships, and it is *not* the lowest number — the
> stacked ensemble is. Two rules decide: the shipped model must produce a
> feature importance, and among families within one standard error of the best,
> the cheapest to deploy wins."

**The point worth landing.** On the prior-value variant the top three families
sit within €5,600 of each other, against a standard error an order of magnitude
larger. Sorting that table to the euro invites a decision the data cannot
support. That is Breiman's one-standard-error rule, from 1984.

---

## 7 — Reproducibility (1 min)

```bash
python scripts/build_features.py && md5 data/processed/training_table.parquet
```

Run twice; the hash matches. Verified across processes with
`PYTHONHASHSEED=random`, and both shipped models refit bit-identically and
reproduce their recorded test MAE to the cent.

---

## 8 — Close (1 min)

> "Six audit phases are in `plans/`. The most useful reading is the list of
> things that were built, measured and *thrown away*: club attendance, because
> season 2020 was played behind closed doors and the column measures a pandemic
> for one year in fourteen; calendar re-indexing, because the apparent 10% win
> was the forecast horizon collapsing from 141 days to 23; knowledge
> distillation, because a regularisation parameter the grid had never been
> allowed to try explained all of it.
>
> The repository's claim is not that the model is good. It is that every number
> in it is one you can reproduce, and every rejected idea has a p-value next to
> it."

---

## Fallback plan

| Fails | Do this |
|---|---|
| No internet | Nothing in the demo needs it. `make test` and every source walkthrough are local. |
| No Kaggle credentials / no data | Skip §3 and §4-live. `make test` runs on committed sample data; show the leakage module and the model cards instead. |
| No trained model | `docs/MODEL_CARD_*.md` and `docs/model_comparison.md` are committed and carry the real numbers. |
| Docker won't start | Run the API directly: `make serve`, then `npm run dev` in `frontend/`. |
| Dashboard broken | The API is the substance; `curl` + `jq` shows the same payloads and reads well on a projector. |
| Everything is broken | `plans/06-final-research-audit.md` is the strongest artefact in the repository and needs no runtime at all. |

**If asked "is this actually accurate?"** — the honest answer is that test MAE
is €2.07M on a target whose median is €1M, so relative error is large and
stated as such. What the project defends is not the accuracy; it is that the
number is measured on seasons the model never saw, that a random split would
read ~40% better and answer a question nobody deploying it would ask, and that
every limitation is quantified rather than mentioned.
