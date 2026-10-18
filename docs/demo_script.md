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
make test                   # 862 pass, 70 deselected, no credentials
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
on the content. So nothing is scraped: labels come from a Kaggle mirror its
publisher dedicates as CC0. That avoids the scraping; it does not settle the
training question, and the dataset card says so (audit F11).
That is the first slide because it explains an architecture decision, not a
legal footnote.

---

## 2 — It runs with nothing installed but Python (1 min)

```bash
make test
```

Expected: `862 passed, 70 deselected`, about fifteen seconds.

> "The 70 skips are integration tests that need a trained model. They skip
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
Removing it moved MAE by an amount the test could not distinguish from zero
(p = 0.49, a row-level test). Say "no detectable cost", not "no cost".

**The second leak story, found later and by someone else.** An independent
audit then found two leaks the project's own audits had missed: lineups dated
after the as-of date folded into role features, and competition strength
averaging labels published after the cutoff. The fix was to make every windowed
feature carry the date of its newest input, so the leakage stage checks each row
by value rather than trusting an argument.

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
asks for. `band_measured_coverage` is what intervals for predictions in this
value band achieved on the test seasons, with the rows behind it, and the
dashboard words it as a record for a group, not a probability for this player.
The quantiles are widened by a rule fixed on pre-test seasons; overall coverage
is 79.1% for this model, and that figure is retrospective, because the widening
was motivated by an earlier 75.1% that had already been seen.

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

> "That round trip is about 25 ms, and it is a requirement rather than an
> accident: a model family only ships if prediction plus explanation is under
> 100 ms on validation rows. A Random Forest once shipped at 440 ms."

---

## 6 — Model selection (2 min)

```bash
cat docs/model_comparison.md
```

> "Eleven families and a written rule. To ship, a family must produce feature
> importances and explain a prediction within 100 ms; among those within one
> standard error of the best validation MAE, the smallest deployed footprint
> wins. No test number is involved, and the artifact records every input, so a
> test recomputes the choice. LightGBM wins both variants."

**The point worth landing.** On the prior-value variant the top six families
sit within €18,500 of each other against a standard error of about €27,000.
Sorting that table to the euro invites a decision the data cannot support.
That is Breiman's one-standard-error rule, from 1984.

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

> "The audit record is in `plans/`, including an outside audit's twenty
> findings and what was done about each. The most useful reading is the list of
> things that were built, measured and *thrown away*: club attendance, because
> season 2020 was played behind closed doors and the column measures a pandemic
> for one year in fourteen; calendar re-indexing, because the apparent 10% win
> was the forecast horizon collapsing from 141 days to 23; knowledge
> distillation, because a regularisation parameter the grid had never been
> allowed to try explained all of it.
>
> The repository's claim is not that the model is good. It is that every number
> in it is one you can reproduce, that it says what each number is evidence of —
> the test figures are retrospective — and that a genuinely unseen season has
> been frozen to check it."

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
is €2.06M on a target whose median is about €1M, so relative error is large and
stated as such, and that the figure is retrospective: the test seasons were not
trained on but were inspected during development. What the project defends is
that the number comes from later seasons under a checked point-in-time rule,
that a random split would read ~30% better and answer a question nobody
deploying it would ask, and that an independent check is set up rather than
claimed.

**If asked "can I use this commercially?"** — no. The data's rights are
uncertain (`docs/DATASET_CARD.md`, "Data rights").
