# Roadmap

Everything here comes from a measurement that has already been taken. Six audit
phases produced a list of things that are genuinely open, and a longer list of
things that were tried and rejected — the second list is in
[`plans/`](plans/) and is deliberately not repeated as future work. An idea that
was measured and did not help is finished, not pending.

Nothing below has a date. This is a single-author project and a roadmap with
dates on it would be fiction.

---

## Open: methodology

### Make the repeated-season protocol the headline result

**Why.** Selection currently uses a single validation season (2022) and the
test seasons are scored once. That is the right discipline for deployment and
thin for a paper: it means the headline number rests on one split.

Every *comparison* in phases 16–18 already used the stronger protocol —
expanding-window held-out seasons 2018–2022 at three seeds, errors pooled and
compared with a paired *t*-test. The result should be reported the same way.

**What it would take.** Report the pooled figure alongside the single-split one
rather than instead of it; the single split still answers "what happens on
seasons after everything I trained on", which the pooled figure does not.

**Blocked on.** Nothing but time — roughly 30× the current training cost.

### Compare against an external baseline

**Why.** "Good" is currently established against this project's own earlier
selves. Nothing here is measured against a published market-value model, so the
absolute quality of €2.07M MAE is unanchored.

**Blocked on.** Finding a published model evaluated on a comparable panel with
a comparable split. Most published football-valuation work uses random splits,
which is not a fair comparison in either direction.

---

## Open: data

### Move the label anchor per competition without moving the horizon

**Why.** Seasons are indexed August–July, and 6.27% of fixtures disagree with
their own competition's declared season — Brazilian Série A, the J1 League,
Eliteserien, Allsvenskan, the K-League, MLS, and every European qualifying
round played in July. Rows with more than 5% of their minutes misfiled carry
0.04–0.12 more median relative error *within every value band*, and that
survives controlling for the population those July fixtures select
([`plans/06`](plans/06-final-research-audit.md) §5).

**Why it is not done.** It was built and measured *worse*. Anchoring on each
competition's last fixture moves the as-of date earlier on 93% of rows and cuts
the median label horizon from 141 days to 23 — a different and much easier
question, which is where the apparent 10% win came from. Holding the horizon
fixed, the gain is null for the scouting model and significantly worse
(−€68,996, p = 0.0005) with prior value, on 6.5% fewer rows.

**Blocked on.** A per-competition calendar table that would let the anchor move
without the horizon moving with it. This dataset has none.

### Pre-2012 coverage

**Closed, not open.** 40,339 reconstructable rows were built and added; only 7
of 54 features exist on such a row and the pooled effect was p = 0.73. Recorded
here because it is the most common suggestion this project receives.

---

## Open: engineering

### Export the model to a format that cannot execute

**Why.** `joblib.load` unpickles, so the service must only ever load artifacts
it produced ([`SECURITY.md`](SECURITY.md)). ONNX or the booster's own text dump
would remove that assumption instead of documenting it.

**The trade.** The fitted preprocessing currently travels inside the same
object as the estimator, which is the single property preventing training and
serving from drifting apart. Splitting them to gain a safer format would
reintroduce the failure this project spends most of its effort avoiding, so it
is a deliberate trade rather than a pending fix.

### Postgres and Redis

**Why not yet.** The storage layer already sits behind a `Protocol` and CI
enforces that boundary, so swapping DuckDB for Postgres is an implementation
detail rather than a refactor. It waits for a measurement that justifies it:
startup is 1.27 s and steady-state RSS is 660 MB, which is not a problem.

### Rate limiting

Out of scope for the service itself and documented as a deployment requirement.
Prediction is CPU-bound at ~30 ms; put a reverse proxy in front of it.

---

## Not planned

- **Scraping Transfermarkt.** Its Terms of Use §11.1 prohibit automated access
  *and* separately prohibit training on the content. This is a legal
  constraint, not a technical one, and no amount of engineering changes it.
- **A bigger model.** Eleven families were searched and the spread across the
  top four is a fraction of one standard error. The signal in this data is in
  the features, not the estimator.
- **`clubs.csv`.** Its interesting columns are all *current* state; joining
  today's squad value to a 2013 row states a fact from 2026.
- **Anything already measured and rejected**, which is most plausible ideas.
  Check [`plans/06`](plans/06-final-research-audit.md) before proposing a
  feature — attendance, transfer history, club trajectory, three alternative
  targets and knowledge distillation are all there with their numbers.
