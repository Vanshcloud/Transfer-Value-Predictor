import Link from "next/link";

export default function AboutPage() {
  return (
    <div className="max-w-3xl space-y-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">About</h1>
        <p className="mt-3 text-lg leading-relaxed text-slate-600 dark:text-slate-300">
          A market-value model for professional footballers, built so that the
          way it can be wrong is as visible as the number it produces.
        </p>
      </div>

      <section>
        <h2 className="text-xl font-semibold">Where the data comes from</h2>
        <p className="mt-2 leading-relaxed text-slate-600 dark:text-slate-400">
          Labels and appearances come from the CC0-licensed Kaggle mirror of
          Transfermarkt data. Transfermarkt itself is{" "}
          <strong>never scraped</strong>: its terms prohibit automated access
          and, separately, prohibit using the content to train machine-learning
          models. Respecting those terms means not doing it, so the project uses
          the openly licensed mirror instead.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold">How it is evaluated</h2>
        <p className="mt-2 leading-relaxed text-slate-600 dark:text-slate-400">
          Every reported number comes from a <strong>temporal split</strong>:
          the model trains on seasons up to 2021, tunes on 2022, and is measured
          once on 2023 onward. That is about 40% worse on euro error than a
          random split would report — and it is the only arrangement that
          resembles being asked about a season that has not happened yet.
        </p>
        <p className="mt-3 leading-relaxed text-slate-600 dark:text-slate-400">
          Leakage is checked as a pipeline stage, not only in tests: feature
          timestamps against label dates, current-state columns on historical
          rows, the target reaching the feature matrix, duplicate
          player-seasons, and overlap between splits. A leak stops a run rather
          than quietly improving a metric.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Two models, on purpose</h2>
        <p className="mt-2 leading-relaxed text-slate-600 dark:text-slate-400">
          Prior market value dominates every other feature — including it moves
          R² from about 0.44 to about 0.78. That makes for an accurate model
          that mostly repeats what the market already said. Both are served and
          labelled honestly: <strong>performance only</strong> is the one that
          can disagree with the market, and <strong>with prior value</strong> is
          the one that tracks it. Publishing only the second would be
          technically true and practically useless.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold">What the intervals mean</h2>
        <p className="mt-2 leading-relaxed text-slate-600 dark:text-slate-400">
          The interval shown with each prediction is not a probability from the
          model — a gradient booster does not produce one, and inventing a
          confidence score would be dishonest. It is measured from the
          model&apos;s own residuals on held-out seasons, for the value band the
          prediction falls into. It is wide. That is the finding, not a defect.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Limitations worth knowing</h2>
        <ul className="mt-2 list-disc space-y-2 pl-5 leading-relaxed text-slate-600 dark:text-slate-400">
          <li>
            Market value is a community-maintained estimate, not an observed
            price. The model reproduces that consensus, including wherever it is
            biased.
          </li>
          <li>
            Error grows with value. A mid-table MAE conceals much larger
            absolute misses at the top of the market.
          </li>
          <li>
            Appearance coverage begins in 2012, so career-length features are
            left-censored and capped.
          </li>
          <li>
            Seasons run August to July, so leagues on a spring–autumn calendar
            are represented less faithfully.
          </li>
        </ul>
      </section>

      <p className="text-sm text-slate-500 dark:text-slate-400">
        <Link
          href="/model"
          className="text-sky-600 hover:underline dark:text-sky-400"
        >
          See the measured numbers →
        </Link>
      </p>
    </div>
  );
}
