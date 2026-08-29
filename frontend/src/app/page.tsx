import Link from "next/link";

export default function Home() {
  return (
    <div className="space-y-12">
      <section className="pt-6">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          What is a footballer worth,
          <br />
          <span className="text-sky-600 dark:text-sky-400">and why?</span>
        </h1>
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-slate-600 dark:text-slate-300">
          A market-value model over 36,880 player-seasons, built so that every
          prediction can be taken apart: what raised it, what lowered it, how
          wrong the model usually is, and which comparable players it is
          reasoning from.
        </p>
        <div className="mt-7 flex flex-wrap gap-3">
          <Link
            href="/players"
            className="rounded-lg bg-sky-600 px-5 py-2.5 font-medium text-white hover:bg-sky-700"
          >
            Find a player
          </Link>
          <Link
            href="/model"
            className="rounded-lg border border-slate-300 px-5 py-2.5 font-medium hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            How good is it?
          </Link>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {[
          {
            title: "Predict",
            body: "A value for any covered player-season, with an interval measured from the model's own errors rather than an invented confidence score.",
          },
          {
            title: "Explain",
            body: "Every prediction decomposed into what pushed it up and what pulled it down, with the exact multiplicative effect of each feature.",
          },
          {
            title: "Explore",
            body: "Change goals, minutes or age and watch the valuation move. Compare players. See which seasons the model considers alike.",
          },
        ].map((item) => (
          <div
            key={item.title}
            className="rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900/60"
          >
            <h2 className="font-semibold">{item.title}</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
              {item.body}
            </p>
          </div>
        ))}
      </section>

      <section className="rounded-xl border border-amber-200 bg-amber-50 p-5 dark:border-amber-900/50 dark:bg-amber-950/20">
        <h2 className="font-semibold text-amber-900 dark:text-amber-200">
          Read the numbers honestly
        </h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-amber-900/90 dark:text-amber-200/80">
          Everything reported here is measured on <strong>seasons the model never
          saw</strong>. That is roughly 60% worse on error than a random split would
          show, and it is the only number that describes predicting a season that has
          not happened yet. Two models are served: one that has never been told the
          market&apos;s opinion — useful for finding disagreement — and one anchored to
          a known valuation, which is more accurate and correspondingly less
          interesting.
        </p>
      </section>
    </div>
  );
}
