import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Nav from "@/components/Nav";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Transfer Value Predictor",
  description:
    "Predict the market value of professional footballers, and explain every prediction.",
};

/**
 * Applied before paint so a dark-mode reload does not flash white. An effect
 * would run after the first render, which is exactly too late.
 */
const THEME_SCRIPT = `
try {
  var stored = localStorage.getItem('theme');
  var dark = stored ? stored === 'dark'
    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.classList.toggle('dark', dark);
} catch (e) {}
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:m-2 focus:rounded focus:bg-white focus:p-2 focus:text-slate-900"
        >
          Skip to content
        </a>
        <Nav />
        <main id="main" className="mx-auto max-w-6xl px-4 py-8">
          {children}
        </main>
        <footer className="mx-auto max-w-6xl px-4 py-10 text-xs text-slate-500 dark:text-slate-400">
          Labels from the CC0 Kaggle mirror of Transfermarkt data. Transfermarkt
          is never scraped. Metrics are measured on held-out seasons.
        </footer>
      </body>
    </html>
  );
}
