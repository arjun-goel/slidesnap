import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SlideSnap Studio — Pre-Transition Lecture & Chalkboard Slide Extractor",
  description:
    "AI & Computer Vision engine that locks onto full lecture boards right before slide transitions, erases walking presenters via reverse-chronological pixel backfill, and compiles crisp A4 study PDFs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-sans antialiased selection:bg-sky-500 selection:text-slate-950">
        {children}
      </body>
    </html>
  );
}
