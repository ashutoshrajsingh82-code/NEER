import "./globals.css";

// Design-system typefaces: Inter for UI/body text, JetBrains Mono for
// data/monospace text (coordinates, depth values, timestamps, code).
//
// Loaded via a standard <link> (not next/font) so the build never depends on
// reaching fonts.googleapis.com at build time — offline/CI builds still
// succeed and simply fall back to the system font stack defined in
// tailwind.config.js until the stylesheet loads in the browser.

export const metadata = {
  title: "NEER — SIH26066",
  description:
    "NEER: Neural Embedding based Estimation and Reconstruction — SIH26066, MoES/INCOIS",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-bg-base text-text-primary font-sans antialiased">
        {children}
      </body>
    </html>
  );
}