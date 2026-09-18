import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "NEER — SIH26066",
  description:
    "NEER: Neural Embedding based Estimation and Reconstruction — SIH26066, MoES/INCOIS",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "system-ui, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
