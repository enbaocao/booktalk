import type { Metadata } from "next";
import "./globals.css";
import Header from "../components/Header";

export const metadata: Metadata = {
  title: "White is for Witching - Narrator Analysis",
  description: "Analysis of narrative voices in Helen Oyeyemi's novel.",
  icons: {
    icon: [
      { url: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">📚</text></svg>' }
    ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
         <link rel="stylesheet" href="https://use.typekit.net/tlf2euo.css" />
      </head>
      <body className="pt-[5%]">
        <Header />
        <main>{children}</main>
      </body>
    </html>
  );
}
