import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/Header";

export const metadata: Metadata = {
  title: "White is for Witching - Narrator Analysis",
  description: "Analysis of narrative voices in Helen Oyeyemi's novel.",
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
