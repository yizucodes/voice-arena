import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Self-Healing Voice Agent | Autonomous AI Testing",
  description: "The first voice agent that fixes itself. Test → Fail → GPT-4o Fix → Re-Test → Loop until perfect.",
  keywords: ["voice agent", "AI", "self-healing", "GPT-4o", "ElevenLabs", "Daytona"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {/* Animated background orbs */}
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        
        {/* Grid overlay */}
        <div className="fixed inset-0 bg-grid pointer-events-none" />
        
        {/* Main content */}
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  );
}
