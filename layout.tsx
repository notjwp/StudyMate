import type {Metadata} from 'next';
import './globals.css';
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from '@/components/providers/ThemeProvider';
import { Exo_2 } from 'next/font/google';

const exo2 = Exo_2({
  subsets: ['latin'],
  variable: '--font-exo2',
});

export const metadata: Metadata = {
  title: 'StudyMate: AI-Powered PDF Q&A',
  description: 'Upload PDFs and get answers, summaries, flashcards, and quizzes.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Source+Code+Pro&display=swap" rel="stylesheet" />
      </head>
      <body className={`${exo2.variable} font-body antialiased bg-background text-foreground`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
