import './globals.css';
import React from 'react';

export const metadata = {
  title: 'Token-Diet Context Compressor',
  description: 'Post-retrieval dynamic sentence-level pruning and RAG optimizer',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-background text-foreground antialiased">
        {children}
      </body>
    </html>
  );
}
