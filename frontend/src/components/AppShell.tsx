import type { ReactNode } from 'react';

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div
      data-testid="app-shell-card"
      className="flex h-[100dvh] w-full overflow-hidden bg-canvas font-sans text-fg"
    >
      <a
        href="#main-content"
        className="skip-link rounded-md bg-accent px-3 py-2 font-mono text-xs font-medium text-canvas"
      >
        Skip to conversation
      </a>
      {children}
    </div>
  );
}
