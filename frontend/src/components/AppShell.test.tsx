import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { AppShell } from '@/components/AppShell';

describe('AppShell', () => {
  it('renders children inside the card', () => {
    render(<AppShell><div data-testid="child">content</div></AppShell>);
    expect(screen.getByTestId('child').textContent).toBe('content');
  });

  it('renders the card container', () => {
    render(<AppShell><span /></AppShell>);
    expect(screen.getByTestId('app-shell-card')).toBeInTheDocument();
  });
});