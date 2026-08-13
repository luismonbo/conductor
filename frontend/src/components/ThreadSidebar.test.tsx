import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ThreadSidebar } from './ThreadSidebar';

const threads = [
  { thread_id: 't1', last_at: '2026-08-13T10:00:00Z', runs: 2, title: 'groceries plan' },
  { thread_id: 't2', last_at: '2026-08-12T09:00:00Z', runs: 1, title: '' },
];

describe('ThreadSidebar', () => {
  it('lists titles, falls back to short id, marks active', () => {
    render(
      <ThreadSidebar threads={threads} activeThreadId="t1" onSelect={() => {}} onNew={() => {}} />,
    );
    expect(screen.getByText('groceries plan')).toBeInTheDocument();
    expect(screen.getByText('t2')).toBeInTheDocument(); // fallback label
    expect(screen.getByText('groceries plan').closest('button')).toHaveAttribute(
      'aria-current', 'true',
    );
  });

  it('fires onSelect and onNew', async () => {
    const onSelect = vi.fn();
    const onNew = vi.fn();
    render(
      <ThreadSidebar threads={threads} activeThreadId={null} onSelect={onSelect} onNew={onNew} />,
    );
    await userEvent.click(screen.getByText('groceries plan'));
    expect(onSelect).toHaveBeenCalledWith('t1');
    await userEvent.click(screen.getByRole('button', { name: /new chat/i }));
    expect(onNew).toHaveBeenCalled();
  });
});
