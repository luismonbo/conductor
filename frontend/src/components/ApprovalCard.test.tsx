import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ApprovalCard } from '@/components/ApprovalCard';
import type { InterruptPayload } from '@/types';

const singleToolPayload: InterruptPayload = {
  mode: 'approval',
  tool_calls: [
    { name: 'calculator', args: { expression: '2+2' }, call_id: 'c1' },
  ],
};

const multiToolPayload: InterruptPayload = {
  mode: 'approval',
  tool_calls: [
    { name: 'read_file', args: { path: 'data.txt' }, call_id: 'c1' },
    { name: 'write_file', args: { path: 'out.txt', content: 'hi' }, call_id: 'c2' },
  ],
};

describe('ApprovalCard', () => {
  it('renders the tool name', () => {
    render(<ApprovalCard payload={singleToolPayload} onApprove={() => {}} onReject={() => {}} />);
    expect(screen.getByText('calculator')).toBeTruthy();
  });

  it('renders tool args as formatted JSON', () => {
    const { container } = render(
      <ApprovalCard payload={singleToolPayload} onApprove={() => {}} onReject={() => {}} />,
    );
    const pre = container.querySelector('pre');
    expect(pre?.textContent).toContain('"expression"');
    expect(pre?.textContent).toContain('"2+2"');
  });

  it('renders all tool names for multi-tool payloads', () => {
    render(<ApprovalCard payload={multiToolPayload} onApprove={() => {}} onReject={() => {}} />);
    expect(screen.getByText('read_file')).toBeTruthy();
    expect(screen.getByText('write_file')).toBeTruthy();
  });

  it('calls onApprove once when Approve is clicked', () => {
    const onApprove = vi.fn();
    render(<ApprovalCard payload={singleToolPayload} onApprove={onApprove} onReject={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /approve/i }));
    expect(onApprove).toHaveBeenCalledOnce();
  });

  it('calls onReject once when Reject is clicked', () => {
    const onReject = vi.fn();
    render(<ApprovalCard payload={singleToolPayload} onApprove={() => {}} onReject={onReject} />);
    fireEvent.click(screen.getByRole('button', { name: /reject/i }));
    expect(onReject).toHaveBeenCalledOnce();
  });

  it('disables both buttons after Approve is clicked', () => {
    render(<ApprovalCard payload={singleToolPayload} onApprove={() => {}} onReject={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /approve/i }));
    expect(screen.getByRole('button', { name: /approve/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /reject/i })).toBeDisabled();
  });

  it('disables both buttons after Reject is clicked', () => {
    render(<ApprovalCard payload={singleToolPayload} onApprove={() => {}} onReject={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /reject/i }));
    expect(screen.getByRole('button', { name: /approve/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /reject/i })).toBeDisabled();
  });

  it('does not fire onApprove twice on double-click', () => {
    const onApprove = vi.fn();
    render(<ApprovalCard payload={singleToolPayload} onApprove={onApprove} onReject={() => {}} />);
    const btn = screen.getByRole('button', { name: /approve/i });
    fireEvent.click(btn);
    fireEvent.click(btn);
    expect(onApprove).toHaveBeenCalledOnce();
  });
});