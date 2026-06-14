import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ApprovalCard } from '@/components/ApprovalCard';
import type { ToolApprovalPayload } from '@/types';

const singleToolPayload: ToolApprovalPayload = {
  mode: 'approval',
  tool_calls: [
    { name: 'calculator', args: { expression: '2+2' }, call_id: 'c1' },
  ],
};

const multiToolPayload: ToolApprovalPayload = {
  mode: 'approval',
  tool_calls: [
    { name: 'read_file', args: { path: 'data.txt' }, call_id: 'c1' },
    { name: 'write_file', args: { path: 'out.txt', content: 'hi' }, call_id: 'c2' },
  ],
};

const noop = () => {};

describe('ApprovalCard', () => {
  it('renders the tool name', () => {
    render(<ApprovalCard payload={singleToolPayload} onApprove={noop} onReject={noop} onFeedback={noop} />);
    expect(screen.getByText('calculator')).toBeTruthy();
  });

  it('renders tool args as formatted JSON', () => {
    const { container } = render(
      <ApprovalCard payload={singleToolPayload} onApprove={noop} onReject={noop} onFeedback={noop} />,
    );
    const pre = container.querySelector('pre');
    expect(pre?.textContent).toContain('"expression"');
    expect(pre?.textContent).toContain('"2+2"');
  });

  it('renders all tool names for multi-tool payloads', () => {
    render(<ApprovalCard payload={multiToolPayload} onApprove={noop} onReject={noop} onFeedback={noop} />);
    expect(screen.getByText('read_file')).toBeTruthy();
    expect(screen.getByText('write_file')).toBeTruthy();
  });

  it('calls onApprove once when Approve is clicked', () => {
    const onApprove = vi.fn();
    render(<ApprovalCard payload={singleToolPayload} onApprove={onApprove} onReject={noop} onFeedback={noop} />);
    fireEvent.click(screen.getByRole('button', { name: /approve/i }));
    expect(onApprove).toHaveBeenCalledOnce();
  });

  it('calls onReject once when Deny is clicked', () => {
    const onReject = vi.fn();
    render(<ApprovalCard payload={singleToolPayload} onApprove={noop} onReject={onReject} onFeedback={noop} />);
    fireEvent.click(screen.getByRole('button', { name: /deny/i }));
    expect(onReject).toHaveBeenCalledOnce();
  });

  it('disables all controls after Approve is clicked', () => {
    render(<ApprovalCard payload={singleToolPayload} onApprove={noop} onReject={noop} onFeedback={noop} />);
    fireEvent.click(screen.getByRole('button', { name: /approve/i }));
    expect(screen.getByRole('button', { name: /approve/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /deny/i })).toBeDisabled();
  });

  it('disables all controls after Deny is clicked', () => {
    render(<ApprovalCard payload={singleToolPayload} onApprove={noop} onReject={noop} onFeedback={noop} />);
    fireEvent.click(screen.getByRole('button', { name: /deny/i }));
    expect(screen.getByRole('button', { name: /approve/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /deny/i })).toBeDisabled();
  });

  it('does not fire onApprove twice on double-click', () => {
    const onApprove = vi.fn();
    render(<ApprovalCard payload={singleToolPayload} onApprove={onApprove} onReject={noop} onFeedback={noop} />);
    const btn = screen.getByRole('button', { name: /approve/i });
    fireEvent.click(btn);
    fireEvent.click(btn);
    expect(onApprove).toHaveBeenCalledOnce();
  });

  it('calls onFeedback with input text when Send is clicked', () => {
    const onFeedback = vi.fn();
    render(<ApprovalCard payload={singleToolPayload} onApprove={noop} onReject={noop} onFeedback={onFeedback} />);
    const input = screen.getByPlaceholderText(/type feedback/i);
    fireEvent.change(input, { target: { value: 'use snake_case instead' } });
    fireEvent.click(screen.getByRole('button', { name: /^send$/i }));
    expect(onFeedback).toHaveBeenCalledWith('use snake_case instead');
  });

  it('calls onFeedback on Enter key in feedback input', () => {
    const onFeedback = vi.fn();
    render(<ApprovalCard payload={singleToolPayload} onApprove={noop} onReject={noop} onFeedback={onFeedback} />);
    const input = screen.getByPlaceholderText(/type feedback/i);
    fireEvent.change(input, { target: { value: 'use camelCase' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onFeedback).toHaveBeenCalledWith('use camelCase');
  });
});
