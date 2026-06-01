import { useRef, useEffect } from 'react';
import type { StreamStatus } from '@/types';

interface ChatInputProps {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onCancel: () => void;
  onReject: () => void;
  streamStatus: StreamStatus;
  disabled: boolean;
}

export function ChatInput({
  value,
  onChange,
  onSend,
  onCancel,
  onReject,
  streamStatus,
  disabled,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const lineHeight = 20;
    const maxHeight = lineHeight * 6;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, [value]);

  useEffect(() => {
    if (streamStatus === 'idle' || streamStatus === 'done' || streamStatus === 'error') {
      textareaRef.current?.focus();
    }
  }, [streamStatus]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSend();
    }
  };

  const isStreaming = streamStatus === 'streaming';
  const isInterrupted = streamStatus === 'interrupted';
  const showActionButton = isStreaming || isInterrupted;

  return (
    <div
      style={{
        borderTop: '1px solid var(--border)',
        padding: '12px 16px',
        display: 'flex',
        gap: '8px',
        alignItems: 'flex-end',
        background: 'var(--bg)',
      }}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Send a message…"
        aria-label="Message"
        rows={1}
        style={{
          flex: 1,
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          padding: '10px 12px',
          color: 'var(--text)',
          fontFamily: 'var(--sans)',
          fontSize: 'var(--text-base)',
          resize: 'none',
          outline: 'none',
          lineHeight: '20px',
          minHeight: '40px',
          transition: 'border-color 0.15s',
        }}
      />
      {showActionButton ? (
        <button
          onClick={isInterrupted ? onReject : onCancel}
          style={{
            padding: '10px 16px',
            background: 'transparent',
            border: '1px solid var(--color-error)',
            borderRadius: '8px',
            color: 'var(--color-error)',
            fontFamily: 'var(--mono)',
            fontSize: 'var(--text-sm)',
            cursor: 'pointer',
            whiteSpace: 'nowrap',
            transition: 'background 0.15s',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(248,113,113,0.1)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
        >
          {isInterrupted ? 'Reject' : 'Cancel'}
        </button>
      ) : (
        <button
          onClick={onSend}
          disabled={disabled || !value.trim()}
          style={{
            padding: '10px 16px',
            background: 'transparent',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--mono)',
            fontSize: 'var(--text-sm)',
            cursor: disabled || !value.trim() ? 'not-allowed' : 'pointer',
            whiteSpace: 'nowrap',
            transition: 'border-color 0.15s, color 0.15s',
            opacity: disabled || !value.trim() ? 0.4 : 1,
          }}
          onMouseEnter={(e) => {
            if (!disabled && value.trim()) {
              e.currentTarget.style.borderColor = 'var(--accent)';
              e.currentTarget.style.color = 'var(--accent)';
            }
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'var(--border)';
            e.currentTarget.style.color = 'var(--text-muted)';
          }}
        >
          Send
        </button>
      )}
    </div>
  );
}
