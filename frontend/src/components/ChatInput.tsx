import { useRef, useEffect } from 'react';
import type { StreamStatus } from '@/types';

interface ChatInputProps {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onCancel: () => void;
  streamStatus: StreamStatus;
  disabled: boolean;
  placeholder?: string;
}

export function ChatInput({
  value,
  onChange,
  onSend,
  onCancel,
  streamStatus,
  disabled,
  placeholder,
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
  const resolvedPlaceholder = placeholder ?? (isInterrupted ? 'Waiting for approval…' : 'Send a message…');

  return (
    <div className="flex flex-shrink-0 items-end gap-2 border-t border-border bg-canvas px-4 py-3">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder={resolvedPlaceholder}
        aria-label="Message"
        rows={1}
        className="min-h-[40px] flex-1 resize-none rounded-md border border-border bg-surface px-3 py-2.5 font-sans text-base leading-5 text-fg outline-none transition placeholder:text-fg-muted focus:border-accent disabled:opacity-50"
      />
      {isStreaming ? (
        <button
          type="button"
          onClick={onCancel}
          className="whitespace-nowrap rounded-md border border-error px-4 py-2.5 font-mono text-sm text-error transition hover:bg-error/10 active:scale-[0.98]"
        >
          Cancel
        </button>
      ) : (
        !disabled && (
          <button
            type="button"
            onClick={onSend}
            disabled={disabled || !value.trim()}
            className="whitespace-nowrap rounded-md border border-border px-4 py-2.5 font-mono text-sm text-fg-muted transition enabled:hover:border-accent enabled:hover:text-accent disabled:cursor-not-allowed disabled:opacity-40 active:enabled:scale-[0.98]"
          >
            Send
          </button>
        )
      )}
    </div>
  );
}
