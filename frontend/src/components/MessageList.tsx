import { useEffect, useRef } from 'react';
import { UserMessage } from '@/components/UserMessage';
import { AssistantMessage } from '@/components/AssistantMessage';
import type { ConversationMessage } from '@/types';

interface MessageListProps {
  messages: ConversationMessage[];
  onApprove: () => void;
  onReject: () => void;
  onFeedback: (text: string) => void;
  onMemoryApprove: () => void;
  onMemoryDeny: () => void;
}

export function MessageList({ messages, onApprove, onReject, onFeedback, onMemoryApprove, onMemoryDeny }: MessageListProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom < 100) {
      const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      bottomRef.current?.scrollIntoView({ behavior: prefersReduced ? 'auto' : 'smooth' });
    }
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 px-4 text-center">
        <span className="font-mono text-2xl text-accent">&gt;_</span>
        <p className="font-mono text-sm text-fg-muted">Send a message to start the conversation</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex flex-1 flex-col gap-2 overflow-y-auto py-4"
    >
      {messages.map((msg) =>
        msg.role === 'user' ? (
          <UserMessage key={msg.id} text={msg.text} />
        ) : (
          <AssistantMessage
            key={msg.id}
            blocks={msg.blocks}
            finalText={msg.finalText}
            isStreaming={msg.isStreaming}
            interruptPayload={msg.interruptPayload}
            onApprove={onApprove}
            onReject={onReject}
            onFeedback={onFeedback}
            onMemoryApprove={onMemoryApprove}
            onMemoryDeny={onMemoryDeny}
          />
        )
      )}
      <div ref={bottomRef} />
    </div>
  );
}
