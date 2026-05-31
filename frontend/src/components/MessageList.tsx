import { useEffect, useRef } from 'react';
import { UserMessage } from '@/components/UserMessage';
import { AssistantMessage } from '@/components/AssistantMessage';
import type { ConversationMessage } from '@/types';

interface MessageListProps {
  messages: ConversationMessage[];
}

export function MessageList({ messages }: MessageListProps) {
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

  return (
    <div
      ref={containerRef}
      style={{
        flex: 1,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        padding: '16px 0',
      }}
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
          />
        )
      )}
      <div ref={bottomRef} />
    </div>
  );
}
