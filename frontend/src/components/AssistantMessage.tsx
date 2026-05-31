import { ThinkingBlock } from '@/components/ThinkingBlock';
import { ToolCallPill } from '@/components/ToolCallPill';
import { ToolResultPill } from '@/components/ToolResultPill';
import type { MessageBlock } from '@/types';

interface AssistantMessageProps {
  blocks: MessageBlock[];
  finalText?: string;
  isStreaming: boolean;
}

export function AssistantMessage({ blocks, finalText, isStreaming }: AssistantMessageProps) {
  return (
    <div style={{ padding: '4px 16px', maxWidth: '80%' }}>
      {blocks.map((block, i) => {
        switch (block.kind) {
          case 'thinking':
            return <ThinkingBlock key={i} text={block.text} />;
          case 'tool_call':
            return (
              <ToolCallPill
                key={block.call_id || i}
                name={block.name}
                args={block.args}
              />
            );
          case 'tool_result':
            return (
              <ToolResultPill
                key={block.call_id ? `${block.call_id}-result` : i}
                text={block.text}
                name={block.name}
                is_error={block.is_error}
              />
            );
        }
      })}
      {finalText && (
        <div
          style={{
            fontFamily: 'var(--sans)',
            fontSize: 'var(--text-base)',
            color: 'var(--text)',
            lineHeight: 1.7,
            marginTop: blocks.length > 0 ? '8px' : '0',
          }}
        >
          {finalText}
        </div>
      )}
      {isStreaming && blocks.length === 0 && (
        <span
          style={{
            fontFamily: 'var(--mono)',
            fontSize: 'var(--text-xs)',
            color: 'var(--text-muted)',
            animation: 'pulse 1.5s ease-in-out infinite',
          }}
        >
          ...
        </span>
      )}
    </div>
  );
}
