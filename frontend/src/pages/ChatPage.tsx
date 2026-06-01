import { useCallback } from 'react';
import { useChatStream } from '@/hooks/useChatStream';
import { StatusBar } from '@/components/StatusBar';
import { MessageList } from '@/components/MessageList';
import { ChatInput } from '@/components/ChatInput';

export function ChatPage() {
  const {
    messages,
    streamStatus,
    currentTool,
    inputValue,
    errorMessage,
    sendMessage,
    resumeStream,
    cancelStream,
    setInputValue,
  } = useChatStream();

  const handleApprove = useCallback(
    () => resumeStream({ approved: true }),
    [resumeStream],
  );
  const handleReject = useCallback(
    () => resumeStream({ approved: false }),
    [resumeStream],
  );

  const isStreaming = streamStatus === 'streaming';
  const isInterrupted = streamStatus === 'interrupted';
  const inputDisabled = isStreaming || isInterrupted;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--bg)',
    }}>
      <header style={{
        borderBottom: '1px solid var(--border)',
        padding: '12px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 'var(--text-sm)',
          color: 'var(--text-muted)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>
          agent harness
        </span>
        <StatusBar streamStatus={streamStatus} currentTool={currentTool} />
      </header>

      {errorMessage && (
        <div style={{
          padding: '8px 16px',
          background: 'rgba(248, 113, 113, 0.08)',
          borderBottom: '1px solid rgba(248, 113, 113, 0.2)',
          fontFamily: 'var(--mono)',
          fontSize: 'var(--text-xs)',
          color: 'var(--color-error)',
          flexShrink: 0,
        }}>
          {errorMessage}
        </div>
      )}

      <MessageList
        messages={messages}
        onApprove={handleApprove}
        onReject={handleReject}
      />

      <ChatInput
        value={inputValue}
        onChange={setInputValue}
        onSend={() => sendMessage(inputValue)}
        onCancel={cancelStream}
        streamStatus={streamStatus}
        disabled={inputDisabled}
      />
    </div>
  );
}
