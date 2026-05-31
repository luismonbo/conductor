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
    sendMessage,
    cancelStream,
    setInputValue,
  } = useChatStream();

  const isStreaming = streamStatus === 'streaming';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--bg)',
    }}>
      {/* Header */}
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

      {/* Message list — takes all remaining space */}
      <MessageList messages={messages} />

      {/* Input */}
      <ChatInput
        value={inputValue}
        onChange={setInputValue}
        onSend={() => sendMessage(inputValue)}
        onCancel={cancelStream}
        isStreaming={isStreaming}
        disabled={isStreaming}
      />
    </div>
  );
}
