import { useCallback, useEffect, useState } from 'react';
import { useChatStream } from '@/hooks/useChatStream';
import { fetchModels, fetchThreads } from '@/api';
import type { ThreadSummary } from '@/types';
import { StatusBar } from '@/components/StatusBar';
import { MessageList } from '@/components/MessageList';
import { ChatInput } from '@/components/ChatInput';
import { ModelPicker } from '@/components/ModelPicker';
import { ThreadSidebar } from '@/components/ThreadSidebar';

export function ChatPage() {
  const {
    messages,
    streamStatus,
    currentTool,
    threadId,
    inputValue,
    errorMessage,
    interruptPayload,
    sendMessage,
    resumeStream,
    cancelStream,
    setInputValue,
    loadThread,
    newThread,
  } = useChatStream();

  const isStreaming = streamStatus === 'streaming';
  const isInterrupted = streamStatus === 'interrupted';
  const inputDisabled = isStreaming || isInterrupted;

  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [threads, setThreads] = useState<ThreadSummary[]>([]);

  useEffect(() => {
    fetchModels()
      .then((r) => {
        setModels(r.models);
        const stored = localStorage.getItem('harness:model:last');
        setSelectedModel(stored && r.models.includes(stored) ? stored : (r.default ?? ''));
      })
      .catch(() => setModels([]));
  }, []);

  // Refresh the sidebar on mount (status starts idle) and after each run.
  useEffect(() => {
    if (streamStatus === 'idle' || streamStatus === 'done') {
      fetchThreads()
        .then((r) => setThreads(r.threads))
        .catch(() => { /* sidebar just stays as-is */ });
    }
  }, [streamStatus]);

  const handleModelChange = useCallback((m: string) => {
    setSelectedModel(m);
    localStorage.setItem('harness:model:last', m);
    if (threadId) localStorage.setItem(`harness:model:${threadId}`, m);
  }, [threadId]);

  const handleSelectThread = useCallback((id: string) => {
    loadThread(id)
      .then(() => {
        const stored = localStorage.getItem(`harness:model:${id}`);
        if (stored) {
          setSelectedModel((current) =>
            models.includes(stored) ? stored : current,
          );
        }
      })
      .catch(() => { /* thread stays unloaded; sidebar unchanged */ });
  }, [loadThread, models]);

  // Tool approval (legacy shape)
  const handleApprove = useCallback(() => resumeStream({ approved: true }), [resumeStream]);
  const handleReject = useCallback(() => resumeStream({ approved: false }), [resumeStream]);

  // Memory & generic feedback (new action shape)
  const handleMemoryApprove = useCallback(() => resumeStream({ action: 'approve' }), [resumeStream]);
  const handleMemoryDeny = useCallback(() => resumeStream({ action: 'deny' }), [resumeStream]);
  const handleFeedback = useCallback(
    (text: string) => resumeStream({ action: 'feedback', feedback: text }, text),
    [resumeStream],
  );

  void interruptPayload; // used via MessageList → AssistantMessage discriminated union

  return (
    <div style={{
      display: 'flex',
      height: '100%',
    }}>
      <ThreadSidebar
        threads={threads}
        activeThreadId={threadId}
        onSelect={handleSelectThread}
        onNew={newThread}
      />

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        flex: 1,
        minWidth: 0,
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <ModelPicker
            models={models}
            value={selectedModel}
            onChange={handleModelChange}
            disabled={inputDisabled}
          />
          <StatusBar streamStatus={streamStatus} currentTool={currentTool} />
        </div>
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
        onFeedback={handleFeedback}
        onMemoryApprove={handleMemoryApprove}
        onMemoryDeny={handleMemoryDeny}
      />

      <ChatInput
        value={inputValue}
        onChange={setInputValue}
        onSend={() => sendMessage(inputValue, selectedModel || undefined)}
        onCancel={cancelStream}
        streamStatus={streamStatus}
        disabled={inputDisabled}
      />
      </div>
    </div>
  );
}
