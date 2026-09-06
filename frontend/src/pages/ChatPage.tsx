import { useCallback, useEffect, useState } from 'react';
import { useChatStream } from '@/hooks/useChatStream';
import { fetchModels, fetchThreads } from '@/api';
import type { ThreadSummary } from '@/types';
import { StatusBar } from '@/components/StatusBar';
import { MessageList } from '@/components/MessageList';
import { ChatInput } from '@/components/ChatInput';
import { ModelPicker } from '@/components/ModelPicker';
import { ThreadSidebar } from '@/components/ThreadSidebar';
import { SidebarSimple } from '@phosphor-icons/react';

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
  const [sidebarOpen, setSidebarOpen] = useState(false);

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
    setSidebarOpen(false);
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

  const handleNewThread = useCallback(() => {
    setSidebarOpen(false);
    newThread();
  }, [newThread]);

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
    <>
      <ThreadSidebar
        threads={threads}
        activeThreadId={threadId}
        onSelect={handleSelectThread}
        onNew={handleNewThread}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex h-full min-w-0 flex-1 flex-col">
        <header className="flex flex-shrink-0 items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open conversations"
              className="-ml-1.5 rounded-md p-1.5 text-fg-muted transition hover:text-fg active:scale-[0.98] md:hidden"
            >
              <SidebarSimple size={18} />
            </button>
            <span className="font-mono text-sm uppercase tracking-wider text-fg-muted">
              agent harness
            </span>
          </div>
          <div className="flex items-center gap-3">
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
          <div className="flex-shrink-0 border-b border-error/20 bg-error/[0.08] px-4 py-2 font-mono text-xs text-error">
            {errorMessage}
          </div>
        )}

        <main id="main-content" className="flex min-h-0 flex-1 flex-col">
          <MessageList
            messages={messages}
            onApprove={handleApprove}
            onReject={handleReject}
            onFeedback={handleFeedback}
            onMemoryApprove={handleMemoryApprove}
            onMemoryDeny={handleMemoryDeny}
          />
        </main>

        <ChatInput
          value={inputValue}
          onChange={setInputValue}
          onSend={() => sendMessage(inputValue, selectedModel || undefined)}
          onCancel={cancelStream}
          streamStatus={streamStatus}
          disabled={inputDisabled}
        />
      </div>
    </>
  );
}
