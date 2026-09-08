import type { ThreadSummary } from '@/types';

interface ThreadSidebarProps {
  threads: ThreadSummary[];
  activeThreadId: string | null;
  onSelect: (threadId: string) => void;
  onNew: () => void;
  open?: boolean;
  onClose?: () => void;
}

function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function ThreadSidebar({ threads, activeThreadId, onSelect, onNew, open = false, onClose }: ThreadSidebarProps) {
  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/60 md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <nav
        aria-label="conversations"
        className={`fixed inset-y-0 left-0 z-40 flex w-60 flex-shrink-0 flex-col overflow-y-auto border-r border-border bg-canvas transition-transform duration-200 md:relative md:z-auto md:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <button
          type="button"
          onClick={onNew}
          className="m-3 rounded-sm border border-border px-3 py-2 text-left font-mono text-xs uppercase tracking-wide text-fg-muted transition hover:bg-[rgba(127,127,127,0.08)] active:scale-[0.98]"
        >
          + new chat
        </button>

        {threads.map((t) => {
          const isActive = t.thread_id === activeThreadId;
          return (
            <button
              key={t.thread_id}
              type="button"
              aria-current={isActive ? 'true' : undefined}
              onClick={() => onSelect(t.thread_id)}
              className={`flex flex-col gap-0.5 border-l-2 px-4 py-2.5 text-left transition active:scale-[0.98] ${
                isActive
                  ? 'border-accent bg-[rgba(127,127,127,0.1)]'
                  : 'border-transparent hover:bg-[rgba(127,127,127,0.06)]'
              }`}
            >
              <span className="max-w-full truncate text-sm text-fg">
                {t.title || t.thread_id.slice(0, 8)}
              </span>
              <span className="font-mono text-xs text-fg-muted">
                {formatWhen(t.last_at)}
              </span>
            </button>
          );
        })}
      </nav>
    </>
  );
}
