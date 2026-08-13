import type { ThreadSummary } from '@/types';

interface ThreadSidebarProps {
  threads: ThreadSummary[];
  activeThreadId: string | null;
  onSelect: (threadId: string) => void;
  onNew: () => void;
}

function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function ThreadSidebar({ threads, activeThreadId, onSelect, onNew }: ThreadSidebarProps) {
  return (
    <nav
      aria-label="conversations"
      style={{
        width: 240,
        flexShrink: 0,
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        overflowY: 'auto',
      }}
    >
      <button
        type="button"
        onClick={onNew}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(127, 127, 127, 0.08)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
        style={{
          margin: '12px',
          padding: '8px 12px',
          fontFamily: 'var(--mono)',
          fontSize: 'var(--text-xs)',
          color: 'var(--text-muted)',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          background: 'transparent',
          border: '1px solid var(--border)',
          borderRadius: '4px',
          cursor: 'pointer',
          textAlign: 'left',
        }}
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
            onMouseEnter={(e) => {
              if (!isActive) e.currentTarget.style.background = 'rgba(127, 127, 127, 0.06)';
            }}
            onMouseLeave={(e) => {
              if (!isActive) e.currentTarget.style.background = 'transparent';
            }}
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
              padding: '10px 16px',
              background: isActive ? 'rgba(127, 127, 127, 0.1)' : 'transparent',
              border: 'none',
              borderLeft: isActive
                ? '2px solid var(--accent)'
                : '2px solid transparent',
              cursor: 'pointer',
              textAlign: 'left',
            }}
          >
            <span
              style={{
                fontSize: 'var(--text-sm)',
                color: 'var(--text)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                maxWidth: '100%',
              }}
            >
              {t.title || t.thread_id.slice(0, 8)}
            </span>
            <span
              style={{
                fontFamily: 'var(--mono)',
                fontSize: 'var(--text-xs)',
                color: 'var(--text-muted)',
              }}
            >
              {formatWhen(t.last_at)}
            </span>
          </button>
        );
      })}
    </nav>
  );
}
