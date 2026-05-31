import { useEffect, useState } from 'react';
import type { StreamStatus } from '@/types';

interface StatusBarProps {
  streamStatus: StreamStatus;
  currentTool: string | null;
}

export function StatusBar({ streamStatus, currentTool }: StatusBarProps) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (streamStatus === 'streaming') {
      setVisible(true);
    } else if (streamStatus === 'done') {
      setVisible(true);
      const t = setTimeout(() => setVisible(false), 2000);
      return () => clearTimeout(t);
    } else {
      setVisible(false);
    }
  }, [streamStatus]);

  if (!visible) return null;

  const label =
    streamStatus === 'streaming'
      ? currentTool
        ? `Using ${currentTool}`
        : 'Thinking'
      : 'Done';

  const isPulsing = streamStatus === 'streaming';

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        padding: '6px 16px',
        fontFamily: 'var(--mono)',
        fontSize: 'var(--text-xs)',
        color: 'var(--text-muted)',
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: 'var(--accent)',
          display: 'inline-block',
          animation: isPulsing ? 'pulse 1.5s ease-in-out infinite' : 'none',
        }}
      />
      {label}
    </div>
  );
}
