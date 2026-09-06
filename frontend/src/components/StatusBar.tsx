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
    } else if (streamStatus === 'interrupted') {
      setVisible(true);
    } else {
      setVisible(false);
    }
  }, [streamStatus]);

  const label =
    streamStatus === 'interrupted'
      ? 'Waiting for approval'
      : streamStatus === 'streaming'
      ? currentTool
        ? `Using ${currentTool}`
        : 'Thinking'
      : 'Done';

  const isPulsing = streamStatus === 'streaming';
  const dotClass = streamStatus === 'interrupted' ? 'bg-warning' : 'bg-accent';

  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
      className="flex min-h-[28px] items-center gap-1.5 px-4 py-1.5 font-mono text-xs text-fg-muted"
    >
      {visible && (
        <>
          <span
            className={`inline-block h-1.5 w-1.5 rounded-full ${dotClass} ${
              isPulsing ? 'animate-[pulse_1.5s_ease-in-out_infinite]' : ''
            }`}
          />
          {label}
        </>
      )}
    </div>
  );
}
