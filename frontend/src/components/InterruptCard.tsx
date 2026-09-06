import { useState, type ReactNode } from 'react';

interface InterruptCardProps {
  label: string;
  children: ReactNode;
  onApprove: () => void;
  onDeny: () => void;
  onFeedback: (text: string) => void;
}

export function InterruptCard({ label, children, onApprove, onDeny, onFeedback }: InterruptCardProps) {
  const [decided, setDecided] = useState(false);
  const [feedbackText, setFeedbackText] = useState('');

  const handleApprove = () => {
    if (decided) return;
    setDecided(true);
    onApprove();
  };

  const handleDeny = () => {
    if (decided) return;
    setDecided(true);
    onDeny();
  };

  const handleFeedback = () => {
    if (decided || !feedbackText.trim()) return;
    setDecided(true);
    onFeedback(feedbackText.trim());
  };

  return (
    <div className="mt-2 rounded-md border border-border bg-surface p-4">
      <div className="mb-2.5 font-mono text-xs uppercase tracking-wider text-fg-muted">
        {label}
      </div>

      {children}

      <div className="mt-3 flex justify-end gap-2">
        <button
          type="button"
          onClick={handleDeny}
          disabled={decided}
          className="rounded-md border border-error px-3.5 py-1.5 font-mono text-xs text-error transition enabled:hover:bg-error/10 disabled:cursor-not-allowed disabled:opacity-40 active:enabled:scale-[0.98]"
        >
          Deny
        </button>
        <button
          type="button"
          onClick={handleApprove}
          disabled={decided}
          className="rounded-md border border-accent px-3.5 py-1.5 font-mono text-xs text-accent transition enabled:hover:opacity-75 disabled:cursor-not-allowed disabled:opacity-40 active:enabled:scale-[0.98]"
        >
          Approve
        </button>
      </div>

      <div className="mt-2.5 flex gap-1.5">
        <input
          type="text"
          value={feedbackText}
          onChange={(e) => setFeedbackText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleFeedback(); }}
          disabled={decided}
          placeholder="Or type feedback to refine…"
          className="flex-1 rounded-md border border-border bg-canvas px-2.5 py-1.5 font-sans text-xs text-fg outline-none placeholder:text-fg-muted disabled:opacity-40"
        />
        <button
          type="button"
          onClick={handleFeedback}
          disabled={decided || !feedbackText.trim()}
          className="whitespace-nowrap rounded-md border border-border px-3 py-1.5 font-mono text-xs text-fg-muted disabled:cursor-not-allowed disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}
