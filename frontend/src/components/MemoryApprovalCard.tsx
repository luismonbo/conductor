import { useState } from 'react';

interface MemoryApprovalCardProps {
  proposed: string;
  onApprove: () => void;
  onDeny: () => void;
  onFeedback: (text: string) => void;
}

export function MemoryApprovalCard({ proposed, onApprove, onDeny, onFeedback }: MemoryApprovalCardProps) {
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
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: '8px',
        padding: '12px 16px',
        marginTop: '8px',
        background: 'var(--surface)',
      }}
    >
      <div
        style={{
          fontFamily: 'var(--mono)',
          fontSize: 'var(--text-xs)',
          color: 'var(--text-muted)',
          marginBottom: '10px',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
        }}
      >
        Save to memory?
      </div>

      <div
        style={{
          fontFamily: 'var(--sans)',
          fontSize: 'var(--text-sm)',
          color: 'var(--text)',
          background: 'var(--bg)',
          padding: '10px 12px',
          borderRadius: '6px',
          borderLeft: '3px solid var(--accent)',
          lineHeight: 1.5,
          marginBottom: '12px',
        }}
      >
        {proposed}
      </div>

      <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
        <button
          type="button"
          onClick={handleDeny}
          disabled={decided}
          style={{
            padding: '6px 14px',
            background: 'transparent',
            border: '1px solid var(--color-error)',
            borderRadius: '6px',
            color: 'var(--color-error)',
            fontFamily: 'var(--mono)',
            fontSize: 'var(--text-xs)',
            cursor: decided ? 'not-allowed' : 'pointer',
            opacity: decided ? 0.4 : 1,
            transition: 'background 0.15s',
          }}
          onMouseEnter={(e) => { if (!decided) e.currentTarget.style.background = 'rgba(248,113,113,0.1)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
        >
          Deny
        </button>
        <button
          type="button"
          onClick={handleApprove}
          disabled={decided}
          style={{
            padding: '6px 14px',
            background: 'transparent',
            border: '1px solid var(--accent)',
            borderRadius: '6px',
            color: 'var(--accent)',
            fontFamily: 'var(--mono)',
            fontSize: 'var(--text-xs)',
            cursor: decided ? 'not-allowed' : 'pointer',
            opacity: decided ? 0.4 : 1,
            transition: 'background 0.15s',
          }}
          onMouseEnter={(e) => { if (!decided) e.currentTarget.style.opacity = '0.75'; }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = decided ? '0.4' : '1'; }}
        >
          Approve
        </button>
      </div>

      <div style={{ marginTop: '10px', display: 'flex', gap: '6px' }}>
        <input
          type="text"
          value={feedbackText}
          onChange={(e) => setFeedbackText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleFeedback(); }}
          disabled={decided}
          placeholder="Or type feedback to refine…"
          style={{
            flex: 1,
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '6px 10px',
            color: 'var(--text)',
            fontFamily: 'var(--sans)',
            fontSize: 'var(--text-xs)',
            outline: 'none',
            opacity: decided ? 0.4 : 1,
          }}
        />
        <button
          type="button"
          onClick={handleFeedback}
          disabled={decided || !feedbackText.trim()}
          style={{
            padding: '6px 12px',
            background: 'transparent',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--mono)',
            fontSize: 'var(--text-xs)',
            cursor: decided || !feedbackText.trim() ? 'not-allowed' : 'pointer',
            opacity: decided || !feedbackText.trim() ? 0.4 : 1,
            whiteSpace: 'nowrap',
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
