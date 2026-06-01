import { useState } from 'react';
import type { InterruptPayload } from '@/types';

interface ApprovalCardProps {
  payload: InterruptPayload;
  onApprove: () => void;
  onReject: () => void;
}

export function ApprovalCard({ payload, onApprove, onReject }: ApprovalCardProps) {
  const [decided, setDecided] = useState(false);

  const handleApprove = () => {
    if (decided) return;
    setDecided(true);
    onApprove();
  };

  const handleReject = () => {
    if (decided) return;
    setDecided(true);
    onReject();
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
        Tool approval required
      </div>

      {payload.tool_calls.map((tc) => (
        <div key={tc.call_id} style={{ marginBottom: '10px' }}>
          <div
            style={{
              fontFamily: 'var(--mono)',
              fontSize: 'var(--text-sm)',
              color: 'var(--accent)',
              marginBottom: '4px',
            }}
          >
            {tc.name}
          </div>
          <pre
            style={{
              margin: 0,
              fontFamily: 'var(--mono)',
              fontSize: 'var(--text-xs)',
              color: 'var(--text-muted)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
              background: 'var(--bg)',
              padding: '6px 8px',
              borderRadius: '4px',
            }}
          >
            {JSON.stringify(tc.args, null, 2)}
          </pre>
        </div>
      ))}

      <div
        style={{
          display: 'flex',
          gap: '8px',
          marginTop: '12px',
          justifyContent: 'flex-end',
        }}
      >
        <button
          type="button"
          onClick={handleReject}
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
          onMouseEnter={(e) => {
            if (!decided) e.currentTarget.style.background = 'rgba(248,113,113,0.1)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
          }}
        >
          Reject
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
          onMouseEnter={(e) => {
            if (!decided) e.currentTarget.style.opacity = '0.75';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.opacity = decided ? '0.4' : '1';
          }}
        >
          Approve
        </button>
      </div>
    </div>
  );
}
