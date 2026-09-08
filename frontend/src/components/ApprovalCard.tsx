import type { ToolApprovalPayload } from '@/types';
import { InterruptCard } from '@/components/InterruptCard';

interface ApprovalCardProps {
  payload: ToolApprovalPayload;
  onApprove: () => void;
  onReject: () => void;
  onFeedback: (text: string) => void;
}

export function ApprovalCard({ payload, onApprove, onReject, onFeedback }: ApprovalCardProps) {
  return (
    <InterruptCard label="Tool approval required" onApprove={onApprove} onDeny={onReject} onFeedback={onFeedback}>
      {payload.tool_calls.map((tc) => (
        <div key={tc.call_id} className="mb-2.5">
          <div className="mb-1 font-mono text-sm text-accent">{tc.name}</div>
          <pre className="m-0 overflow-x-auto whitespace-pre-wrap break-all rounded-sm bg-canvas p-2 font-mono text-xs text-fg-muted">
            {JSON.stringify(tc.args, null, 2)}
          </pre>
        </div>
      ))}
    </InterruptCard>
  );
}
