import { InterruptCard } from '@/components/InterruptCard';

interface MemoryApprovalCardProps {
  proposed: string;
  onApprove: () => void;
  onDeny: () => void;
  onFeedback: (text: string) => void;
}

export function MemoryApprovalCard({ proposed, onApprove, onDeny, onFeedback }: MemoryApprovalCardProps) {
  return (
    <InterruptCard label="Save to memory?" onApprove={onApprove} onDeny={onDeny} onFeedback={onFeedback}>
      <div className="mb-3 rounded-sm border-l-[3px] border-accent bg-canvas px-3 py-2.5 font-sans text-sm leading-relaxed text-fg">
        {proposed}
      </div>
    </InterruptCard>
  );
}
