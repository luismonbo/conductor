interface UserMessageProps {
  text: string;
}

export function UserMessage({ text }: UserMessageProps) {
  return (
    <div className="flex justify-end px-4 py-1">
      <div className="max-w-[70%] whitespace-pre-wrap break-words rounded-tl-lg rounded-tr-lg rounded-bl-lg rounded-br-[2px] border border-border bg-surface px-3.5 py-2.5 font-mono text-sm leading-relaxed text-fg">
        {text}
      </div>
    </div>
  );
}
