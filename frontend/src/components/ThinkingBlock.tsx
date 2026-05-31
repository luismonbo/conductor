import { marked } from 'marked';
import { useMemo } from 'react';

interface ThinkingBlockProps {
  text: string;
}

export function ThinkingBlock({ text }: ThinkingBlockProps) {
  const html = useMemo(() => marked.parse(text, { async: false }) as string, [text]);

  return (
    <div
      className="thinking-prose"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
