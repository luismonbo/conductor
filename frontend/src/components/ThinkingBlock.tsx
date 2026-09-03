import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { useMemo } from 'react';

interface ThinkingBlockProps {
  text: string;
}

export function ThinkingBlock({ text }: ThinkingBlockProps) {
  const html = useMemo(
    () => (text ? DOMPurify.sanitize(marked.parse(text, { async: false }) as string) : ''),
    [text]
  );

  if (!text) return null;

  return (
    <div
      className="thinking-prose"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
