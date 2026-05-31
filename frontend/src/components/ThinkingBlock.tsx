import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { useMemo } from 'react';

interface ThinkingBlockProps {
  text: string;
}

export function ThinkingBlock({ text }: ThinkingBlockProps) {
  if (!text) return null;

  const html = useMemo(
    () => DOMPurify.sanitize(marked.parse(text, { async: false }) as string),
    [text]
  );

  return (
    <div
      className="thinking-prose"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
