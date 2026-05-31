export async function* parseSSEStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on double-newline (SSE message boundary)
      const messages = buffer.split('\n\n');
      // Last element is incomplete — keep it in the buffer
      buffer = messages.pop() ?? '';

      for (const message of messages) {
        if (!message.trim()) continue;

        const lines = message.split('\n');
        const dataLines: string[] = [];

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            dataLines.push(line.slice(6));
          } else if (line.startsWith('event:') || line.startsWith('id:') || line.startsWith('retry:')) {
            // SSE metadata fields — ignore
          }
          // Empty lines within a block are separators — ignore
        }

        if (dataLines.length === 0) continue;

        const dataStr = dataLines.join('\n');
        if (dataStr === '[DONE]') continue;

        try {
          yield JSON.parse(dataStr);
        } catch {
          // Malformed JSON — skip (dev: check network tab if events are missing)
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
