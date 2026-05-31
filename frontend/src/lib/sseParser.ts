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
        const line = message.trim();
        if (!line) continue;

        const dataLine = line.startsWith('data: ') ? line.slice(6) : line;
        if (!dataLine || dataLine === '[DONE]') continue;

        try {
          yield JSON.parse(dataLine);
        } catch {
          // Malformed JSON — skip silently
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
