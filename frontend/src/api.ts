import { parseSSEStream } from '@/lib/sseParser';
import {
  isAgentEvent,
  isConversationIdPayload,
  type AgentEvent,
  type CancelResponse,
  type ChatRequest,
  type ConversationIdPayload,
  type HealthResponse,
} from '@/types';

const BASE = import.meta.env.VITE_API_URL ?? '/api';

// Streams chat events from POST /chat/stream
// Yields: ConversationIdPayload first, then AgentEvent objects
export async function* streamChat(
  payload: ChatRequest,
  signal: AbortSignal,
): AsyncGenerator<ConversationIdPayload | AgentEvent> {
  const response = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Stream request failed: ${response.status} ${response.statusText}`);
  }

  if (!response.body) {
    throw new Error('Response body is null');
  }

  for await (const raw of parseSSEStream(response.body)) {
    if (isConversationIdPayload(raw) || isAgentEvent(raw)) {
      yield raw;
    }
    // Unknown payloads are silently skipped
  }
}

export async function cancelChat(conversationId: string): Promise<CancelResponse> {
  const response = await fetch(`${BASE}/cancel/${conversationId}`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(`Cancel request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<CancelResponse>;
}

export async function healthCheck(): Promise<HealthResponse> {
  const response = await fetch(`${BASE}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<HealthResponse>;
}
