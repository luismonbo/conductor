// Raw event from the backend stream
export type AgentEventType = 'thinking' | 'tool_call' | 'tool_result' | 'done' | 'error';

export interface AgentEvent {
  readonly type: AgentEventType;
  readonly text: string;
  readonly name: string;
  readonly args: Record<string, unknown>;
  readonly call_id: string;
  readonly is_error: boolean;
  readonly stopped_reason: string;
}

// First SSE payload (not an AgentEvent)
export interface ConversationIdPayload {
  readonly conversation_id: string;
}

// Type guard to distinguish first payload from agent events
export function isConversationIdPayload(v: unknown): v is ConversationIdPayload {
  return (
    typeof v === 'object' &&
    v !== null &&
    'conversation_id' in v &&
    typeof (v as ConversationIdPayload).conversation_id === 'string' &&
    !('type' in v)
  );
}

export function isAgentEvent(v: unknown): v is AgentEvent {
  return (
    typeof v === 'object' &&
    v !== null &&
    'type' in v &&
    typeof (v as AgentEvent).type === 'string'
  );
}

// Message blocks rendered in the UI
export interface ThinkingBlock {
  readonly kind: 'thinking';
  text: string; // mutable — streams in progressively
}

export interface ToolCallBlock {
  readonly kind: 'tool_call';
  readonly name: string;
  readonly args: Record<string, unknown>;
  readonly call_id: string;
}

export interface ToolResultBlock {
  readonly kind: 'tool_result';
  readonly text: string;
  readonly name: string;
  readonly call_id: string;
  readonly is_error: boolean;
}

export type MessageBlock = ThinkingBlock | ToolCallBlock | ToolResultBlock;

// Conversation messages
export interface UserMessage {
  readonly id: string;
  readonly role: 'user';
  readonly text: string;
}

export interface AssistantMessage {
  readonly id: string;
  readonly role: 'assistant';
  blocks: MessageBlock[]; // grows as stream arrives
  finalText?: string;     // from done event if different from last thinking
  isStreaming: boolean;
}

export type ConversationMessage = UserMessage | AssistantMessage;

// Stream status
export type StreamStatus = 'idle' | 'streaming' | 'done' | 'error';

// API request shape
export interface ChatRequest {
  readonly message: string;
  readonly conversation_id?: string;
}

// Cancel response
export interface CancelResponse {
  readonly status: 'cancelled' | 'not_found';
  readonly conversation_id: string;
}

// Health response
export interface HealthResponse {
  readonly status: string;
  readonly backend: string;
}
