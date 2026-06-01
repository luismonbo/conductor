// Raw event from the backend stream
export type AgentEventType =
  | 'thinking'
  | 'tool_call'
  | 'tool_result'
  | 'interrupt'
  | 'final'
  | 'error';

export interface AgentEvent {
  readonly type: AgentEventType;
  readonly text: string;
  readonly name: string;
  readonly args: Record<string, unknown>;
  readonly call_id: string;
  readonly is_error: boolean;
  readonly stopped_reason: string;
}

// Interrupt payload carried inside AgentEvent.args when type === 'interrupt'
export interface InterruptPayload {
  readonly mode: 'approval';
  readonly tool_calls: ReadonlyArray<{
    name: string;
    args: Record<string, unknown>;
    call_id: string;
  }>;
}

// First SSE payload (not an AgentEvent)
export interface ThreadIdPayload {
  readonly thread_id: string;
}

// Type guards
export function isThreadIdPayload(v: unknown): v is ThreadIdPayload {
  return (
    typeof v === 'object' &&
    v !== null &&
    'thread_id' in v &&
    typeof (v as ThreadIdPayload).thread_id === 'string' &&
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
  text: string;
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
  blocks: MessageBlock[];
  finalText?: string;
  isStreaming: boolean;
  interruptPayload?: InterruptPayload;
}

export type ConversationMessage = UserMessage | AssistantMessage;

// Stream status
export type StreamStatus = 'idle' | 'streaming' | 'interrupted' | 'done' | 'error';

// API request shapes
export interface ChatRequest {
  readonly message: string;
  readonly thread_id?: string;
}

export interface ResumeRequest {
  readonly decision: Record<string, unknown>;
}

// Cancel response
export interface CancelResponse {
  readonly status: 'cancelled' | 'not_found';
  readonly thread_id: string;
}

// Health response
export interface HealthResponse {
  readonly status: string;
  readonly backend: string;
}
