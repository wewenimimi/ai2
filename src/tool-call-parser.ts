// Some Workers AI models (Hermes-2-Pro and other Mistral-template models) emit
// tool calls inside `<tool_call>...</tool_call>` XML tags as raw content tokens
// when streaming, instead of populating the structured `tool_calls` field.
// This parser is a streaming state machine that:
//   - emits clean text for everything outside the tags
//   - parses JSON inside the tags into synthetic tool-call records
// It handles tags split across chunks and lenient JSON (single-quoted dicts
// emitted by some Hermes builds).

export interface ParsedToolCall {
  name: string;
  arguments: string; // already JSON-stringified (single-encoded)
}

export interface ToolCallParserCallbacks {
  onText: (chunk: string) => void;
  onToolCall: (call: ParsedToolCall) => void;
}

const OPEN_TAG = "<tool_call>";
const CLOSE_TAG = "</tool_call>";

export function createToolCallStreamParser(cb: ToolCallParserCallbacks) {
  let mode: "text" | "in_tag" = "text";
  let buffer = "";

  const flushText = (s: string) => {
    if (s) cb.onText(s);
  };

  const tryParseJson = (raw: string): unknown | null => {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    try {
      return JSON.parse(trimmed);
    } catch {}
    // Hermes occasionally emits Python-style dicts. Naive single→double-quote
    // swap is enough for the structures the chat template produces.
    try {
      return JSON.parse(trimmed.replace(/'/g, '"'));
    } catch {}
    return null;
  };

  const parseToolCall = (jsonRaw: string) => {
    const parsed = tryParseJson(jsonRaw) as { name?: string; arguments?: unknown } | null;
    if (!parsed || typeof parsed.name !== "string") return null;
    const args =
      typeof parsed.arguments === "string"
        ? parsed.arguments
        : JSON.stringify(parsed.arguments ?? {});
    cb.onToolCall({ name: parsed.name, arguments: args });
    return true;
  };

  function feed(chunk: string): void {
    if (!chunk) return;
    buffer += chunk;

    // Drain the buffer in a loop until no more transitions are possible.
    while (buffer.length > 0) {
      if (mode === "text") {
        const openIdx = buffer.indexOf(OPEN_TAG);
        if (openIdx === -1) {
          // No full open tag in buffer. Emit everything up to the last "<" so
          // we hold back a possible partial tag for the next chunk.
          const ltIdx = buffer.lastIndexOf("<");
          if (ltIdx === -1) {
            flushText(buffer);
            buffer = "";
          } else {
            flushText(buffer.slice(0, ltIdx));
            buffer = buffer.slice(ltIdx);
            // If what's left can't possibly become "<tool_call>", emit it.
            if (!OPEN_TAG.startsWith(buffer)) {
              flushText(buffer);
              buffer = "";
            }
          }
          return;
        }
        // Found a full open tag.
        flushText(buffer.slice(0, openIdx));
        buffer = buffer.slice(openIdx + OPEN_TAG.length);
        mode = "in_tag";
        // Loop again to handle the tag body in this same call.
      } else {
        const closeIdx = buffer.indexOf(CLOSE_TAG);
        if (closeIdx === -1) {
          // Hold the whole buffer until close arrives.
          return;
        }
        const jsonRaw = buffer.slice(0, closeIdx);
        parseToolCall(jsonRaw);
        buffer = buffer.slice(closeIdx + CLOSE_TAG.length);
        mode = "text";
        // Continue draining — there may be more text or another tool call.
      }
    }
  }

  function end(): void {
    if (mode === "in_tag") {
      // Stream ended inside a tag. Best effort: try to parse what we have.
      const jsonRaw = buffer;
      if (!parseToolCall(jsonRaw)) {
        // Couldn't parse — fall back to surfacing it as text so the user
        // at least sees what the model produced.
        flushText(OPEN_TAG + buffer);
      }
    } else if (buffer) {
      flushText(buffer);
    }
    buffer = "";
  }

  return { feed, end };
}
