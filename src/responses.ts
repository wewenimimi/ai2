import type { Context } from "hono";
import { runAI } from "./ai-client";
import { inlineImageUrls } from "./image-inline";
import { isVisionModel, resolveChatModel, VISION_DEFAULT_MODEL } from "./mapping";
import { createToolCallStreamParser } from "./tool-call-parser";
import type { Env } from "./types";

// OpenAI Responses API (`POST /v1/responses`) — released 2025. Some clients
// (n8n, LangChain.js, the new OpenAI SDK helpers) prefer it over the legacy
// /v1/chat/completions. We translate it into the same Workers AI chat call.

interface ResponsesContentPart {
  type: string;
  text?: string;
  image_url?: string | { url: string };
}

interface ResponsesInputItem {
  role?: "system" | "user" | "assistant" | "developer";
  content?: string | ResponsesContentPart[];
  type?: string;
}

interface ResponsesRequest {
  model: string;
  input: string | ResponsesInputItem[];
  instructions?: string;
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  max_output_tokens?: number;
  max_tokens?: number;
  seed?: number;
  tools?: unknown[];
  tool_choice?: unknown;
  response_format?: { type: string; json_schema?: unknown };
  text?: { format?: { type: string; json_schema?: unknown } };
  user?: string;
  previous_response_id?: string;
  store?: boolean;
}

// Adapt Responses-API content into the chat.completions content shape that
// Workers AI expects. Plain text collapses to a string; if any image parts
// are present we keep the multipart array and convert input_image →
// image_url so vision models receive it intact.
function adaptResponsesContent(content: ResponsesInputItem["content"]): string | unknown[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  const hasImage = content.some(
    (p) => p && (p.type === "input_image" || p.type === "image_url" || (p as any).image_url),
  );
  if (!hasImage) {
    return content
      .map((p) => {
        if ((p.type === "input_text" || p.type === "output_text" || p.type === "text") && p.text) return p.text;
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }

  // Multipart with images — emit chat.completions parts.
  const out: unknown[] = [];
  for (const p of content) {
    if (!p) continue;
    if ((p.type === "input_text" || p.type === "output_text" || p.type === "text") && p.text) {
      out.push({ type: "text", text: p.text });
      continue;
    }
    if (p.type === "input_image" || p.type === "image_url" || (p as any).image_url) {
      const raw = (p as any).image_url;
      const url = typeof raw === "string" ? raw : raw?.url;
      if (url) out.push({ type: "image_url", image_url: { url } });
      continue;
    }
  }
  return out;
}

function inputContainsImage(input: ResponsesRequest["input"]): boolean {
  if (typeof input === "string" || !Array.isArray(input)) return false;
  for (const item of input as any[]) {
    if (!item || typeof item !== "object") continue;
    const c = item.content;
    if (Array.isArray(c) && c.some((p: any) => p && (p.type === "input_image" || p.type === "image_url" || p.image_url))) {
      return true;
    }
  }
  return false;
}

// Responses API allows several item types in the `input` array:
//   - { type: "message", role, content }
//   - { type: "function_call", id, call_id, name, arguments }
//   - { type: "function_call_output", call_id, output }
// In multi-turn agent loops (n8n, LangChain), the second and third types
// carry the tool call/result history that the model needs to keep going.
// Convert them to chat.completions equivalents:
//   function_call          → { role:"assistant", tool_calls:[{ id, type:"function", function:{name, arguments} }] }
//   function_call_output   → { role:"tool", tool_call_id, content }
type ChatTurn = {
  role: string;
  content?: string | unknown[];
  tool_calls?: unknown[];
  tool_call_id?: string;
};

function adaptInputToMessages(req: ResponsesRequest): ChatTurn[] {
  const messages: ChatTurn[] = [];
  if (req.instructions) messages.push({ role: "system", content: req.instructions });

  if (typeof req.input === "string") {
    messages.push({ role: "user", content: req.input });
    return messages;
  }
  if (!Array.isArray(req.input)) return messages;

  for (const item of req.input as any[]) {
    if (!item || typeof item !== "object") continue;

    if (item.type === "function_call") {
      const args = typeof item.arguments === "string"
        ? item.arguments
        : JSON.stringify(item.arguments ?? {});
      const callId = item.call_id ?? item.id ?? `call_${messages.length}`;
      messages.push({
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: callId,
            type: "function",
            function: { name: item.name, arguments: args },
          },
        ],
      });
      continue;
    }

    if (item.type === "function_call_output") {
      const output = typeof item.output === "string" ? item.output : JSON.stringify(item.output ?? "");
      messages.push({
        role: "tool",
        tool_call_id: item.call_id ?? item.id ?? "",
        content: output,
      });
      continue;
    }

    // Default: treat as a message item (with or without explicit type:"message").
    const role = item.role === "developer" ? "system" : (item.role ?? "user");
    const content = adaptResponsesContent(item.content);
    if ((typeof content === "string" && content) || (Array.isArray(content) && content.length > 0) || item.role === "assistant") {
      messages.push({ role, content: content as any });
    }
  }
  return messages;
}

// The Responses API and chat.completions disagree on the tool schema:
//   Responses API:    { type:"function", name, description, parameters, strict? }
//   chat.completions: { type:"function", function:{ name, description, parameters, strict? } }
// Workers AI chat models accept the chat.completions shape, so we normalize.
function adaptResponsesTools(tools: unknown[]): unknown[] {
  return tools.map((t: any) => {
    if (!t || typeof t !== "object") return t;
    if (t.type === "function" && t.function && typeof t.function === "object") return t; // already chat shape
    if (t.type === "function" && (t.name || t.parameters)) {
      return {
        type: "function",
        function: {
          name: t.name,
          description: t.description,
          parameters: t.parameters,
          ...(typeof t.strict === "boolean" ? { strict: t.strict } : {}),
        },
      };
    }
    return t;
  });
}

// Inverse: chat.completions tool_calls → Responses API output items.
//   chat: [{ id, type:"function", function:{ name, arguments } }]
//   resp: [{ type:"function_call", id, call_id, name, arguments }]
//
// `arguments` must be a single-encoded JSON string. Some Workers AI models
// (Granite in particular) occasionally double-encode it: the value parses to
// another JSON string instead of an object. We unwrap that to keep n8n /
// LangChain happy, since they expect JSON.parse(arguments) to yield the args.
function normalizeArguments(raw: unknown): string {
  if (raw == null) return "{}";
  if (typeof raw !== "string") {
    try { return JSON.stringify(raw); } catch { return "{}"; }
  }
  let parsed: unknown;
  try { parsed = JSON.parse(raw); } catch { return raw; }
  if (typeof parsed === "string") {
    try {
      const inner = JSON.parse(parsed);
      return JSON.stringify(inner);
    } catch {
      return raw;
    }
  }
  return raw;
}

function toolCallsToResponsesItems(toolCalls: any[]): unknown[] {
  return toolCalls.map((tc, idx) => ({
    type: "function_call",
    id: tc.id ?? `fc_${idx}`,
    call_id: tc.id ?? `call_${idx}`,
    name: tc.function?.name ?? tc.name,
    arguments: normalizeArguments(tc.function?.arguments ?? tc.arguments ?? "{}"),
    status: "completed",
  }));
}

function generateResponseId(): string {
  return "resp_" + crypto.randomUUID().replace(/-/g, "").slice(0, 24);
}

function generateMessageId(): string {
  return "msg_" + crypto.randomUUID().replace(/-/g, "").slice(0, 24);
}

export async function handleResponses(c: Context<{ Bindings: Env }>) {
  let body: ResponsesRequest;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid JSON body", type: "invalid_request_error" } }, 400);
  }

  if (!body.input) {
    return c.json({ error: { message: "`input` is required", type: "invalid_request_error" } }, 400);
  }

  let model = resolveChatModel(body.model, c.env.DEFAULT_CHAT_MODEL ?? "@cf/meta/llama-3.3-70b-instruct-fp8-fast");
  // Auto-route to a vision model when input items include image parts and
  // the resolved model can't handle them. Caller's body.model is preserved
  // in the response so OpenAI clients still see what they sent.
  const hasImage = inputContainsImage(body.input);
  if (hasImage && !isVisionModel(model)) {
    model = VISION_DEFAULT_MODEL;
  }
  const messages = adaptInputToMessages(body);
  if (hasImage) await inlineImageUrls(messages);
  const stream = body.stream === true;

  const aiInput: Record<string, unknown> = { messages, stream };
  if (typeof body.temperature === "number") aiInput.temperature = body.temperature;
  if (typeof body.top_p === "number") aiInput.top_p = body.top_p;
  if (typeof (body as any).top_k === "number") aiInput.top_k = (body as any).top_k;
  const maxOut = body.max_output_tokens ?? body.max_tokens;
  if (typeof maxOut === "number") aiInput.max_tokens = maxOut;
  if (typeof body.seed === "number") aiInput.seed = body.seed;
  if (typeof (body as any).frequency_penalty === "number") aiInput.frequency_penalty = (body as any).frequency_penalty;
  if (typeof (body as any).presence_penalty === "number") aiInput.presence_penalty = (body as any).presence_penalty;
  if (typeof (body as any).repetition_penalty === "number") aiInput.repetition_penalty = (body as any).repetition_penalty;
  if ((body as any).stop !== undefined) aiInput.stop = (body as any).stop;
  if (Array.isArray(body.tools) && body.tools.length > 0) aiInput.tools = adaptResponsesTools(body.tools);
  if (body.tool_choice !== undefined) aiInput.tool_choice = body.tool_choice;
  const responseFormat = body.response_format ?? body.text?.format;
  if (responseFormat) aiInput.response_format = responseFormat;

  const responseId = generateResponseId();
  const messageId = generateMessageId();
  const created = Math.floor(Date.now() / 1000);
  const modelLabel = body.model || model;

  if (!stream) {
    let result: any;
    try {
      result = await runAI(c.env, model, aiInput);
    } catch (err) {
      return c.json(
        { error: { message: (err as Error).message ?? "Workers AI call failed", type: "upstream_error" } },
        502,
      );
    }

    // Same dual-shape handling as chat.ts: legacy `response` field vs.
    // OpenAI-native `choices[].message.content` (Granite, DeepSeek-R1, ...).
    const nativeChoice = Array.isArray(result?.choices) ? result.choices[0] : null;
    const rawText: string = nativeChoice?.message?.content
      ?? (typeof result?.response === "string" ? result.response : null)
      ?? (typeof result?.result?.response === "string" ? result.result.response : null)
      ?? "";
    // Two reasoning shapes coexist:
    //   - <think>...</think> in content (DeepSeek-R1, QwQ)
    //   - native message.reasoning field (Gemma 4, o-series)
    const nativeReasoning =
      typeof nativeChoice?.message?.reasoning === "string" && nativeChoice.message.reasoning.trim()
        ? nativeChoice.message.reasoning.trim()
        : null;
    const thinkMatch = nativeReasoning ? null : rawText.match(/^\s*<think>([\s\S]*?)<\/think>\s*([\s\S]*)$/);
    const text = thinkMatch ? thinkMatch[2].trim() : rawText;
    const reasoning = nativeReasoning ?? (thinkMatch ? thinkMatch[1].trim() : null);
    const inputTokens = result?.usage?.prompt_tokens ?? 0;
    const outputTokens = result?.usage?.completion_tokens ?? 0;
    const toolCalls = nativeChoice?.message?.tool_calls?.length
      ? nativeChoice.message.tool_calls
      : (Array.isArray(result?.tool_calls) && result.tool_calls.length ? result.tool_calls : null);

    const output: unknown[] = [];
    // Reasoning items come first in OpenAI's o-series response shape.
    if (reasoning) {
      output.push({
        type: "reasoning",
        id: "rs_" + crypto.randomUUID().replace(/-/g, "").slice(0, 24),
        summary: [],
        content: [{ type: "reasoning_text", text: reasoning }],
      });
    }
    if (text) {
      output.push({
        type: "message",
        id: messageId,
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text, annotations: [] }],
      });
    }
    if (toolCalls) {
      output.push(...toolCallsToResponsesItems(toolCalls));
    }
    if (output.length === 0) {
      // Always include a message item, even if empty, so consumers don't crash.
      output.push({
        type: "message",
        id: messageId,
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text: "", annotations: [] }],
      });
    }

    return c.json({
      id: responseId,
      object: "response",
      created_at: created,
      status: "completed",
      model: modelLabel,
      output,
      output_text: text,
      usage: {
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      },
      parallel_tool_calls: true,
      tool_choice: body.tool_choice ?? "auto",
      tools: body.tools ?? [],
    });
  }

  // Streaming: emit the SSE event types the Responses API consumers (n8n,
  // LangChain.js, OpenAI's responses.stream helper) listen for.
  let upstream: ReadableStream<Uint8Array>;
  try {
    upstream = (await runAI(c.env, model, aiInput, { stream: true })) as ReadableStream<Uint8Array>;
  } catch (err) {
    return c.json(
      { error: { message: (err as Error).message ?? "Workers AI call failed", type: "upstream_error" } },
      502,
    );
  }

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const initialResponse = {
    id: responseId,
    object: "response",
    created_at: created,
    status: "in_progress",
    model: modelLabel,
    output: [],
    parallel_tool_calls: true,
    tool_choice: body.tool_choice ?? "auto",
    tools: body.tools ?? [],
  };

  const out = new ReadableStream<Uint8Array>({
    async start(controller) {
      let sequence = 0;
      const writeEvent = (event: string, data: unknown) => {
        controller.enqueue(
          encoder.encode(`event: ${event}\ndata: ${JSON.stringify({ ...(data as object), sequence_number: sequence++ })}\n\n`),
        );
      };

      writeEvent("response.created", { type: "response.created", response: initialResponse });
      writeEvent("response.in_progress", { type: "response.in_progress", response: initialResponse });

      // Lazily emit the message-item lifecycle so we don't ship an empty
      // message when the model only produces tool calls.
      let messageStarted = false;
      let messageOutputIndex = -1;
      let nextOutputIndex = 0;
      const startMessageItem = () => {
        if (messageStarted) return;
        messageOutputIndex = nextOutputIndex++;
        messageStarted = true;
        writeEvent("response.output_item.added", {
          type: "response.output_item.added",
          output_index: messageOutputIndex,
          item: { type: "message", id: messageId, status: "in_progress", role: "assistant", content: [] },
        });
        writeEvent("response.content_part.added", {
          type: "response.content_part.added",
          item_id: messageId,
          output_index: messageOutputIndex,
          content_index: 0,
          part: { type: "output_text", text: "", annotations: [] },
        });
      };

      // Aggregate tool calls by index across chunks. Each tool call produces
      // its own output item with the function_call.* event lifecycle.
      type ToolCallAcc = {
        outputIndex: number;
        itemId: string;
        callId: string;
        name: string;
        argsBuffer: string;
        added: boolean;
      };
      const toolCalls = new Map<number, ToolCallAcc>();

      const reader = upstream.getReader();
      let buffer = "";
      let fullText = "";
      let upstreamUsage: any = null;

      // Parser for the inline `<tool_call>...</tool_call>` blocks some Mistral/
      // Hermes models stream as raw content. Each parsed tool call becomes a
      // synthetic accumulator entry so the same close-out path handles them
      // alongside structured tool_calls deltas from other models.
      let nextSyntheticIdx = 10000;
      const tagParser = createToolCallStreamParser({
        onText: (chunk: string) => {
          startMessageItem();
          fullText += chunk;
          writeEvent("response.output_text.delta", {
            type: "response.output_text.delta",
            item_id: messageId,
            output_index: messageOutputIndex,
            content_index: 0,
            delta: chunk,
          });
        },
        onToolCall: ({ name, arguments: args }) => {
          const idx = nextSyntheticIdx++;
          const acc = ensureToolCallStarted(idx, undefined, name);
          // Emit the entire arguments string as a single delta so consumers
          // see at least one *.arguments.delta event before *.done.
          if (args) {
            acc.argsBuffer += args;
            writeEvent("response.function_call_arguments.delta", {
              type: "response.function_call_arguments.delta",
              item_id: acc.itemId,
              output_index: acc.outputIndex,
              delta: args,
            });
          }
        },
      });

      const ensureToolCallStarted = (idx: number, id: string | undefined, name: string | undefined): ToolCallAcc => {
        let acc = toolCalls.get(idx);
        if (!acc) {
          acc = {
            outputIndex: nextOutputIndex++,
            itemId: "fc_" + crypto.randomUUID().replace(/-/g, "").slice(0, 24),
            callId: id ?? `call_${idx}`,
            name: name ?? "",
            argsBuffer: "",
            added: false,
          };
          toolCalls.set(idx, acc);
        } else if (id && !acc.callId.startsWith("call_")) {
          acc.callId = id;
        }
        if (name && !acc.name) acc.name = name;
        if (!acc.added && acc.name) {
          writeEvent("response.output_item.added", {
            type: "response.output_item.added",
            output_index: acc.outputIndex,
            item: {
              type: "function_call",
              id: acc.itemId,
              call_id: acc.callId,
              name: acc.name,
              arguments: "",
              status: "in_progress",
            },
          });
          acc.added = true;
        }
        return acc;
      };

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let sepIndex: number;
          while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
            const rawEvent = buffer.slice(0, sepIndex);
            buffer = buffer.slice(sepIndex + 2);

            for (const line of rawEvent.split("\n")) {
              if (!line.startsWith("data:")) continue;
              const data = line.slice(5).trim();
              if (!data || data === "[DONE]") continue;
              try {
                const parsed = JSON.parse(data);
                const nativeDelta = Array.isArray(parsed.choices) ? parsed.choices[0]?.delta : null;
                const token: string = nativeDelta?.content ?? parsed.response ?? parsed.delta ?? "";
                if (parsed.usage) upstreamUsage = parsed.usage;

                if (token) tagParser.feed(token);

                // Tool calls — both shapes:
                //   OpenAI native delta: { tool_calls:[{index, id?, function:{name?, arguments?}}] }
                //   Legacy CF: top-level tool_calls:[{name, arguments}]
                const deltaTcs = nativeDelta?.tool_calls ?? parsed.tool_calls;
                if (Array.isArray(deltaTcs) && deltaTcs.length) {
                  for (let i = 0; i < deltaTcs.length; i++) {
                    const tc: any = deltaTcs[i];
                    const idx = typeof tc.index === "number" ? tc.index : i;
                    const fnName = tc.function?.name ?? tc.name;
                    const acc = ensureToolCallStarted(idx, tc.id, fnName);
                    const argsChunk =
                      tc.function?.arguments ?? (typeof tc.arguments === "string" ? tc.arguments : tc.arguments ? JSON.stringify(tc.arguments) : "");
                    if (argsChunk) {
                      acc.argsBuffer += argsChunk;
                      writeEvent("response.function_call_arguments.delta", {
                        type: "response.function_call_arguments.delta",
                        item_id: acc.itemId,
                        output_index: acc.outputIndex,
                        delta: argsChunk,
                      });
                    }
                  }
                }
              } catch {
                // Ignore malformed chunks.
              }
            }
          }
        }
      } catch (err) {
        writeEvent("response.failed", {
          type: "response.failed",
          response: { ...initialResponse, status: "failed", error: { message: (err as Error).message } },
        });
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
        return;
      }

      // Flush any text/tool-call still buffered inside the inline-tag parser.
      tagParser.end();

      // Close the message item only if we actually started one.
      const hasText = fullText.length > 0;
      let finalMessageItem: unknown = null;
      if (messageStarted) {
        writeEvent("response.output_text.done", {
          type: "response.output_text.done",
          item_id: messageId,
          output_index: messageOutputIndex,
          content_index: 0,
          text: fullText,
        });
        writeEvent("response.content_part.done", {
          type: "response.content_part.done",
          item_id: messageId,
          output_index: messageOutputIndex,
          content_index: 0,
          part: { type: "output_text", text: fullText, annotations: [] },
        });
        finalMessageItem = {
          type: "message",
          id: messageId,
          status: "completed",
          role: "assistant",
          content: [{ type: "output_text", text: fullText, annotations: [] }],
        };
        writeEvent("response.output_item.done", {
          type: "response.output_item.done",
          output_index: messageOutputIndex,
          item: finalMessageItem,
        });
      }

      // Close each tool call.
      const finalToolItems: unknown[] = [];
      for (const acc of toolCalls.values()) {
        // Some upstream shapes deliver `arguments` as an object up front, not
        // streamed; ensure a final args.done event with whatever we have.
        const finalArgs = normalizeArguments(acc.argsBuffer || "{}");
        writeEvent("response.function_call_arguments.done", {
          type: "response.function_call_arguments.done",
          item_id: acc.itemId,
          output_index: acc.outputIndex,
          arguments: finalArgs,
        });
        const item = {
          type: "function_call",
          id: acc.itemId,
          call_id: acc.callId,
          name: acc.name,
          arguments: finalArgs,
          status: "completed",
        };
        finalToolItems.push(item);
        writeEvent("response.output_item.done", {
          type: "response.output_item.done",
          output_index: acc.outputIndex,
          item,
        });
      }

      const usage = {
        input_tokens: upstreamUsage?.prompt_tokens ?? 0,
        output_tokens: upstreamUsage?.completion_tokens ?? 0,
        total_tokens: upstreamUsage?.total_tokens
          ?? (upstreamUsage?.prompt_tokens ?? 0) + (upstreamUsage?.completion_tokens ?? 0),
      };
      const finalOutput = [...(finalMessageItem ? [finalMessageItem] : []), ...finalToolItems];

      writeEvent("response.completed", {
        type: "response.completed",
        response: {
          ...initialResponse,
          status: "completed",
          output: finalOutput,
          output_text: fullText,
          usage,
        },
      });
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  return new Response(out, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
