import type { Context } from "hono";
import { runAI } from "./ai-client";
import { inlineImageUrls } from "./image-inline";
import { isVisionModel, resolveChatModel, VISION_DEFAULT_MODEL } from "./mapping";
import { createToolCallStreamParser } from "./tool-call-parser";
import type { ChatCompletionRequest, ChatMessage, Env } from "./types";

// Adapt OpenAI multi-part content for Workers AI. Text-only messages are
// flattened to a string (what most CF chat models expect); messages that
// include images are forwarded as the multi-part array so vision-capable
// models (e.g. @cf/meta/llama-3.2-11b-vision-instruct) receive them intact.
function adaptContent(content: ChatMessage["content"]): string | unknown[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  const hasImage = content.some((p) => p && p.type === "image_url");
  if (hasImage) return content;

  return content
    .map((part) => (part.type === "text" && part.text ? part.text : ""))
    .filter(Boolean)
    .join("\n");
}

// Normalize tool definitions in case a caller sends the Responses API shape
// (`{type:"function", name, parameters}`) into chat.completions, which expects
// the wrapped form (`{type:"function", function:{name, parameters}}`).
function adaptTools(tools: unknown[]): unknown[] {
  return tools.map((t: any) => {
    if (!t || typeof t !== "object") return t;
    if (t.type === "function" && t.function && typeof t.function === "object") return t;
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

function adaptMessages(messages: ChatMessage[]) {
  return messages.map((m) => ({
    role: m.role,
    content: adaptContent(m.content),
    ...(m.tool_call_id ? { tool_call_id: m.tool_call_id } : {}),
    ...(m.tool_calls ? { tool_calls: m.tool_calls } : {}),
  }));
}

function messagesContainImage(messages: ChatMessage[]): boolean {
  for (const m of messages) {
    const c = m.content;
    if (Array.isArray(c) && c.some((p) => p && (p as any).type === "image_url")) return true;
  }
  return false;
}

function generateId(): string {
  return "chatcmpl-" + crypto.randomUUID().replace(/-/g, "").slice(0, 24);
}

// Two reasoning conventions on Workers AI:
//   - <think>...</think> inside content (DeepSeek-R1, QwQ)
//   - a separate `message.reasoning` field (Gemma 4, OpenAI o-series)
// We unify both into `reasoning_content` on the outgoing chat.completion.
function splitReasoning(raw: string): { content: string; reasoning?: string } {
  if (!raw) return { content: raw };
  const match = raw.match(/^\s*<think>([\s\S]*?)<\/think>\s*([\s\S]*)$/);
  if (!match) return { content: raw };
  return { content: match[2].trim(), reasoning: match[1].trim() };
}

export async function handleChatCompletions(c: Context<{ Bindings: Env }>) {
  let body: ChatCompletionRequest;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid JSON body", type: "invalid_request_error" } }, 400);
  }

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return c.json({ error: { message: "`messages` is required", type: "invalid_request_error" } }, 400);
  }

  let model = resolveChatModel(body.model, c.env.DEFAULT_CHAT_MODEL ?? "@cf/meta/llama-3.3-70b-instruct-fp8-fast");
  // Auto-route to a vision model when the request contains image_url parts
  // and the resolved model isn't vision-capable. The caller's `body.model`
  // (e.g. "gpt-4o") still appears in the response so OpenAI clients see what
  // they sent.
  const hasImage = messagesContainImage(body.messages);
  if (hasImage && !isVisionModel(model)) {
    model = VISION_DEFAULT_MODEL;
  }
  const stream = body.stream === true;

  const adaptedMessages = adaptMessages(body.messages);
  // Workers AI vision models accept only data: URIs. Fetch any remote
  // image_url and inline it as base64 before the upstream call.
  if (hasImage) await inlineImageUrls(adaptedMessages);

  const aiInput: Record<string, unknown> = {
    messages: adaptedMessages,
    stream,
  };
  if (typeof body.temperature === "number") aiInput.temperature = body.temperature;
  if (typeof body.top_p === "number") aiInput.top_p = body.top_p;
  if (typeof (body as any).top_k === "number") aiInput.top_k = (body as any).top_k;
  if (typeof body.max_tokens === "number") aiInput.max_tokens = body.max_tokens;
  if (typeof body.max_completion_tokens === "number") aiInput.max_tokens = body.max_completion_tokens;
  if (typeof body.frequency_penalty === "number") aiInput.frequency_penalty = body.frequency_penalty;
  if (typeof body.presence_penalty === "number") aiInput.presence_penalty = body.presence_penalty;
  if (typeof (body as any).repetition_penalty === "number") aiInput.repetition_penalty = (body as any).repetition_penalty;
  if (typeof body.seed === "number") aiInput.seed = body.seed;
  if (body.stop !== undefined) aiInput.stop = body.stop;
  if (Array.isArray(body.tools) && body.tools.length > 0) aiInput.tools = adaptTools(body.tools);
  if (body.tool_choice !== undefined) aiInput.tool_choice = body.tool_choice;
  if (body.response_format) aiInput.response_format = body.response_format;

  const id = generateId();
  const created = Math.floor(Date.now() / 1000);

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

    // Workers AI returns one of two shapes depending on the model:
    //   - Legacy (Llama/Mistral/...): { response: "text", tool_calls?, usage? }
    //   - OpenAI-native (IBM Granite, DeepSeek-R1, newer models):
    //       { choices: [{ message: { content, tool_calls } , finish_reason }], usage }
    const nativeChoice = Array.isArray(result?.choices) ? result.choices[0] : null;
    const rawText: string = nativeChoice?.message?.content
      ?? (typeof result?.response === "string" ? result.response : null)
      ?? (typeof result?.result?.response === "string" ? result.result.response : null)
      ?? "";
    const { content: text, reasoning: thinkBlockReasoning } = splitReasoning(rawText);
    // Gemma 4 (and OpenAI's o-series shape on Workers AI) puts the chain of
    // thought in `message.reasoning` directly. Prefer that when both are
    // present; otherwise fall back to whatever we extracted from <think>.
    const nativeReasoning =
      typeof nativeChoice?.message?.reasoning === "string" && nativeChoice.message.reasoning.trim()
        ? nativeChoice.message.reasoning.trim()
        : null;
    const reasoning = nativeReasoning ?? thinkBlockReasoning;
    const toolCalls = nativeChoice?.message?.tool_calls?.length
      ? nativeChoice.message.tool_calls
      : (result?.tool_calls?.length ? result.tool_calls : undefined);
    const finishReason = nativeChoice?.finish_reason ?? (toolCalls ? "tool_calls" : "stop");
    const usage = result?.usage ?? {};

    return c.json({
      id,
      object: "chat.completion",
      created,
      model: body.model || model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: toolCalls ? null : text,
            ...(reasoning ? { reasoning_content: reasoning } : {}),
            ...(toolCalls ? { tool_calls: toolCalls } : {}),
          },
          finish_reason: finishReason,
        },
      ],
      usage: {
        prompt_tokens: usage.prompt_tokens ?? 0,
        completion_tokens: usage.completion_tokens ?? 0,
        total_tokens: usage.total_tokens ?? (usage.prompt_tokens ?? 0) + (usage.completion_tokens ?? 0),
      },
    });
  }

  // Streaming branch: convert Workers AI SSE → OpenAI SSE deltas.
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
  const modelLabel = body.model || model;

  const out = new ReadableStream<Uint8Array>({
    async start(controller) {
      const writeEvent = (payload: unknown) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      };

      // Initial role delta — many OpenAI clients expect it before any content.
      writeEvent({
        id,
        object: "chat.completion.chunk",
        created,
        model: modelLabel,
        choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
      });

      const reader = upstream.getReader();
      let buffer = "";
      let sawToolCalls = false;
      let upstreamFinishReason: string | null = null;
      let upstreamUsage: any = null;
      let nextSyntheticToolIdx = 10000;

      // Inline-tag parser (Hermes / Mistral chat-template models stream tool
      // calls as raw `<tool_call>{...}</tool_call>` content tokens). Text
      // outside tags becomes a normal content delta; parsed tool calls become
      // synthetic tool_calls deltas in chat.completions shape.
      const tagParser = createToolCallStreamParser({
        onText: (chunk: string) => {
          writeEvent({
            id,
            object: "chat.completion.chunk",
            created,
            model: modelLabel,
            choices: [{ index: 0, delta: { content: chunk }, finish_reason: null }],
          });
        },
        onToolCall: ({ name, arguments: args }) => {
          sawToolCalls = true;
          const idx = nextSyntheticToolIdx++;
          writeEvent({
            id,
            object: "chat.completion.chunk",
            created,
            model: modelLabel,
            choices: [
              {
                index: 0,
                delta: {
                  tool_calls: [
                    {
                      index: idx,
                      id: `call_${idx}`,
                      type: "function",
                      function: { name, arguments: args },
                    },
                  ],
                },
                finish_reason: null,
              },
            ],
          });
        },
      });

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          // Parse SSE: events separated by blank lines, lines start with "data: ".
          let sepIndex: number;
          while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
            const rawEvent = buffer.slice(0, sepIndex);
            buffer = buffer.slice(sepIndex + 2);

            for (const line of rawEvent.split("\n")) {
              if (!line.startsWith("data:")) continue;
              const data = line.slice(5).trim();
              if (!data) continue;
              if (data === "[DONE]") continue;

              try {
                const parsed = JSON.parse(data);
                // Two upstream shapes:
                //   Legacy: { response: "token", tool_calls?, p?, usage? }
                //   OpenAI-native (Granite/DeepSeek-R1): { choices:[{ delta:{ content, tool_calls }, finish_reason }], usage? }
                const nativeChoice = Array.isArray(parsed.choices) ? parsed.choices[0] : null;
                const nativeDelta = nativeChoice?.delta ?? null;
                const token: string = nativeDelta?.content ?? parsed.response ?? parsed.delta ?? "";
                const deltaToolCalls = nativeDelta?.tool_calls ?? parsed.tool_calls;

                if (parsed.usage) upstreamUsage = parsed.usage;
                if (nativeChoice?.finish_reason) upstreamFinishReason = nativeChoice.finish_reason;

                if (token) tagParser.feed(token);
                if (deltaToolCalls && deltaToolCalls.length) {
                  sawToolCalls = true;
                  // Normalize legacy `[{name, arguments}]` into chat.completions delta shape.
                  const normalized = deltaToolCalls.map((tc: any, idx: number) => {
                    if (tc?.function) return { index: tc.index ?? idx, ...tc };
                    return {
                      index: idx,
                      id: tc.id ?? `call_${idx}`,
                      type: "function",
                      function: {
                        name: tc.name,
                        arguments: typeof tc.arguments === "string" ? tc.arguments : JSON.stringify(tc.arguments ?? {}),
                      },
                    };
                  });
                  writeEvent({
                    id,
                    object: "chat.completion.chunk",
                    created,
                    model: modelLabel,
                    choices: [{ index: 0, delta: { tool_calls: normalized }, finish_reason: null }],
                  });
                }
              } catch {
                // Ignore malformed chunks; Workers AI occasionally emits keep-alives.
              }
            }
          }
        }
      } catch (err) {
        // Stream broke mid-flight. Best we can do per the OpenAI streaming
        // shape: log, emit a clean stop chunk, and close. The error field on
        // a delta chunk is non-standard and breaks some clients.
        console.error("[/v1/chat] stream error:", (err as Error).message);
      }

      // Flush anything left buffered inside the inline-tag parser.
      tagParser.end();

      const finishReason = upstreamFinishReason ?? (sawToolCalls ? "tool_calls" : "stop");
      const finalChunk: Record<string, unknown> = {
        id,
        object: "chat.completion.chunk",
        created,
        model: modelLabel,
        choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
      };
      if (upstreamUsage) {
        finalChunk.usage = {
          prompt_tokens: upstreamUsage.prompt_tokens ?? 0,
          completion_tokens: upstreamUsage.completion_tokens ?? 0,
          total_tokens: upstreamUsage.total_tokens ?? (upstreamUsage.prompt_tokens ?? 0) + (upstreamUsage.completion_tokens ?? 0),
        };
      }
      writeEvent(finalChunk);
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
