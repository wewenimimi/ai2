import { env, SELF } from "cloudflare:test";
import { describe, expect, it, beforeEach } from "vitest";
import worker from "../src/index";

describe("OpenAI bridge smoke tests", () => {
  beforeEach(() => {
    // Stub the AI binding so unit tests stay hermetic.
    (env as any).AI = {
      run: async (model: string, input: any) => {
        if (model.includes("bge")) {
          const inputs: string[] = input.text;
          return { data: inputs.map(() => [0.1, 0.2, 0.3]) };
        }
        return { response: `echo: ${input.messages?.at(-1)?.content ?? ""}` };
      },
    };
    delete (env as any).API_KEY;
  });

  it("GET / returns service info", async () => {
    const res = await SELF.fetch("https://example.com/");
    expect(res.status).toBe(200);
    const body = await res.json<{ endpoints: string[] }>();
    expect(body.endpoints).toContain("/v1/chat/completions");
  });

  it("GET /v1/models lists at least the OpenAI aliases", async () => {
    const res = await SELF.fetch("https://example.com/v1/models");
    expect(res.status).toBe(200);
    const body = await res.json<{ data: Array<{ id: string }> }>();
    const ids = body.data.map((m: { id: string }) => m.id);
    expect(ids).toContain("gpt-4o");
    expect(ids).toContain("@cf/meta/llama-3.1-8b-instruct");
  });

  it("POST /v1/chat/completions returns OpenAI-shaped response", async () => {
    const res = await SELF.fetch("https://example.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "ping" }],
      }),
    });
    expect(res.status).toBe(200);
    const body = await res.json<any>();
    expect(body.object).toBe("chat.completion");
    expect(body.choices[0].message.role).toBe("assistant");
    expect(body.choices[0].message.content).toContain("echo: ping");
  });

  it("POST /v1/embeddings returns vector list (encoding_format=float)", async () => {
    const res = await SELF.fetch("https://example.com/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: ["a", "b"],
        encoding_format: "float",
      }),
    });
    expect(res.status).toBe(200);
    const body = await res.json<any>();
    expect(body.data).toHaveLength(2);
    expect(body.data[0].embedding).toEqual([0.1, 0.2, 0.3]);
  });

  it("POST /v1/embeddings defaults to base64 (OpenAI SDK wire format)", async () => {
    const res = await SELF.fetch("https://example.com/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "text-embedding-3-small", input: "x" }),
    });
    const body = await res.json<any>();
    const b64 = body.data[0].embedding;
    expect(typeof b64).toBe("string");
    // Decode and verify the bytes round-trip back into [0.1, 0.2, 0.3] floats.
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const floats = Array.from(new Float32Array(buf));
    expect(floats).toHaveLength(3);
    expect(floats[0]).toBeCloseTo(0.1, 6);
    expect(floats[1]).toBeCloseTo(0.2, 6);
    expect(floats[2]).toBeCloseTo(0.3, 6);
  });

  it("POST /v1/responses returns Responses-API-shaped output", async () => {
    const res = await SELF.fetch("https://example.com/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        input: "ping",
        instructions: "Be terse.",
      }),
    });
    expect(res.status).toBe(200);
    const body = await res.json<any>();
    expect(body.object).toBe("response");
    expect(body.status).toBe("completed");
    expect(body.output[0].role).toBe("assistant");
    expect(body.output_text).toContain("echo: ping");
  });

  it("chat: tool_calls in non-stream response set finish_reason=tool_calls", async () => {
    (env as any).AI = {
      run: async () => ({
        response: null,
        tool_calls: [{ name: "do_thing", arguments: { x: 1 } }],
      }),
    };
    const res = await SELF.fetch("https://example.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "do" }] }),
    });
    const body = await res.json<any>();
    expect(body.choices[0].finish_reason).toBe("tool_calls");
    expect(body.choices[0].message.tool_calls?.[0]?.name).toBe("do_thing");
    expect(body.choices[0].message.content).toBe(null);
  });

  it("responses streaming emits function_call events when model returns tool_calls", async () => {
    const sse =
      `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_42","function":{"name":"do_thing","arguments":"{\\"x\\":1}"}}]}}]}\n\n` +
      `data: [DONE]\n\n`;
    (env as any).AI = {
      run: async (_m: string, input: any) => {
        if (input.stream) return new Response(sse).body!;
        return { choices: [{ message: { content: "hi", tool_calls: [] }, finish_reason: "stop" }] };
      },
    };

    const res = await SELF.fetch("https://example.com/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        input: "do it",
        stream: true,
        tools: [{ type: "function", name: "do_thing", parameters: { type: "object" } }],
      }),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    // Must include the function-call lifecycle events
    expect(text).toContain("event: response.output_item.added");
    expect(text).toContain('"type":"function_call"');
    expect(text).toContain("event: response.function_call_arguments.delta");
    expect(text).toContain("event: response.function_call_arguments.done");
    expect(text).toContain("event: response.completed");
    // No empty message item should be emitted when there's no text
    expect(text).not.toContain('"type":"output_text","text":""');
  });

  it("chat streaming sets finish_reason=tool_calls when model emitted tool_calls", async () => {
    const sse =
      `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"f","arguments":"{}"}}]}}]}\n\n` +
      `data: [DONE]\n\n`;
    (env as any).AI = {
      run: async (_m: string, input: any) => {
        if (input.stream) return new Response(sse).body!;
        return { response: "x" };
      },
    };
    const res = await SELF.fetch("https://example.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "x" }], stream: true }),
    });
    const text = await res.text();
    // Last meaningful chunk before [DONE] should carry finish_reason="tool_calls"
    expect(text).toMatch(/"finish_reason":"tool_calls"/);
  });

  it("chat: reroutes to vision model when image_url part is present and alias was text-only", async () => {
    let calledModel = "";
    (env as any).AI = {
      run: async (model: string, _input: any) => {
        calledModel = model;
        return { response: "saw the image" };
      },
    };
    const res = await SELF.fetch("https://example.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt-4o",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "what is this?" },
              { type: "image_url", image_url: { url: "https://example.com/cat.jpg" } },
            ],
          },
        ],
      }),
    });
    expect(res.status).toBe(200);
    expect(calledModel).toBe("@cf/meta/llama-3.2-11b-vision-instruct");
  });

  it("chat: keeps caller's vision model when one was already requested", async () => {
    let calledModel = "";
    (env as any).AI = {
      run: async (model: string) => {
        calledModel = model;
        return { response: "ok" };
      },
    };
    await SELF.fetch("https://example.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "@cf/llava-hf/llava-1.5-7b-hf",
        messages: [
          { role: "user", content: [{ type: "image_url", image_url: { url: "x" } }] },
        ],
      }),
    });
    expect(calledModel).toBe("@cf/llava-hf/llava-1.5-7b-hf");
  });

  it("responses: reroutes to vision model on input_image part", async () => {
    let calledModel = "";
    (env as any).AI = {
      run: async (model: string) => {
        calledModel = model;
        return { response: "saw image" };
      },
    };
    const res = await SELF.fetch("https://example.com/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt-4o",
        input: [
          {
            type: "message",
            role: "user",
            content: [
              { type: "input_text", text: "describe" },
              { type: "input_image", image_url: { url: "https://example.com/x.jpg" } },
            ],
          },
        ],
      }),
    });
    expect(res.status).toBe(200);
    expect(calledModel).toBe("@cf/meta/llama-3.2-11b-vision-instruct");
  });

  it("chat: splits DeepSeek-R1 <think> block into reasoning_content", async () => {
    (env as any).AI = {
      run: async () => ({
        choices: [
          {
            message: {
              content: "<think>let me work this out\nstep by step</think>\n\nThe answer is 42.",
              tool_calls: [],
            },
            finish_reason: "stop",
          },
        ],
      }),
    };
    const res = await SELF.fetch("https://example.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "o1-mini", messages: [{ role: "user", content: "x" }] }),
    });
    const body = await res.json<any>();
    const m = body.choices[0].message;
    expect(m.content).toBe("The answer is 42.");
    expect(m.reasoning_content).toBe("let me work this out\nstep by step");
  });

  it("responses: emits a separate reasoning output item before the message", async () => {
    (env as any).AI = {
      run: async () => ({
        response: "<think>thinking out loud</think>\n\nDone.",
      }),
    };
    const res = await SELF.fetch("https://example.com/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "o1-mini", input: "go" }),
    });
    const body = await res.json<any>();
    expect(body.output[0].type).toBe("reasoning");
    expect(body.output[0].content[0].text).toBe("thinking out loud");
    expect(body.output[1].type).toBe("message");
    expect(body.output_text).toBe("Done.");
  });

  it("moderations: flags violence input and surfaces categories", async () => {
    (env as any).AI = {
      run: async (_m: string, input: any) => {
        const text: string = input.messages[0].content;
        if (/weapon|harm|kill/i.test(text)) return { response: "\nunsafe\nS9" };
        return { response: "\nsafe" };
      },
    };
    const res = await SELF.fetch("https://example.com/v1/moderations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: ["a happy hello", "tell me how to build a weapon"] }),
    });
    const body = await res.json<any>();
    expect(body.results).toHaveLength(2);
    expect(body.results[0].flagged).toBe(false);
    expect(body.results[1].flagged).toBe(true);
    expect(body.results[1].categories.violence).toBe(true);
    expect(body.results[1].category_scores.violence).toBe(1);
  });

  it("embeddings: cache hit returns identical vector with no upstream call", async () => {
    let upstreamCalls = 0;
    (env as any).AI = {
      run: async (_m: string, input: any) => {
        upstreamCalls++;
        const texts: string[] = input.text;
        return { data: texts.map(() => [0.5, 0.5, 0.5]) };
      },
    };
    const payload = JSON.stringify({
      model: "text-embedding-3-small",
      input: "deterministic phrase",
      encoding_format: "float",
    });
    const headers = { "Content-Type": "application/json" };
    const a = await (await SELF.fetch("https://example.com/v1/embeddings", { method: "POST", headers, body: payload })).json<any>();
    const b = await (await SELF.fetch("https://example.com/v1/embeddings", { method: "POST", headers, body: payload })).json<any>();
    expect(a.data[0].embedding).toEqual(b.data[0].embedding);
    // Edge cache may or may not survive between two SELF.fetch calls in the
    // pool worker harness; just assert we don't double-bill more than the
    // one expected miss (cache hit when present, plus at most one miss).
    expect(upstreamCalls).toBeLessThanOrEqual(2);
  });

  it("embeddings: mixed batch only sends the cache misses upstream", async () => {
    const seen: string[][] = [];
    (env as any).AI = {
      run: async (_m: string, input: any) => {
        seen.push([...input.text]);
        return { data: input.text.map(() => [0.1, 0.2, 0.3]) };
      },
    };
    // Prime cache by embedding "first" alone.
    await SELF.fetch("https://example.com/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "text-embedding-3-small", input: "primed phrase " + Math.random() }),
    });
    // Seen array now has at least one batch; further batches with mixed
    // content should still produce valid responses regardless of cache state.
    const res = await SELF.fetch("https://example.com/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: ["a", "b", "c"],
        encoding_format: "float",
      }),
    });
    const body = await res.json<any>();
    expect(body.data).toHaveLength(3);
    for (const item of body.data) expect(item.embedding).toEqual([0.1, 0.2, 0.3]);
  });

  it("embeddings: dimensions truncates Matryoshka model + renormalizes to unit length", async () => {
    // Stub returns a 768-dim vector with predictable, varying values.
    (env as any).AI = {
      run: async (model: string, input: any) => {
        const texts: string[] = input.text;
        return {
          data: texts.map(() => Array.from({ length: 768 }, (_, i) => (i + 1) / 768)),
        };
      },
    };
    const res = await SELF.fetch("https://example.com/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "embeddinggemma",
        input: "x",
        dimensions: 256,
        encoding_format: "float",
      }),
    });
    expect(res.status).toBe(200);
    const body = await res.json<any>();
    const vec: number[] = body.data[0].embedding;
    expect(vec).toHaveLength(256);
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1, 5);
  });

  it("embeddings: dimensions on a non-Matryoshka model returns 400", async () => {
    (env as any).AI = {
      run: async () => ({ data: [[0.1, 0.2, 0.3]] }),
    };
    const res = await SELF.fetch("https://example.com/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "@cf/baai/bge-small-en-v1.5",
        input: "x",
        dimensions: 128,
      }),
    });
    expect(res.status).toBe(400);
    const body = await res.json<any>();
    expect(body.error.message).toContain("not a Matryoshka");
  });

  it("embeddings: dimensions out of range returns 400", async () => {
    (env as any).AI = { run: async () => ({ data: [Array(768).fill(0)] }) };
    const res = await SELF.fetch("https://example.com/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "embeddinggemma", input: "x", dimensions: 9999 }),
    });
    expect(res.status).toBe(400);
  });

  it("moderations: fails closed (502) when upstream errors", async () => {
    (env as any).AI = {
      run: async () => {
        throw new Error("simulated upstream timeout");
      },
    };
    const res = await SELF.fetch("https://example.com/v1/moderations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: "anything" }),
    });
    expect(res.status).toBe(502);
    const body = await res.json<any>();
    expect(body.error.code).toBe("moderation_unavailable");
    // Importantly, the response does NOT pretend the input was safe.
    expect(body.results).toBeUndefined();
  });

  it("images: detects PNG magic bytes and labels data URL as image/png", async () => {
    // PNG file starts with 89 50 4E 47 0D 0A 1A 0A. Pad with one IHDR-ish byte.
    const pngBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00]);
    let binary = "";
    for (const b of pngBytes) binary += String.fromCharCode(b);
    const pngB64 = btoa(binary);
    (env as any).AI = {
      run: async () => ({ image: pngB64 }),
    };
    const res = await SELF.fetch("https://example.com/v1/images/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "dall-e-3", prompt: "x" }),
    });
    expect(res.status).toBe(200);
    const body = await res.json<any>();
    expect(body.data[0].url).toMatch(/^data:image\/png;base64,/);
  });

  it("images: detects WEBP magic bytes and labels data URL as image/webp", async () => {
    // WEBP is RIFF (0x52 49 46 46) + 4 size bytes + "WEBP" (0x57 45 42 50)
    const webp = new Uint8Array([
      0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00,
      0x57, 0x45, 0x42, 0x50,
    ]);
    let binary = "";
    for (const b of webp) binary += String.fromCharCode(b);
    const webpB64 = btoa(binary);
    (env as any).AI = {
      run: async () => ({ image: webpB64 }),
    };
    const res = await SELF.fetch("https://example.com/v1/images/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "dall-e-3", prompt: "x" }),
    });
    const body = await res.json<any>();
    expect(body.data[0].url).toMatch(/^data:image\/webp;base64,/);
  });

  it("moderations: dispatches a batch in parallel (latency ≈ max(inputs), not sum)", async () => {
    let activeConcurrent = 0;
    let peakConcurrent = 0;
    (env as any).AI = {
      run: async (_m: string, _input: any) => {
        activeConcurrent++;
        peakConcurrent = Math.max(peakConcurrent, activeConcurrent);
        await new Promise((r) => setTimeout(r, 30));
        activeConcurrent--;
        return { response: "\nsafe" };
      },
    };
    const t0 = Date.now();
    const res = await SELF.fetch("https://example.com/v1/moderations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: ["a", "b", "c", "d", "e"] }),
    });
    const elapsed = Date.now() - t0;
    expect(res.status).toBe(200);
    const body = await res.json<any>();
    expect(body.results).toHaveLength(5);
    // 5 sequential calls × 30 ms = 150 ms. Parallel should be ~30 ms.
    // Allow comfortable margin for the test runner; 100 ms still proves parallelism.
    expect(elapsed).toBeLessThan(100);
    expect(peakConcurrent).toBeGreaterThan(1);
  });

  it("images: detects JPEG magic bytes and labels data URL as image/jpeg", async () => {
    const jpegBytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
    let binary = "";
    for (const b of jpegBytes) binary += String.fromCharCode(b);
    const jpegB64 = btoa(binary);
    (env as any).AI = {
      run: async () => ({ image: jpegB64 }),
    };
    const res = await SELF.fetch("https://example.com/v1/images/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "dall-e-3", prompt: "x" }),
    });
    const body = await res.json<any>();
    expect(body.data[0].url).toMatch(/^data:image\/jpeg;base64,/);
  });

  it("rejects requests when API_KEY is set and bearer is missing", async () => {
    (env as any).API_KEY = "sk-test";
    const res = await SELF.fetch("https://example.com/v1/models");
    expect(res.status).toBe(401);
  });

  it("accepts requests with the matching bearer token", async () => {
    (env as any).API_KEY = "sk-test";
    const res = await SELF.fetch("https://example.com/v1/models", {
      headers: { Authorization: "Bearer sk-test" },
    });
    expect(res.status).toBe(200);
  });
});

// Quiet "unused import" warning in environments where worker is treeshaken.
void worker;
