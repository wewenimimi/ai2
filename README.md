# OpenAI ↔ Workers AI Bridge

Drop-in OpenAI-compatible API for **Cloudflare Workers AI**. Deploy this Worker to your own Cloudflare account and any tool that speaks the OpenAI API — **n8n**, **LibreChat**, **Open WebUI**, **Cursor**, **Continue.dev**, the OpenAI SDKs, **LangChain** — can talk to every Workers AI capability:

- **Chat + tools + vision + reasoning** — Gemma 4 (26B), Llama 3.3 (70B), Llama 3.2-vision (11B), Granite 4, Mistral Small 3.1, Hermes-2-Pro, DeepSeek-R1, QwQ 32B, Qwen 3 30B, Qwen2.5-coder
- **Embeddings + Matryoshka truncation** — EmbeddingGemma 300M, BGE (small / base / large / m3), Qwen3-embedding
- **Speech-to-text + translation** — Whisper, Whisper-large-v3-turbo
- **Text-to-speech** — MeloTTS (multilingual), Deepgram Aura-1 (English)
- **Image generation** — Flux schnell, SDXL Lightning, Dreamshaper
- **Moderation** — Llama Guard 3 (8B)

[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/MauricioPerera/openai-workers-ai-bridge)

## Endpoints

| OpenAI endpoint | Status | Notes |
|---|---|---|
| `GET  /v1/models` | ✅ | OpenAI-style aliases (`gpt-4o`, `text-embedding-3-small`, `tts-1`, `dall-e-3`) plus native `@cf/...` IDs. |
| `POST /v1/chat/completions` | ✅ | Streaming (SSE) and non-streaming. Tool / function calling. Vision via `image_url` parts (auto-routes to a vision model and inlines remote URLs as base64). Reasoning models (`o1`, `o3`, `gemma-4`) return chain-of-thought as `message.reasoning_content`. |
| `POST /v1/responses` | ✅ | OpenAI's newer Responses API. Streaming + non-streaming. Multi-turn agent loops with `function_call` / `function_call_output` items. Used by recent n8n / LangChain.js / OpenAI SDK helpers. |
| `POST /v1/embeddings` | ✅ | **Matryoshka truncation via `dimensions`** parameter (EmbeddingGemma, BGE-M3, Qwen3-embedding). **Base64-encoded vectors by default** to match the OpenAI Node SDK wire format; pass `encoding_format: "float"` for JSON arrays. **Edge-cached** by SHA-256 of `(model, text, dim)` — repeat queries cost 0 neurons. String or array input. *Dimensions differ from OpenAI — see below.* |
| `POST /v1/audio/transcriptions` | ✅ | Whisper (`whisper-1` → `@cf/openai/whisper`, or `whisper-large-v3-turbo`). OpenAI extras forwarded: `prompt` (decoder vocab hint), `temperature`, `timestamp_granularities[]`. Output formats: `json`, `text`, `verbose_json` (with `segments` and word-level timestamps), `vtt`, `srt`. |
| `POST /v1/audio/translations` | ✅ | Same shape as transcriptions, but returns English regardless of source language (Whisper translate task on `whisper-large-v3-turbo`). |
| `POST /v1/audio/speech` | ✅ | TTS via `@cf/myshell-ai/melotts` (multilingual, WAV) or `@cf/deepgram/aura-1` (English, MP3). |
| `POST /v1/images/generations` | ✅ | Flux schnell, SDXL Lightning, Dreamshaper. Returns `b64_json` or a `data:` URL. |
| `POST /v1/moderations` | ✅ | `@cf/meta/llama-guard-3-8b`. S1-S14 hazard categories mapped onto OpenAI's `categories` shape. |

Any model id starting with `@<provider>/` (e.g. `@cf/...`, `@hf/...`) is forwarded to Workers AI verbatim, so you can target every model your account has access to without waiting for an alias.

## What makes this different from a one-night-hack bridge

- **Multi-turn tool calling on `/v1/responses` actually works.** Each new request's `input` array carries the conversation history including the previous turn's **`function_call` items** (the tool invocations the *model* emitted on the prior turn) and **`function_call_output` items** (the results the *agent* fed back after executing those tools). The bridge translates both into the chat-completions equivalents — assistant messages with `tool_calls`, and `role:"tool"` messages with the tool result — so the model "remembers" its own prior decisions and the agent loop continues across turns. Most agents in n8n / LangChain rely on this loop and silently fail when a proxy drops these items.
- **Streaming tool calls emit the right Responses API events** (`response.output_item.added` with `type:"function_call"`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.output_item.done`) — the lifecycle that LangChain's `responses.stream()` and n8n's agent nodes consume.
- **Hermes / Mistral chat-template models** stream tool calls as raw `<tool_call>...</tool_call>` content tokens because Workers AI doesn't parse the XML in streaming mode. The bridge ships a streaming state machine that detects, buffers and parses these blocks (including Python-style single-quoted JSON some Hermes builds emit) and emits proper `function_call` events.
- **Dual-shape upstream parsing.** Workers AI returns either the legacy `{response: "...", tool_calls: [...]}` shape (Llama, Mistral) or the OpenAI-native `{choices:[{message:{content,tool_calls},finish_reason}]}` shape (Granite, DeepSeek-R1, newer models). The bridge handles both transparently.
- **Granite / DeepSeek-R1 double-encoded `arguments` get unwrapped.** Some models hand back `tool_calls[].arguments` as a JSON-encoded string of a JSON string. The bridge normalizes them so `JSON.parse(arguments)` on the client side just works.
- **Vision auto-routing.** Send `gpt-4o` with an `image_url` part and the bridge reroutes to `@cf/meta/llama-3.2-11b-vision-instruct` automatically. The caller's model id is preserved in the response so OpenAI clients see what they sent.
- **Remote image URLs are auto-inlined.** Workers AI vision models reject `https://` URLs and require `data:` URIs. The bridge fetches the URL (with a User-Agent so Wikipedia / GitHub user-content don't 400), validates content-type, caps at 10 MB, and sends the base64 data URI upstream.
- **Optional REST API path.** The `env.AI` binding has been observed to behave inconsistently for tool calling depending on the calling context. Setting `CLOUDFLARE_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` as Worker secrets switches the bridge to the public REST endpoint, which has been more reliable. The binding remains the default when no token is configured.
- **Constant-time API key compare.** Cheap and free — no reason to be timing-channel sleepy on bearer auth.
- **Reasoning split — two formats unified.** DeepSeek-R1 / QwQ emit a **leading `<think>...</think>` block followed by the final answer**, both inside `message.content` as a single string (e.g. `"<think>\nlet me work this out\n</think>\n\nThe answer is X."`). Gemma 4 instead returns the chain-of-thought as a **separate `message.reasoning` string field**, sibling of `message.content` (matching OpenAI's o-series response shape). The bridge detects both upstream shapes and unifies them — the response always exposes the trace as `message.reasoning_content` (chat.completions) or as a `{type:"reasoning"}` output item (Responses API), with `content` containing only the final answer.
- **Gemma 4 — vision + reasoning + tools in one model.** Google's `gemma-4-26b-a4b-it` does all three: pass `image_url` parts, pass `tools`, get back a real `function_call` plus the model's reasoning trace. Available via `gemma-4` alias or the native `@cf/google/gemma-4-26b-a4b-it` id.
- **Matryoshka embedding truncation.** EmbeddingGemma, BGE-M3, and Qwen3-embedding are trained for it; the bridge accepts OpenAI's `dimensions` parameter and truncates + L2-renormalizes against those models. Non-Matryoshka models reject `dimensions` with a 400 instead of corrupting your vectors silently.
- **Embeddings cache.** SHA-256 of `(model, text, dim)` keyed against the edge Cache API for a week — `dim` is part of the key so a 256-dim Matryoshka truncation and a full-dim request to the same model don't poison each other. RAG pipelines that repeat queries pay neurons once; the second call returns the same vector for free.
- **Optional rate limiting.** Add a `[[unsafe.bindings]]` ratelimit binding in `wrangler.toml` and `/v1/*` is throttled per-API-key (or per-IP). Without the binding, no rate limiting — the deploy template still works.

## Deploy

### One click

Click the **Deploy to Cloudflare** badge above. Cloudflare clones the repo into your account, installs deps, binds Workers AI automatically, and deploys the Worker.

(Recommended) Add an API key so the endpoint isn't open to the world:
```bash
wrangler secret put API_KEY
# paste any string, e.g. sk-myproject-7f3a...
```

### From the CLI

```bash
git clone https://github.com/MauricioPerera/openai-workers-ai-bridge.git
cd openai-workers-ai-bridge
npm install
npx wrangler login
npx wrangler deploy
npx wrangler secret put API_KEY                  # optional but recommended
npx wrangler secret put CLOUDFLARE_TOKEN         # optional — see below
npx wrangler secret put CLOUDFLARE_ACCOUNT_ID    # optional — see below
```

`CLOUDFLARE_TOKEN` is an API token with **Workers AI Read + Workers AI Write** scope (create one at https://dash.cloudflare.com/profile/api-tokens). When set, the bridge calls Workers AI via the REST endpoint instead of the AI binding.

## Local development

```bash
cp .dev.vars.example .dev.vars   # edit API_KEY for local auth (optional)
npm install
npm run dev                       # http://127.0.0.1:8787
npm test                          # hermetic unit tests via vitest-pool-workers
```

Sanity checks:
```bash
# Chat
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'

# Vision (URL is fetched and inlined automatically)
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":[
       {"type":"text","text":"one word for this image"},
       {"type":"image_url","image_url":{"url":"https://example.com/cat.jpg"}}
     ]}]}'

# Whisper
curl http://127.0.0.1:8787/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-local-dev-change-me" \
  -F "file=@audio.mp3" -F "model=whisper-1"

# TTS
curl http://127.0.0.1:8787/v1/audio/speech \
  -H "Authorization: Bearer sk-local-dev-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"hello world","voice":"alloy"}' \
  -o out.wav

# Image generation
curl http://127.0.0.1:8787/v1/images/generations \
  -H "Authorization: Bearer sk-local-dev-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"dall-e-3","prompt":"a smiling sun, flat illustration","response_format":"b64_json"}'
```

## Use it in n8n

1. **Credentials → New → OpenAI**:
   - API Key: whatever you set with `wrangler secret put API_KEY`
   - Base URL: `https://openai-workers-ai-bridge.<subdomain>.workers.dev/v1`
2. Use the credential in any **OpenAI Chat Model** node, **Embeddings OpenAI** node, or as the LLM in an **AI Agent** / **Chain** node. Pick `gpt-4o`, `gpt-4o-mini`, or any `@cf/...` / `@hf/...` id directly.
3. The agent loop — multi-turn tool calling, function_call → function_call_output round-trips, streaming — works on **non-vision text models**: Llama 3.3 70B, Hermes-2-Pro, Granite, DeepSeek-R1, Qwen.

## Use it in LibreChat

In `librechat.yaml`:
```yaml
endpoints:
  custom:
    - name: "Workers AI"
      apiKey: "${WORKERS_AI_KEY}"
      baseURL: "https://openai-workers-ai-bridge.<subdomain>.workers.dev/v1"
      models:
        default: ["gpt-4o-mini", "gpt-4o", "@cf/ibm-granite/granite-4.0-h-micro", "@hf/nousresearch/hermes-2-pro-mistral-7b"]
        fetch: false
      titleConvo: true
      titleModel: "gpt-4o-mini"
      modelDisplayLabel: "Workers AI"
```

For TTS / image features in LibreChat, point its TTS service and image generation at the same base URL.

## Use it from the OpenAI SDK

```ts
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.WORKERS_AI_KEY,
  baseURL: "https://openai-workers-ai-bridge.<subdomain>.workers.dev/v1",
});

// Chat with streaming
const res = await client.chat.completions.create({
  model: "gpt-4o",                              // → @cf/meta/llama-3.3-70b-instruct-fp8-fast
  messages: [{ role: "user", content: "hi" }],
  stream: true,
});
for await (const chunk of res) process.stdout.write(chunk.choices[0]?.delta?.content ?? "");

// Image
const img = await client.images.generate({
  model: "dall-e-3",                            // → @cf/black-forest-labs/flux-1-schnell
  prompt: "a smiling cartoon sun",
  response_format: "b64_json",
});
```

## Embeddings — dimensions, Matryoshka, and the OpenAI mismatch

Workers AI's BGE family does **not** match OpenAI's embedding dimensions. If you have vectors stored from OpenAI and try to query through this bridge, similarity scores will be wrong (different vector spaces, different dimensions).

| Alias | Routes to | Native dim | OpenAI native |
|---|---|---|---|
| `text-embedding-ada-002` | `@cf/baai/bge-base-en-v1.5` | 768 | 1536 |
| `text-embedding-3-small` | `@cf/baai/bge-small-en-v1.5` | 384 | 1536 |
| `text-embedding-3-large` | `@cf/baai/bge-large-en-v1.5` | 1024 | 3072 |
| `embeddinggemma` / `gemma-embedding` | `@cf/google/embeddinggemma-300m` | 768 | — |

Either re-embed your existing corpus through this bridge once, or target a Workers AI embedding model directly with `@cf/...`.

### Matryoshka truncation (`dimensions` parameter)

Three Workers AI embedding models are trained with Matryoshka representation learning, which means a prefix of the vector is still a valid embedding in the same semantic space. The bridge supports OpenAI's `dimensions` parameter against these models — it truncates to the requested size and L2-renormalizes so cosine similarity stays consistent.

| Model | Native | Recommended tiers |
|---|---|---|
| `@cf/google/embeddinggemma-300m` | 768 | 128, 256, 512, 768 |
| `@cf/baai/bge-m3` | 1024 | 256, 512, 1024 |
| `@cf/qwen/qwen3-embedding-0.6b` | 1024 | 256, 512, 1024 |

```bash
curl .../v1/embeddings \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"embeddinggemma","input":"...","dimensions":256}'
```

Sending `dimensions` to a non-Matryoshka model (BGE small/base/large) returns a 400 with an explanation — the prefix of those vectors is meaningless and silently truncating would corrupt your corpus.

EmbeddingGemma is multilingual (100+ languages) and currently the strongest small embedder on Workers AI for cross-lingual retrieval.

## Configuration

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `API_KEY` | secret | *(unset)* | If set, requires `Authorization: Bearer <API_KEY>` on `/v1/*`. |
| `CLOUDFLARE_TOKEN` | secret | *(unset)* | Optional. With `CLOUDFLARE_ACCOUNT_ID`, switches the bridge to the Workers AI REST API instead of the binding. Requires `Workers AI Read + Write` scope. |
| `CLOUDFLARE_ACCOUNT_ID` | secret | *(unset)* | Pair with `CLOUDFLARE_TOKEN`. |
| `DEFAULT_CHAT_MODEL` | var | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | Used when the request's `model` doesn't match a known alias. |
| `DEFAULT_EMBEDDING_MODEL` | var | `@cf/baai/bge-m3` | Same, for `/v1/embeddings`. |

Set vars in `wrangler.toml` (or the Cloudflare dashboard); set secrets with `wrangler secret put NAME`.

## Limitations & notes

- **You pay** — calls hit Workers AI on *your* Cloudflare account; you pay for the neurons consumed.
- **Token usage on streaming** — Workers AI doesn't always emit `usage` on the final SSE chunk; `usage` may be zero in streaming responses.
- **Image generation `n>1`** — Workers AI returns one image per call. The bridge always emits `data[0]`.
- **TTS transcoding** — `response_format` is accepted but the bridge does not transcode codecs. melotts emits WAV, aura emits MP3, and the bridge serves whichever the model produced with the right `Content-Type`.
- **n8n's HTTP_Request tool** exposes its parameters as `properties: {}` (an empty schema). Some models (Llama 70B, Hermes 7B, depending on the prompt) refuse to call no-arg tools and reply in text. Configure the tool with explicit properties or supply enough context in the user message.

## Pairs well with

- **[js-vector-store](https://github.com/MauricioPerera/js-vector-store)** — zero-dependency vanilla JS vector store with `matryoshkaSearch`, IVF, and 4 quantisations (Float32 / Int8 / 3-bit polar / 1-bit). Combines naturally with the bridge's `dimensions` parameter on `/v1/embeddings`: the bridge truncates and L2-renormalises Matryoshka vectors at request time, the store does multi-stage search across the same dimensional slices. See [`examples/rag-with-js-vector-store.mjs`](examples/rag-with-js-vector-store.mjs) for an end-to-end runnable example.
- **[just-bash-data](https://github.com/MauricioPerera/just-bash-data)** — `db` (Mongo-style document store) and `vec` (vector similarity) commands for [`just-bash`](https://github.com/vercel-labs/just-bash) shell agents. Point its `vec` command at any embedding model the bridge serves to give an in-shell agent OpenAI-compatible RAG with Workers AI economics.

## Project layout

```
src/
├── index.ts             Hono router, CORS, bearer auth (constant-time), optional rate limit
├── ai-client.ts         Workers AI client — REST API or binding fallback
├── chat.ts              /v1/chat/completions (stream + non-stream + tools + vision + reasoning)
├── responses.ts         /v1/responses (stream + non-stream + tools + multi-turn + reasoning)
├── embeddings.ts        /v1/embeddings (with edge cache)
├── audio.ts             /v1/audio/transcriptions, /v1/audio/translations
├── speech.ts            /v1/audio/speech
├── images.ts            /v1/images/generations
├── moderations.ts       /v1/moderations (Llama Guard 3 → OpenAI categories)
├── models.ts            /v1/models, /v1/models/:id
├── mapping.ts           OpenAI → Workers AI alias table + vision/reasoning detection
├── tool-call-parser.ts  Streaming <tool_call>...</tool_call> parser (Hermes/Mistral)
├── image-inline.ts      Auto-fetch and base64-inline remote image URLs
└── types.ts             Shared types
```

## License

MIT.
