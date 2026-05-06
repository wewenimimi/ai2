import type { Context } from "hono";
import type { Env } from "./types";

interface SpeechRequest {
  model?: string;
  input: string;
  voice?: string;
  response_format?: "mp3" | "opus" | "aac" | "flac" | "wav" | "pcm";
  speed?: number;
  language?: string; // non-OpenAI extension, used by melotts
}

const TTS_ALIASES: Record<string, string> = {
  "tts-1": "@cf/myshell-ai/melotts",
  "tts-1-hd": "@cf/myshell-ai/melotts",
};

// Workers AI's @cf/deepgram/aura-1 voice catalogue.
const AURA_VOICES = new Set([
  "angus", "asteria", "arcas", "athena", "helios", "hera",
  "luna", "orion", "orpheus", "perseus", "stella", "zeus",
]);

// Soft mapping from OpenAI voice names to Aura speakers — used when the
// chosen model is aura-1 and the caller sends an OpenAI-style voice.
const OPENAI_TO_AURA_VOICE: Record<string, string> = {
  alloy: "angus",
  echo: "orion",
  fable: "perseus",
  onyx: "zeus",
  nova: "luna",
  shimmer: "asteria",
};

function resolveTTSModel(requested: string | undefined): string {
  if (!requested) return "@cf/myshell-ai/melotts";
  if (requested.startsWith("@cf/")) return requested;
  return TTS_ALIASES[requested] ?? "@cf/myshell-ai/melotts";
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// Aura returns numbered byte-array keys when JSON-serialized; the binding
// hands them back as a Response/ReadableStream/Uint8Array depending on
// transport. Normalize to Uint8Array.
function coerceToBytes(result: unknown): Uint8Array | null {
  if (result instanceof Uint8Array) return result;
  if (result instanceof ArrayBuffer) return new Uint8Array(result);
  if (typeof result === "string") return base64ToBytes(result);
  if (result && typeof result === "object") {
    const r = result as any;
    if (typeof r.audio === "string") return base64ToBytes(r.audio);
    if (r.audio instanceof Uint8Array) return r.audio;
    if (r.audio instanceof ArrayBuffer) return new Uint8Array(r.audio);
    // Numbered-keys array shape (REST API serialization of binary bytes)
    const keys = Object.keys(r).filter((k) => /^\d+$/.test(k));
    if (keys.length > 16) {
      const arr = new Uint8Array(keys.length);
      for (let i = 0; i < keys.length; i++) arr[i] = r[String(i)];
      return arr;
    }
  }
  return null;
}

export async function handleSpeech(
  c: Context<{ Bindings: Env & { CLOUDFLARE_TOKEN?: string; CLOUDFLARE_ACCOUNT_ID?: string } }>,
) {
  let body: SpeechRequest;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid JSON body", type: "invalid_request_error" } }, 400);
  }
  if (!body.input || typeof body.input !== "string") {
    return c.json({ error: { message: "`input` is required (string)", type: "invalid_request_error" } }, 400);
  }

  const model = resolveTTSModel(body.model);
  const isAura = model.includes("aura");
  const isMelo = model.includes("melotts");

  const aiInput: Record<string, unknown> = {};
  if (isMelo) {
    aiInput.prompt = body.input;
    // melotts uses ISO language codes. Caller may pass `language` directly,
    // otherwise default to English. OpenAI voice names are ignored — they
    // don't map onto melotts languages.
    aiInput.lang = body.language ?? "en";
  } else if (isAura) {
    aiInput.text = body.input;
    const voice = body.voice;
    if (voice && AURA_VOICES.has(voice)) aiInput.speaker = voice;
    else if (voice && OPENAI_TO_AURA_VOICE[voice]) aiInput.speaker = OPENAI_TO_AURA_VOICE[voice];
    else aiInput.speaker = "angus";
  } else {
    // Unknown TTS model — pass the request through as-is, common keys included.
    aiInput.text = body.input;
    aiInput.prompt = body.input;
    if (body.voice) aiInput.speaker = body.voice;
    if (body.language) aiInput.lang = body.language;
  }

  // Use the REST API directly (binding doesn't expose binary streams cleanly
  // and TTS responses are binary). Falls back to env.AI.run for envs without
  // a token configured.
  const token = c.env.CLOUDFLARE_TOKEN;
  const acct = c.env.CLOUDFLARE_ACCOUNT_ID;

  let bytes: Uint8Array | null = null;
  let upstreamCT: string | null = null;

  if (token && acct) {
    const res = await fetch(`https://api.cloudflare.com/client/v4/accounts/${acct}/ai/run/${model}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify(aiInput),
    });
    upstreamCT = res.headers.get("content-type");
    if (!res.ok) {
      const errBody = await res.text();
      return c.json(
        { error: { message: `Workers AI ${res.status}: ${errBody.slice(0, 400)}`, type: "upstream_error" } },
        502,
      );
    }
    if (upstreamCT?.startsWith("audio/")) {
      bytes = new Uint8Array(await res.arrayBuffer());
    } else {
      const json: any = await res.json();
      const result = json?.result ?? json;
      bytes = coerceToBytes(result);
    }
  } else {
    try {
      const result = await c.env.AI.run(model as keyof AiModels, aiInput as never);
      bytes = coerceToBytes(result);
    } catch (err) {
      return c.json(
        { error: { message: (err as Error).message ?? "Workers AI call failed", type: "upstream_error" } },
        502,
      );
    }
  }

  if (!bytes || bytes.length === 0) {
    return c.json(
      { error: { message: "TTS upstream returned no audio", type: "upstream_error" } },
      502,
    );
  }

  // Decide what Content-Type to advertise. melotts emits WAV; aura emits MP3.
  // We don't transcode, so the actual format follows the upstream — we just
  // label it correctly. If the caller asked for `response_format` we honour
  // it as the file extension hint but can't actually convert codecs here.
  let contentType = upstreamCT && upstreamCT.startsWith("audio/") ? upstreamCT : null;
  if (!contentType) {
    if (isAura) contentType = "audio/mpeg";
    else contentType = "audio/wav"; // melotts default
  }

  return new Response(bytes, {
    headers: {
      "Content-Type": contentType,
      "Cache-Control": "no-store",
    },
  });
}
