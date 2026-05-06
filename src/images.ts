import type { Context } from "hono";
import type { Env } from "./types";

interface ImagesRequest {
  model?: string;
  prompt: string;
  n?: number;
  size?: string; // "1024x1024" / "512x512" / "1024x1792" / ...
  quality?: "standard" | "hd";
  response_format?: "url" | "b64_json";
  style?: "vivid" | "natural";
  // pass-through extensions for native models
  steps?: number;
  num_steps?: number;
  guidance?: number;
  seed?: number;
  negative_prompt?: string;
}

const IMAGE_ALIASES: Record<string, string> = {
  "dall-e-3": "@cf/black-forest-labs/flux-1-schnell",
  "dall-e-2": "@cf/black-forest-labs/flux-1-schnell",
  "flux": "@cf/black-forest-labs/flux-1-schnell",
  "flux-schnell": "@cf/black-forest-labs/flux-1-schnell",
  "sdxl": "@cf/bytedance/stable-diffusion-xl-lightning",
  "sdxl-lightning": "@cf/bytedance/stable-diffusion-xl-lightning",
  "dreamshaper": "@cf/lykon/dreamshaper-8-lcm",
};

function resolveImageModel(requested: string | undefined): string {
  if (!requested) return "@cf/black-forest-labs/flux-1-schnell";
  if (requested.startsWith("@cf/")) return requested;
  return IMAGE_ALIASES[requested] ?? "@cf/black-forest-labs/flux-1-schnell";
}

function parseSize(size: string | undefined): { width?: number; height?: number } {
  if (!size) return {};
  const m = size.match(/^(\d+)x(\d+)$/);
  if (!m) return {};
  return { width: parseInt(m[1], 10), height: parseInt(m[2], 10) };
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

// Detect image MIME type from the first few bytes of a base64-decoded
// payload. Avoids hardcoding `image/jpeg` for everything — different models
// hand back different formats and a future Workers AI model could switch.
function detectImageMime(b64: string): string {
  // Decode just enough bytes to inspect the magic. atob is fine for this:
  // we only need ~12 chars (~9 bytes) which costs nothing.
  let head = "";
  try { head = atob(b64.slice(0, 16)); } catch { return "image/jpeg"; }
  const b = (i: number) => head.charCodeAt(i) & 0xff;
  // JPEG: FF D8 FF
  if (b(0) === 0xff && b(1) === 0xd8 && b(2) === 0xff) return "image/jpeg";
  // PNG: 89 50 4E 47 0D 0A 1A 0A
  if (b(0) === 0x89 && b(1) === 0x50 && b(2) === 0x4e && b(3) === 0x47) return "image/png";
  // GIF: 47 49 46 38 (GIF8)
  if (b(0) === 0x47 && b(1) === 0x49 && b(2) === 0x46 && b(3) === 0x38) return "image/gif";
  // WEBP: 52 49 46 46 ?? ?? ?? ?? 57 45 42 50  (RIFF....WEBP). Other RIFF
  // formats (WAVE / AVI / RMI) start the same way but never match WEBP at
  // bytes 8-11, so we check all four to avoid a theoretical false positive.
  if (
    b(0) === 0x52 && b(1) === 0x49 && b(2) === 0x46 && b(3) === 0x46 &&
    b(8) === 0x57 && b(9) === 0x45 && b(10) === 0x42 && b(11) === 0x50
  ) return "image/webp";
  return "image/jpeg";
}

// Workers AI image-gen models hand back results in two shapes:
//   - { image: base64-string }  — Flux schnell
//   - raw binary bytes          — SDXL Lightning, Dreamshaper
// Normalize to a base64 string ready for OpenAI's b64_json field.
function coerceToBase64(result: unknown): string | null {
  if (typeof result === "string") {
    // Heuristic: looks like base64?
    if (/^[A-Za-z0-9+/=]+$/.test(result.slice(0, 64))) return result;
    // Otherwise treat as binary bytes encoded as latin-1 string.
    const bytes = new Uint8Array(result.length);
    for (let i = 0; i < result.length; i++) bytes[i] = result.charCodeAt(i) & 0xff;
    return bytesToBase64(bytes);
  }
  if (result instanceof Uint8Array) return bytesToBase64(result);
  if (result instanceof ArrayBuffer) return bytesToBase64(new Uint8Array(result));
  if (result && typeof result === "object") {
    const r = result as any;
    if (typeof r.image === "string") return r.image;
    if (r.image instanceof Uint8Array) return bytesToBase64(r.image);
  }
  return null;
}

export async function handleImages(
  c: Context<{ Bindings: Env & { CLOUDFLARE_TOKEN?: string; CLOUDFLARE_ACCOUNT_ID?: string } }>,
) {
  let body: ImagesRequest;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid JSON body", type: "invalid_request_error" } }, 400);
  }
  if (!body.prompt || typeof body.prompt !== "string") {
    return c.json({ error: { message: "`prompt` is required (string)", type: "invalid_request_error" } }, 400);
  }

  const model = resolveImageModel(body.model);
  const isFlux = model.includes("flux");
  const { width, height } = parseSize(body.size);

  const aiInput: Record<string, unknown> = { prompt: body.prompt };
  if (isFlux) {
    aiInput.steps = body.steps ?? body.num_steps ?? 4;
  } else {
    if (typeof body.steps === "number") aiInput.num_steps = body.steps;
    if (typeof body.num_steps === "number") aiInput.num_steps = body.num_steps;
    if (typeof body.guidance === "number") aiInput.guidance = body.guidance;
    if (width) aiInput.width = width;
    if (height) aiInput.height = height;
    if (body.negative_prompt) aiInput.negative_prompt = body.negative_prompt;
  }
  if (typeof body.seed === "number") aiInput.seed = body.seed;

  const token = c.env.CLOUDFLARE_TOKEN;
  const acct = c.env.CLOUDFLARE_ACCOUNT_ID;
  let b64: string | null = null;

  if (token && acct) {
    const res = await fetch(`https://api.cloudflare.com/client/v4/accounts/${acct}/ai/run/${model}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify(aiInput),
    });
    const ct = res.headers.get("content-type") ?? "";
    if (!res.ok) {
      const text = await res.text();
      return c.json(
        { error: { message: `Workers AI ${res.status}: ${text.slice(0, 400)}`, type: "upstream_error" } },
        502,
      );
    }
    if (ct.startsWith("image/") || ct.includes("octet-stream")) {
      b64 = bytesToBase64(new Uint8Array(await res.arrayBuffer()));
    } else {
      const json: any = await res.json();
      b64 = coerceToBase64(json?.result ?? json);
    }
  } else {
    try {
      const result = await c.env.AI.run(model as keyof AiModels, aiInput as never);
      b64 = coerceToBase64(result);
    } catch (err) {
      return c.json(
        { error: { message: (err as Error).message ?? "Workers AI call failed", type: "upstream_error" } },
        502,
      );
    }
  }

  if (!b64) {
    return c.json(
      { error: { message: "Image upstream returned no data", type: "upstream_error" } },
      502,
    );
  }

  // OpenAI default response_format is "url"; we don't run hosted storage so
  // we surface a data: URL instead. b64_json is supported as-is. Detect
  // the actual image MIME from magic bytes so the data URL labels e.g.
  // PNG bytes as image/png and not image/jpeg.
  const wantUrl = body.response_format !== "b64_json";
  const mime = detectImageMime(b64);
  const item = wantUrl
    ? { url: `data:${mime};base64,${b64}`, revised_prompt: body.prompt }
    : { b64_json: b64, revised_prompt: body.prompt };

  return c.json({
    created: Math.floor(Date.now() / 1000),
    data: [item],
  });
}
