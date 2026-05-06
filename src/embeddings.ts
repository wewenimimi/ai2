import type { Context } from "hono";
import { runAI } from "./ai-client";
import { getMatryoshkaInfo, resolveEmbeddingModel } from "./mapping";
import type { EmbeddingsRequest, Env } from "./types";

// Matryoshka truncation: take the first `dim` components and re-normalize so
// the result has unit L2 norm. EmbeddingGemma (and other Matryoshka-trained
// models) are designed so that this prefix is still a valid embedding in the
// same semantic space.
function truncateAndRenormalize(vec: number[], dim: number): number[] {
  const head = vec.slice(0, dim);
  let sumSq = 0;
  for (let i = 0; i < head.length; i++) sumSq += head[i] * head[i];
  if (sumSq === 0) return head;
  const norm = Math.sqrt(sumSq);
  for (let i = 0; i < head.length; i++) head[i] = head[i] / norm;
  return head;
}

// OpenAI's official Node SDK defaults to `encoding_format: "base64"` and
// expects the server to return `embedding` as a base64-encoded little-endian
// Float32 buffer rather than a JSON number array. If we don't honour that,
// the SDK happily decodes the literal "[-0.305, ..." JSON as base64 and
// produces garbage. Match the spec.
function vectorToBase64(vec: number[]): string {
  const buf = new ArrayBuffer(vec.length * 4);
  const view = new Float32Array(buf);
  for (let i = 0; i < vec.length; i++) view[i] = vec[i];
  const bytes = new Uint8Array(buf);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

// Embeddings are deterministic for a given (model, text, dim) tuple. Cache
// them in the Worker's edge cache so repeated calls (RAG pipelines, n8n
// loops) don't burn neurons on identical inputs. The cache key is a SHA-256
// of `${model}@${dim}\0${text}` (or just `${model}\0${text}` when no
// truncation is requested) — different `dimensions` values land in
// different buckets so a 256-dim Matryoshka request and a full-dim request
// don't poison each other.
const CACHE_TTL_SECONDS = 60 * 60 * 24 * 7; // one week

async function vectorCacheKey(model: string, text: string): Promise<string> {
  const data = new TextEncoder().encode(model + "\0" + text);
  const buf = await crypto.subtle.digest("SHA-256", data);
  const bytes = new Uint8Array(buf);
  let hex = "";
  for (let i = 0; i < bytes.length; i++) hex += bytes[i].toString(16).padStart(2, "0");
  // Cache API requires an http(s) URL key.
  return `https://cache.local/embeddings/${model.replace(/[^a-z0-9]/gi, "_")}/${hex}`;
}

async function readCached(model: string, text: string): Promise<number[] | null> {
  const cache = (caches as any).default as Cache | undefined;
  if (!cache) return null;
  try {
    const key = await vectorCacheKey(model, text);
    const hit = await cache.match(new Request(key));
    if (!hit) return null;
    return (await hit.json()) as number[];
  } catch {
    return null;
  }
}

async function writeCached(model: string, text: string, vector: number[]): Promise<void> {
  const cache = (caches as any).default as Cache | undefined;
  if (!cache) return;
  try {
    const key = await vectorCacheKey(model, text);
    const res = new Response(JSON.stringify(vector), {
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": `public, max-age=${CACHE_TTL_SECONDS}, s-maxage=${CACHE_TTL_SECONDS}`,
      },
    });
    await cache.put(new Request(key), res);
  } catch {
    // best-effort
  }
}

export async function handleEmbeddings(c: Context<{ Bindings: Env }>) {
  let body: EmbeddingsRequest;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid JSON body", type: "invalid_request_error" } }, 400);
  }

  if (!body.input) {
    return c.json({ error: { message: "`input` is required", type: "invalid_request_error" } }, 400);
  }

  const inputs = Array.isArray(body.input) ? body.input : [body.input];
  if (inputs.some((s) => typeof s !== "string")) {
    return c.json({ error: { message: "`input` must be string or string[]", type: "invalid_request_error" } }, 400);
  }
  // Workers AI embedding endpoints accept large batches but the practical
  // cap before the request times out is around a few hundred inputs. Reject
  // explicitly so callers get a clean error rather than a truncated result.
  const BATCH_CAP = 256;
  if (inputs.length > BATCH_CAP) {
    return c.json(
      {
        error: {
          message: `\`input\` array length ${inputs.length} exceeds the per-request cap of ${BATCH_CAP}. Split into multiple calls.`,
          type: "invalid_request_error",
        },
      },
      400,
    );
  }

  const model = resolveEmbeddingModel(body.model, c.env.DEFAULT_EMBEDDING_MODEL ?? "@cf/baai/bge-m3");

  // Validate the OpenAI-compat `dimensions` parameter against this model's
  // Matryoshka tiers. Only models trained with Matryoshka representation
  // learning can be safely truncated; anything else gets a 400.
  const matryoshka = getMatryoshkaInfo(model);
  let targetDim: number | null = null;
  if (typeof body.dimensions === "number") {
    if (!matryoshka) {
      return c.json(
        {
          error: {
            message: `Model ${model} is not a Matryoshka embedding model and does not support the \`dimensions\` parameter. Drop the field or pick a Matryoshka model (embeddinggemma-300m, bge-m3, qwen3-embedding-0.6b).`,
            type: "invalid_request_error",
          },
        },
        400,
      );
    }
    if (body.dimensions < 1 || body.dimensions > matryoshka.nativeDim) {
      return c.json(
        {
          error: {
            message: `\`dimensions\` must be between 1 and ${matryoshka.nativeDim} for ${model}. Trained tiers: ${matryoshka.tiers.join(", ")}.`,
            type: "invalid_request_error",
          },
        },
        400,
      );
    }
    targetDim = body.dimensions;
  }

  // Cache the *truncated* output, not the native one. Different `dimensions`
  // values produce different vectors, so the cache key must distinguish them.
  const cacheModelKey = targetDim ? `${model}@${targetDim}` : model;

  // Check cache for each input; collect misses for a single upstream call.
  const cached = await Promise.all(inputs.map((t) => readCached(cacheModelKey, t)));
  const misses: { idx: number; text: string }[] = [];
  cached.forEach((v, i) => {
    if (!v) misses.push({ idx: i, text: inputs[i] });
  });

  let upstreamResult: any = null;
  if (misses.length > 0) {
    try {
      upstreamResult = await runAI(c.env, model, { text: misses.map((m) => m.text) });
    } catch (err) {
      return c.json(
        { error: { message: (err as Error).message ?? "Workers AI call failed", type: "upstream_error" } },
        502,
      );
    }

    const fresh: number[][] = upstreamResult?.data ?? [];
    if (fresh.length !== misses.length) {
      return c.json(
        { error: { message: "Upstream returned fewer vectors than requested", type: "upstream_error" } },
        502,
      );
    }
    // Stitch fresh vectors back into the cached array and prime the cache.
    // Apply Matryoshka truncation here so the cached value already has the
    // right shape and a second hit doesn't re-truncate.
    await Promise.all(
      misses.map(async (m, i) => {
        const vec = targetDim ? truncateAndRenormalize(fresh[i], targetDim) : fresh[i];
        cached[m.idx] = vec;
        await writeCached(cacheModelKey, m.text, vec);
      }),
    );
  }

  // Default to base64 to match OpenAI's wire format. Clients that send
  // `encoding_format: "float"` get a JSON number array; everyone else gets
  // base64 (which the OpenAI Node SDK decodes back to Float32Array).
  const wantFloat = body.encoding_format === "float";

  return c.json({
    object: "list",
    data: cached.map((embedding, index) => ({
      object: "embedding",
      index,
      embedding: wantFloat ? (embedding ?? []) : vectorToBase64(embedding ?? []),
    })),
    model: body.model || model,
    usage: {
      prompt_tokens: upstreamResult?.usage?.prompt_tokens ?? 0,
      total_tokens: upstreamResult?.usage?.total_tokens ?? 0,
    },
  });
}
