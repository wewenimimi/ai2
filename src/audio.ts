import type { Context } from "hono";
import { runAI } from "./ai-client";
import type { Env } from "./types";

const WHISPER_ALIASES: Record<string, string> = {
  "whisper-1": "@cf/openai/whisper",
  "whisper-large": "@cf/openai/whisper-large-v3-turbo",
  "whisper-large-v3": "@cf/openai/whisper-large-v3-turbo",
};

function resolveWhisperModel(requested: string | undefined): string {
  if (!requested) return "@cf/openai/whisper";
  if (requested.startsWith("@cf/")) return requested;
  return WHISPER_ALIASES[requested] ?? "@cf/openai/whisper";
}

// Convert ArrayBuffer to base64 without spilling >100KB strings on the stack.
function toBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function handleWhisper(
  c: Context<{ Bindings: Env }>,
  task: "transcribe" | "translate",
) {
  let form: FormData;
  try {
    form = await c.req.formData();
  } catch {
    return c.json(
      { error: { message: "Body must be multipart/form-data", type: "invalid_request_error" } },
      400,
    );
  }

  const file = form.get("file") as unknown as Blob | null;
  if (!file || typeof (file as Blob).arrayBuffer !== "function") {
    return c.json({ error: { message: "`file` field is required", type: "invalid_request_error" } }, 400);
  }

  const requestedModel = (form.get("model") as string | null) ?? undefined;
  const responseFormat = (form.get("response_format") as string | null) ?? "json";
  const language = (form.get("language") as string | null) ?? undefined;
  const promptHint = (form.get("prompt") as string | null) ?? undefined;
  const temperatureRaw = form.get("temperature") as string | null;
  const temperature = temperatureRaw != null ? parseFloat(temperatureRaw) : undefined;
  const granularitiesRaw = form.getAll("timestamp_granularities[]") as string[];
  const granularities =
    granularitiesRaw.length > 0
      ? granularitiesRaw
      : (form.get("timestamp_granularities") as string | null)?.split(",").filter(Boolean);

  // Whisper translate task only works on whisper-large-v3-turbo, not the
  // original whisper-1 model. Force the large variant when translating.
  const model =
    task === "translate"
      ? resolveWhisperModel(requestedModel?.includes("large") ? requestedModel : "whisper-large-v3")
      : resolveWhisperModel(requestedModel);

  const buffer = await file.arrayBuffer();

  // The two Whisper variants on Workers AI take different inputs: the original
  // model wants a byte array, the large-v3-turbo wants base64. Extra OpenAI
  // params (prompt, temperature, timestamp_granularities) are forwarded only
  // to the large model; the legacy one ignores them.
  const aiInput: Record<string, unknown> = model.includes("whisper-large-v3")
    ? {
        audio: toBase64(buffer),
        task,
        ...(language ? { source_lang: language } : {}),
        ...(task === "translate" ? { target_lang: "en" } : {}),
        ...(promptHint ? { initial_prompt: promptHint } : {}),
        ...(typeof temperature === "number" && !Number.isNaN(temperature) ? { temperature } : {}),
        ...(granularities && granularities.length ? { timestamp_granularities: granularities } : {}),
      }
    : { audio: [...new Uint8Array(buffer)] };

  let result: any;
  try {
    result = await runAI(c.env, model, aiInput);
  } catch (err) {
    return c.json(
      { error: { message: (err as Error).message ?? "Workers AI call failed", type: "upstream_error" } },
      502,
    );
  }

  const text: string = result?.text ?? result?.transcription ?? "";

  if (responseFormat === "text") {
    return new Response(text, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
  }
  if (responseFormat === "vtt" && typeof result?.vtt === "string") {
    return new Response(result.vtt, { headers: { "Content-Type": "text/vtt; charset=utf-8" } });
  }
  if (responseFormat === "verbose_json") {
    return c.json({
      task,
      language: language ?? "unknown",
      duration: result?.duration ?? 0,
      text,
      words: result?.words ?? [],
      segments: result?.segments ?? [],
    });
  }
  if (responseFormat === "srt") {
    // Best-effort SRT generation from word-level timestamps.
    const words: any[] = result?.words ?? [];
    const lines: string[] = [];
    let idx = 1;
    for (let i = 0; i < words.length; i += 10) {
      const chunk = words.slice(i, i + 10);
      if (chunk.length === 0) continue;
      const start = chunk[0].start ?? 0;
      const end = chunk[chunk.length - 1].end ?? start + 1;
      const fmt = (s: number) => {
        const h = Math.floor(s / 3600).toString().padStart(2, "0");
        const m = Math.floor((s % 3600) / 60).toString().padStart(2, "0");
        const sec = Math.floor(s % 60).toString().padStart(2, "0");
        const ms = Math.floor((s - Math.floor(s)) * 1000).toString().padStart(3, "0");
        return `${h}:${m}:${sec},${ms}`;
      };
      lines.push(`${idx++}\n${fmt(start)} --> ${fmt(end)}\n${chunk.map((w) => w.word ?? w.text ?? "").join("").trim()}\n`);
    }
    const srt = lines.join("\n") || `1\n00:00:00,000 --> 00:00:01,000\n${text}\n`;
    return new Response(srt, { headers: { "Content-Type": "application/x-subrip; charset=utf-8" } });
  }

  return c.json({ text });
}

export async function handleTranscriptions(c: Context<{ Bindings: Env }>) {
  return handleWhisper(c, "transcribe");
}

export async function handleTranslations(c: Context<{ Bindings: Env }>) {
  return handleWhisper(c, "translate");
}
