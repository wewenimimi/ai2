import type { Context } from "hono";
import { runAI } from "./ai-client";
import type { Env } from "./types";

interface ModerationsRequest {
  model?: string;
  input: string | string[];
}

// Llama Guard 3 hazard taxonomy → OpenAI moderation categories.
// https://huggingface.co/meta-llama/Llama-Guard-3-8B
const LG3_TO_OPENAI: Record<string, string[]> = {
  S1: ["violence"],                                 // Violent Crimes
  S2: ["harassment"],                               // Non-Violent Crimes
  S3: ["sexual"],                                   // Sex-Related Crimes
  S4: ["sexual", "sexual/minors"],                  // Child Sexual Exploitation
  S5: ["harassment"],                               // Defamation
  S6: [],                                           // Specialized Advice — no OpenAI equivalent
  S7: [],                                           // Privacy — no OpenAI equivalent
  S8: [],                                           // Intellectual Property — no OpenAI equivalent
  S9: ["violence", "violence/graphic"],             // Indiscriminate Weapons
  S10: ["hate"],                                    // Hate
  S11: ["self-harm", "self-harm/intent", "self-harm/instructions"], // Suicide & Self-Harm
  S12: ["sexual"],                                  // Sexual Content
  S13: [],                                          // Elections — no OpenAI equivalent
  S14: [],                                          // Code Interpreter Abuse — no OpenAI equivalent
};

const OPENAI_CATEGORIES = [
  "sexual",
  "hate",
  "harassment",
  "self-harm",
  "sexual/minors",
  "hate/threatening",
  "violence/graphic",
  "self-harm/intent",
  "self-harm/instructions",
  "harassment/threatening",
  "violence",
] as const;

interface ModerationResult {
  flagged: boolean;
  categories: Record<string, boolean>;
  category_scores: Record<string, number>;
}

function classifyOne(raw: string): ModerationResult {
  const text = raw.trim().toLowerCase();
  const flagged = text.startsWith("unsafe");

  const categories: Record<string, boolean> = {};
  const category_scores: Record<string, number> = {};
  for (const c of OPENAI_CATEGORIES) {
    categories[c] = false;
    category_scores[c] = 0;
  }

  if (!flagged) return { flagged, categories, category_scores };

  // Extract S-codes (S1, S9, S10, ...)
  const codes = raw.match(/\bS\d{1,2}\b/g) ?? [];
  for (const code of codes) {
    const mapped = LG3_TO_OPENAI[code] ?? [];
    for (const cat of mapped) {
      categories[cat] = true;
      category_scores[cat] = 1.0;
    }
  }
  return { flagged, categories, category_scores };
}

export async function handleModerations(
  c: Context<{ Bindings: Env & { CLOUDFLARE_TOKEN?: string; CLOUDFLARE_ACCOUNT_ID?: string } }>,
) {
  let body: ModerationsRequest;
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

  const model = body.model?.startsWith("@cf/")
    ? body.model
    : "@cf/meta/llama-guard-3-8b";

  // Moderation must fail closed: returning {flagged:false} on upstream
  // error would tell the caller "this content is safe" when in fact we
  // never inspected it. We dispatch all inputs in parallel via
  // Promise.allSettled (so batch latency stays at max(inputs), not sum)
  // and surface the first failure as a 502 — fail-closed semantics
  // preserved without the sequential-loop tax.
  const settled = await Promise.allSettled(
    inputs.map((text) =>
      runAI(c.env, model, { messages: [{ role: "user", content: text }] }),
    ),
  );
  const firstRejection = settled.find((s) => s.status === "rejected") as
    | PromiseRejectedResult
    | undefined;
  if (firstRejection) {
    console.error("[/v1/moderations] upstream error:", (firstRejection.reason as Error)?.message ?? firstRejection.reason);
    return c.json(
      {
        error: {
          message:
            "Moderation upstream failed. The bridge intentionally fails closed instead of returning `flagged:false`. Retry, switch to a different moderation model, or apply your own block policy.",
          type: "upstream_error",
          code: "moderation_unavailable",
        },
      },
      502,
    );
  }
  const results: ModerationResult[] = settled.map((s) => {
    const r = (s as PromiseFulfilledResult<any>).value;
    const raw: string = r?.response ?? r?.choices?.[0]?.message?.content ?? "";
    return classifyOne(raw);
  });

  return c.json({
    id: "modr_" + crypto.randomUUID().replace(/-/g, "").slice(0, 24),
    model: body.model ?? model,
    results,
  });
}
