import type { Env } from "./types";

// Wrapper around Workers AI. Uses the REST API when CLOUDFLARE_TOKEN +
// CLOUDFLARE_ACCOUNT_ID secrets are configured (recommended — the binding
// has been observed to behave inconsistently for tool calling depending on
// the calling context). Falls back to the AI binding otherwise.
export interface AIRunOptions {
  stream?: boolean;
}

export async function runAI(
  env: Env & { CLOUDFLARE_TOKEN?: string; CLOUDFLARE_ACCOUNT_ID?: string },
  model: string,
  input: Record<string, unknown>,
  opts: AIRunOptions = {},
): Promise<unknown> {
  const token = env.CLOUDFLARE_TOKEN;
  const acct = env.CLOUDFLARE_ACCOUNT_ID;

  if (token && acct) {
    const res = await fetch(`https://api.cloudflare.com/client/v4/accounts/${acct}/ai/run/${model}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(input),
    });

    if (opts.stream) {
      // For streaming, the REST API returns SSE directly in the body.
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Workers AI ${res.status}: ${errText.slice(0, 300)}`);
      }
      if (!res.body) throw new Error("Workers AI: empty stream body");
      return res.body;
    }

    const json = await res.json<any>();
    if (!res.ok || json?.success === false) {
      const errs = Array.isArray(json?.errors) ? json.errors.map((e: any) => `${e.code}: ${e.message}`).join("; ") : "";
      throw new Error(`Workers AI ${res.status}${errs ? ` — ${errs}` : ""}`);
    }
    return json?.result ?? json;
  }

  return env.AI.run(model as keyof AiModels, input as never);
}
