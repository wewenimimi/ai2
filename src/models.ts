import type { Context } from "hono";
import { ADVERTISED_MODELS } from "./mapping";
import type { Env } from "./types";

export function handleListModels(c: Context<{ Bindings: Env }>) {
  const created = Math.floor(Date.now() / 1000);
  return c.json({
    object: "list",
    data: ADVERTISED_MODELS.map((id) => ({
      id,
      object: "model",
      created,
      owned_by: id.startsWith("@cf/") ? id.split("/")[1] ?? "cloudflare" : "openai-compat",
    })),
  });
}

export function handleRetrieveModel(c: Context<{ Bindings: Env }>) {
  const id = c.req.param("id");
  if (!id || !ADVERTISED_MODELS.includes(id)) {
    return c.json({ error: { message: `Model '${id}' not found`, type: "invalid_request_error" } }, 404);
  }
  return c.json({
    id,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: id.startsWith("@cf/") ? id.split("/")[1] ?? "cloudflare" : "openai-compat",
  });
}
