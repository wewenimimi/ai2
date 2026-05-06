import { cloudflareTest } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.toml" },
      miniflare: {
        bindings: {
          DEFAULT_CHAT_MODEL: "@cf/meta/llama-3.1-8b-instruct",
          DEFAULT_EMBEDDING_MODEL: "@cf/baai/bge-m3",
        },
      },
    }),
  ],
});
