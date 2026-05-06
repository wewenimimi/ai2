// End-to-end RAG demo: openai-workers-ai-bridge + js-vector-store.
//
// Embeds a small corpus through the bridge (using EmbeddingGemma + the
// OpenAI `dimensions` parameter for Matryoshka truncation), stores the
// vectors quantised to int8 (~4× compression), and runs a multi-stage
// matryoshkaSearch over the result.
//
// Usage:
//   npm install openai js-vector-store
//   BRIDGE_URL=https://openai-workers-ai-bridge.<your-subdomain>.workers.dev/v1 \
//   API_KEY=sk-cfwai-... \
//   node examples/rag-with-js-vector-store.mjs

import OpenAI from "openai";
import { VectorStore, QuantizedStore, MemoryStorageAdapter } from "js-vector-store";

const BRIDGE_URL = process.env.BRIDGE_URL ?? "https://openai-workers-ai-bridge.example.workers.dev/v1";
const API_KEY = process.env.API_KEY ?? "sk-replace-me";

const client = new OpenAI({ apiKey: API_KEY, baseURL: BRIDGE_URL });

const corpus = [
  { id: "doc-1", text: "Cloudflare Workers AI runs LLMs on the edge in 300+ cities.", category: "infra" },
  { id: "doc-2", text: "EmbeddingGemma is a 300M-parameter multilingual embedding model trained with Matryoshka representation learning.", category: "models" },
  { id: "doc-3", text: "Matryoshka embeddings can be safely truncated to a prefix while keeping their semantic meaning.", category: "models" },
  { id: "doc-4", text: "BGE-small produces 384-dim vectors, which is great for tight storage budgets.", category: "models" },
  { id: "doc-5", text: "Hono is a tiny, fast HTTP framework that fits Cloudflare Workers naturally.", category: "infra" },
  { id: "doc-6", text: "PolarQuant compresses embeddings to ~3 bits per dimension with full recall.", category: "indexing" },
  { id: "doc-7", text: "IVF clusters vectors with k-means and only scans the closest centroids at query time.", category: "indexing" },
  { id: "doc-8", text: "CDN caching makes static assets faster but doesn't help with model inference.", category: "infra" },
];

async function embedAll(texts) {
  const res = await client.embeddings.create({
    model: "embeddinggemma",
    input: texts,
    // Truncate to 256 dims via the bridge's Matryoshka path. The vectors
    // come back already L2-normalised so cosine similarity Just Works.
    dimensions: 256,
  });
  return res.data.map((d) => d.embedding);
}

async function main() {
  console.log(`Bridge: ${BRIDGE_URL}`);
  console.log(`Indexing ${corpus.length} documents (dim=256, int8 quantised)...`);

  const t0 = Date.now();
  const vectors = await embedAll(corpus.map((d) => d.text));
  console.log(`  embeddings ready in ${Date.now() - t0}ms`);

  // Int8 quantisation: ~4× smaller than Float32 and search cost stays linear.
  // For million-vector workloads, swap in PolarQuantizedStore (3-bit, ~21×)
  // or BinaryQuantizedStore (1-bit, ~32×) — same API.
  const store = new QuantizedStore(new MemoryStorageAdapter(), 256);
  for (let i = 0; i < corpus.length; i++) {
    store.set("docs", corpus[i].id, vectors[i], { text: corpus[i].text, category: corpus[i].category });
  }
  store.flush();

  // Query.
  const query = "How do I store embeddings on a tight memory budget?";
  console.log(`\nQuery: "${query}"`);
  const [qVec] = await embedAll([query]);

  // Multi-stage matryoshka search: fast filter at dim=128, refine at 256.
  // For larger corpora you'd add intermediate stages (128 → 256 → 512 → full).
  const results = store.matryoshkaSearch("docs", qVec, 3, [128, 256], "cosine");

  console.log(`\nTop ${results.length}:`);
  for (const r of results) {
    console.log(`  ${r.score.toFixed(4)}  ${r.id}  [${r.metadata.category}]  ${r.metadata.text}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
