// Workers AI vision models accept only `data:` URIs for image inputs (not
// remote URLs). OpenAI clients can send either, so we transparently fetch
// any http/https URL the caller provides and inline it as base64.
//
// Failures are intentionally non-fatal: we leave the original URL in place
// and let the upstream return its own error so the caller sees the real
// reason (timeout, 404, oversized, …).
//
// Defenses against abuse (the bridge fetches arbitrary URLs the caller
// supplied, which is a textbook SSRF/DoS vector):
//   * blocked private / loopback / link-local / multicast hostnames so a
//     caller can't probe the Worker's internal network or cloud metadata
//     endpoints
//   * AbortController timeout caps how long any one fetch can stall
//   * pre-flight Content-Length check rejects oversized files before
//     downloading them
//   * post-flight byte cap as a backstop when the server lies about
//     Content-Length or omits it

const MAX_BYTES = 10 * 1024 * 1024; // 10 MB hard ceiling
const FETCH_TIMEOUT_MS = 10_000;

// Hostnames / IP literals we refuse to fetch. Covers loopback, RFC 1918
// private ranges, link-local + cloud metadata IPs, IPv6 loopback / ULA /
// link-local, and the "internal" / ".local" mDNS conventions.
function isBlockedHostname(hostname: string): boolean {
  const h = hostname.toLowerCase().replace(/^\[|]$/g, ""); // strip IPv6 brackets

  if (!h || h === "localhost" || h.endsWith(".localhost")) return true;
  if (h.endsWith(".local") || h.endsWith(".internal")) return true;
  if (h === "metadata.google.internal" || h === "metadata.azure.com") return true;

  // IPv4 literal? — block private / loopback / link-local / cloud metadata.
  const v4 = h.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (v4) {
    const [, a, b] = v4.map(Number) as [number, number, number, number, number];
    if (a === 10) return true;                       // 10.0.0.0/8
    if (a === 127) return true;                      // 127.0.0.0/8 loopback
    if (a === 0) return true;                        // 0.0.0.0/8
    if (a === 169 && b === 254) return true;         // 169.254/16 link-local + AWS/GCP metadata
    if (a === 172 && b >= 16 && b <= 31) return true; // 172.16/12
    if (a === 192 && b === 168) return true;         // 192.168/16
    if (a >= 224) return true;                       // multicast + reserved
    return false;
  }

  // IPv6 literal — block ::1, fc00::/7 (ULA), fe80::/10 (link-local), ::
  // The link-local range fe80::/10 corresponds to first-group prefixes
  // fe80..febf (the second nibble must be 8, 9, a or b). The previous
  // string-prefix check missed fe81..fe8f.
  if (h.includes(":")) {
    if (h === "::1" || h === "::") return true;
    if (h.startsWith("fc") || h.startsWith("fd")) return true;
    if (/^fe[89ab][0-9a-f]:/i.test(h)) return true;
    return false;
  }

  return false;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function fetchAsDataUri(url: string): Promise<string | null> {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
  if (isBlockedHostname(parsed.hostname)) return null;

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      redirect: "follow",
      signal: ctrl.signal,
      headers: {
        // Many image hosts (Wikipedia, GitHub user-content, …) reject
        // requests without a User-Agent.
        "User-Agent": "openai-workers-ai-bridge/0.1 (+https://github.com/MauricioPerera/openai-workers-ai-bridge)",
        Accept: "image/*,*/*;q=0.8",
      },
    });
    if (!res.ok) return null;

    // Trust the server's Content-Length when present so we reject big files
    // before ever buffering them. This is a soft check — some servers omit
    // the header or lie, so the post-flight byte check below stays.
    const cl = res.headers.get("content-length");
    if (cl) {
      const bytes = parseInt(cl, 10);
      if (Number.isFinite(bytes) && bytes > MAX_BYTES) return null;
    }

    const contentType = (res.headers.get("content-type") || "").split(";")[0]?.trim() || "image/jpeg";
    if (!contentType.startsWith("image/")) return null;

    const buf = await res.arrayBuffer();
    if (buf.byteLength > MAX_BYTES) return null;
    return `data:${contentType};base64,${bytesToBase64(new Uint8Array(buf))}`;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function extractUrl(part: any): string | null {
  if (!part) return null;
  // chat.completions: { type:"image_url", image_url:{url} | url }
  // Responses input: { type:"input_image", image_url:{url} | url } or top-level url
  const raw = part.image_url ?? part.url;
  if (typeof raw === "string") return raw;
  if (raw && typeof raw.url === "string") return raw.url;
  return null;
}

function isHttpUrl(s: string): boolean {
  return /^https?:\/\//i.test(s);
}

// Walk a chat.completions-style messages array and inline any remote image
// URLs as base64 data URIs in place. Returns the same array reference for
// convenience.
export async function inlineImageUrls(messages: any[]): Promise<any[]> {
  const jobs: Promise<void>[] = [];
  for (const m of messages) {
    if (!m || !Array.isArray(m.content)) continue;
    for (const part of m.content) {
      if (!part || part.type !== "image_url") continue;
      const url = extractUrl(part);
      if (!url || !isHttpUrl(url)) continue;
      jobs.push(
        fetchAsDataUri(url).then((dataUri) => {
          if (!dataUri) return;
          part.image_url = { url: dataUri };
        }),
      );
    }
  }
  if (jobs.length) await Promise.all(jobs);
  return messages;
}

// Exported for unit tests.
export const __testing = { isBlockedHostname };
