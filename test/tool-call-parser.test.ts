import { describe, expect, it } from "vitest";
import { createToolCallStreamParser, type ParsedToolCall } from "../src/tool-call-parser";

function run(chunks: string[]) {
  const text: string[] = [];
  const calls: ParsedToolCall[] = [];
  const p = createToolCallStreamParser({
    onText: (s) => text.push(s),
    onToolCall: (c) => calls.push(c),
  });
  for (const ch of chunks) p.feed(ch);
  p.end();
  return { text: text.join(""), calls };
}

describe("tool-call stream parser", () => {
  it("passes plain text through unchanged", () => {
    const r = run(["hello ", "world"]);
    expect(r.text).toBe("hello world");
    expect(r.calls).toHaveLength(0);
  });

  it("extracts a single tool_call block (whole-buffer)", () => {
    const r = run([`<tool_call>\n{"name":"f","arguments":{"x":1}}\n</tool_call>`]);
    expect(r.text).toBe("");
    expect(r.calls).toHaveLength(1);
    expect(r.calls[0].name).toBe("f");
    expect(JSON.parse(r.calls[0].arguments)).toEqual({ x: 1 });
  });

  it("handles open tag split across chunks", () => {
    const r = run(["before<", "tool", "_call>", `{"name":"f","arguments":{}}`, "</tool_call>after"]);
    expect(r.text).toBe("beforeafter");
    expect(r.calls).toHaveLength(1);
    expect(r.calls[0].name).toBe("f");
  });

  it("handles close tag split across chunks", () => {
    const r = run([`<tool_call>{"name":"f","arguments":{}}</to`, "ol_", "call>tail"]);
    expect(r.text).toBe("tail");
    expect(r.calls).toHaveLength(1);
  });

  it("accepts python-dict / single-quoted JSON inside the tag", () => {
    const r = run([`<tool_call>\n{'arguments': {}, 'name': 'HTTP_Request'}\n</tool_call>`]);
    expect(r.calls).toHaveLength(1);
    expect(r.calls[0].name).toBe("HTTP_Request");
    expect(r.calls[0].arguments).toBe("{}");
  });

  it("emits text and tool_call separately when both appear", () => {
    const r = run(["thinking... ", `<tool_call>{"name":"f","arguments":{"q":"x"}}</tool_call>`, " done"]);
    expect(r.text).toBe("thinking...  done");
    expect(r.calls).toHaveLength(1);
  });

  it("supports multiple tool_calls in one stream", () => {
    const r = run([
      `<tool_call>{"name":"a","arguments":{}}</tool_call>`,
      `<tool_call>{"name":"b","arguments":{"k":1}}</tool_call>`,
    ]);
    expect(r.calls).toHaveLength(2);
    expect(r.calls[0].name).toBe("a");
    expect(r.calls[1].name).toBe("b");
  });

  it("does not leak a partial open-tag prefix as text", () => {
    // Chunk that ends mid-tag should hold those bytes back, not emit them.
    const text: string[] = [];
    const p = createToolCallStreamParser({ onText: (s) => text.push(s), onToolCall: () => {} });
    p.feed("hello <to");
    expect(text.join("")).toBe("hello "); // "<to" held back
  });

  it("flushes a non-matching '<' as text when more arrives", () => {
    const r = run(["a<b", " c"]);
    expect(r.text).toBe("a<b c");
  });

  it("recovers if stream ends mid-tag with parseable JSON", () => {
    const r = run([`<tool_call>{"name":"f","arguments":{}}`]);
    expect(r.calls).toHaveLength(1);
    expect(r.calls[0].name).toBe("f");
  });
});
