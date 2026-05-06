import { describe, expect, it } from "vitest";
import { __testing } from "../src/image-inline";

const { isBlockedHostname } = __testing;

describe("image-inline SSRF blocklist", () => {
  it("blocks loopback IPv4", () => {
    expect(isBlockedHostname("127.0.0.1")).toBe(true);
    expect(isBlockedHostname("127.255.255.255")).toBe(true);
  });

  it("blocks RFC 1918 private ranges", () => {
    expect(isBlockedHostname("10.0.0.1")).toBe(true);
    expect(isBlockedHostname("10.255.255.255")).toBe(true);
    expect(isBlockedHostname("172.16.0.1")).toBe(true);
    expect(isBlockedHostname("172.31.255.255")).toBe(true);
    expect(isBlockedHostname("192.168.0.1")).toBe(true);
    expect(isBlockedHostname("192.168.255.255")).toBe(true);
  });

  it("does NOT block 172.x outside 16-31", () => {
    expect(isBlockedHostname("172.15.0.1")).toBe(false);
    expect(isBlockedHostname("172.32.0.1")).toBe(false);
  });

  it("blocks link-local + cloud metadata IPs", () => {
    expect(isBlockedHostname("169.254.169.254")).toBe(true); // AWS/GCP metadata
    expect(isBlockedHostname("169.254.1.1")).toBe(true);
  });

  it("blocks 0.0.0.0/8", () => {
    expect(isBlockedHostname("0.0.0.0")).toBe(true);
  });

  it("blocks multicast and reserved", () => {
    expect(isBlockedHostname("224.0.0.1")).toBe(true);
    expect(isBlockedHostname("239.255.255.255")).toBe(true);
    expect(isBlockedHostname("255.255.255.255")).toBe(true);
  });

  it("blocks localhost variants", () => {
    expect(isBlockedHostname("localhost")).toBe(true);
    expect(isBlockedHostname("LOCALHOST")).toBe(true);
    expect(isBlockedHostname("foo.localhost")).toBe(true);
  });

  it("blocks .local and .internal mDNS / cloud-internal", () => {
    expect(isBlockedHostname("printer.local")).toBe(true);
    expect(isBlockedHostname("queue.internal")).toBe(true);
    expect(isBlockedHostname("metadata.google.internal")).toBe(true);
    expect(isBlockedHostname("metadata.azure.com")).toBe(true);
  });

  it("blocks IPv6 loopback / ULA / link-local", () => {
    expect(isBlockedHostname("::1")).toBe(true);
    expect(isBlockedHostname("[::1]")).toBe(true);
    expect(isBlockedHostname("::")).toBe(true);
    expect(isBlockedHostname("fc00::1")).toBe(true);
    expect(isBlockedHostname("fd12:3456::1")).toBe(true);
    expect(isBlockedHostname("fe80::1")).toBe(true);
  });

  it("blocks the entire IPv6 link-local range fe80::/10 (regression for fe81-fe8f gap)", () => {
    // fe80..febf — must all be blocked. The previous string-prefix check
    // missed fe81..fe8f because it only matched "fe80:", "fe9*", "fea*", "feb*".
    for (const second of ["80", "81", "85", "8f", "90", "9a", "a0", "af", "b0", "bf"]) {
      const ip = `fe${second}::1`;
      expect(isBlockedHostname(ip)).toBe(true);
    }
  });

  it("does NOT block IPv6 outside fe80::/10 (fec0:: is deprecated site-local but not link-local)", () => {
    expect(isBlockedHostname("fec0::1")).toBe(false);
    expect(isBlockedHostname("fe7f::1")).toBe(false);
  });

  it("allows public hostnames", () => {
    expect(isBlockedHostname("example.com")).toBe(false);
    expect(isBlockedHostname("upload.wikimedia.org")).toBe(false);
    expect(isBlockedHostname("raw.githubusercontent.com")).toBe(false);
  });

  it("allows public IPv4", () => {
    expect(isBlockedHostname("8.8.8.8")).toBe(false);
    expect(isBlockedHostname("1.1.1.1")).toBe(false);
  });

  it("allows public IPv6", () => {
    expect(isBlockedHostname("2606:4700::1111")).toBe(false);
  });
});
