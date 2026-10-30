import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cfg } from "../config.ts";

export interface AuditRecord {
  seq: number;
  ts: string;
  prevHash: string;
  hash: string;
  payload: Record<string, unknown>;
}

/**
 * Append-only, hash-chained log. Each record commits to its predecessor, so a
 * deletion or edit anywhere in the file breaks verification from that point on.
 *
 * This is the artefact that turns "the model said so" into "here is what the
 * model saw, which weights it ran, and which rule fired".
 */
export class AuditChain {
  #file: string;
  #prevHash = "0".repeat(64);
  #seq = 0;
  #memory: AuditRecord[] = [];

  constructor(dir: string = cfg.auditDir) {
    mkdirSync(dir, { recursive: true });
    this.#file = join(dir, `audit-${new Date().toISOString().slice(0, 10)}.jsonl`);
    this.#loadTail();
  }

  #loadTail(): void {
    if (!existsSync(this.#file)) return;
    const lines = readFileSync(this.#file, "utf8").trim().split("\n").filter(Boolean);
    const last = lines.at(-1);
    if (!last) return;
    try {
      const rec = JSON.parse(last) as AuditRecord;
      this.#prevHash = rec.hash;
      this.#seq = rec.seq;
      this.#memory = lines.slice(-cfg.auditMaxMemory).map((l) => JSON.parse(l) as AuditRecord);
    } catch {
      // corrupt tail: start a new chain rather than silently trusting it
      this.#prevHash = "0".repeat(64);
      this.#seq = 0;
    }
  }

  append(payload: Record<string, unknown>): AuditRecord {
    const seq = this.#seq + 1;
    const hash = createHash("sha256").update(this.#prevHash + stable(payload)).digest("hex");
    const rec: AuditRecord = { seq, ts: new Date().toISOString(), prevHash: this.#prevHash, hash, payload };
    appendFileSync(this.#file, JSON.stringify(rec) + "\n");
    this.#prevHash = hash;
    this.#seq = seq;
    this.#memory.push(rec);
    if (this.#memory.length > cfg.auditMaxMemory) this.#memory.shift();
    return rec;
  }

  recent(n = 50): AuditRecord[] {
    return this.#memory.slice(-n);
  }

  verify(): { ok: boolean; brokenAt: number | null } {
    if (!existsSync(this.#file)) return { ok: true, brokenAt: null };
    const lines = readFileSync(this.#file, "utf8").trim().split("\n").filter(Boolean);
    let prev = "0".repeat(64);
    for (const line of lines) {
      const rec = JSON.parse(line) as AuditRecord;
      const expect = createHash("sha256").update(prev + stable(rec.payload)).digest("hex");
      if (rec.prevHash !== prev || rec.hash !== expect) return { ok: false, brokenAt: rec.seq };
      prev = rec.hash;
    }
    return { ok: true, brokenAt: null };
  }
}

/** JSON with sorted keys — hashes must not depend on property order. */
export function stable(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  const obj = value as Record<string, unknown>;
  return `{${Object.keys(obj).sort().map((k) => `${JSON.stringify(k)}:${stable(obj[k])}`).join(",")}}`;
}

export const audit = new AuditChain();
