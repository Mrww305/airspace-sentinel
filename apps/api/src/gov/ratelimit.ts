import { cfg } from "../config.ts";

interface Bucket {
  tokens: number;
  updated: number;
}

/**
 * Per-IP token bucket + a global daily ceiling. A public inference endpoint with
 * no ceiling is a way to donate someone else's CPU to a scraper.
 */
export class RateLimiter {
  #buckets = new Map<string, Bucket>();
  #daily = { day: new Date().toISOString().slice(0, 10), count: 0 };

  check(ip: string, cost = 1): { ok: boolean; retryAfterS: number; reason?: string } {
    this.#rollDay();
    if (this.#daily.count + cost > cfg.dailyCap) {
      return { ok: false, retryAfterS: 3600, reason: "global daily cap reached" };
    }

    const now = Date.now();
    const perMin = cfg.ratePerMin;
    const bucket = this.#buckets.get(ip) ?? { tokens: perMin, updated: now };
    const refill = ((now - bucket.updated) / 60_000) * perMin;
    bucket.tokens = Math.min(perMin, bucket.tokens + refill);
    bucket.updated = now;

    if (bucket.tokens < cost) {
      this.#buckets.set(ip, bucket);
      return { ok: false, retryAfterS: Math.ceil(((cost - bucket.tokens) / perMin) * 60), reason: "per-IP rate limit" };
    }

    bucket.tokens -= cost;
    this.#buckets.set(ip, bucket);
    this.#daily.count += cost;

    if (this.#buckets.size > 5000) this.#prune(now);
    return { ok: true, retryAfterS: 0 };
  }

  #rollDay(): void {
    const today = new Date().toISOString().slice(0, 10);
    if (today !== this.#daily.day) this.#daily = { day: today, count: 0 };
  }

  #prune(now: number): void {
    for (const [ip, b] of this.#buckets) if (now - b.updated > 600_000) this.#buckets.delete(ip);
  }
}

export const limiter = new RateLimiter();
