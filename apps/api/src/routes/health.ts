import type { FastifyInstance } from "fastify";
import { cfg } from "../config.ts";
import { audit } from "../gov/auditlog.ts";
import { getDetector } from "../models/registry.ts";

export async function healthRoutes(app: FastifyInstance): Promise<void> {
  app.get("/health", async () => {
    const detector = await getDetector();
    return {
      ok: true,
      service: "airspace-sentinel",
      version: "0.1.0",
      model: {
        id: detector.id,
        kind: detector.kind,
        sha256: detector.sha256,
        precision: detector.precision,
        inputSize: detector.inputSize,
      },
      caps: {
        maxFrames: cfg.maxFrames,
        maxBytes: cfg.maxBytes,
        ratePerMin: cfg.ratePerMin,
        dailyCap: cfg.dailyCap,
        retentionHours: cfg.retentionHours,
      },
      audit: audit.verify(),
      uptimeS: Math.round(process.uptime()),
    };
  });

  app.get("/api/v1/audit/recent", async (req) => {
    const limit = Math.min(200, Number((req.query as { limit?: string }).limit ?? 20));
    return { records: audit.recent(limit), chain: audit.verify() };
  });
}
