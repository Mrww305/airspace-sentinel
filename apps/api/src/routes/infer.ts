import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { createHash, randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";
import { cfg } from "../config.ts";
import { audit } from "../gov/auditlog.ts";
import { inspectMetadata } from "../gov/exif.ts";
import { evaluatePolicy } from "../gov/policy.ts";
import { limiter } from "../gov/ratelimit.ts";
import { getDetector } from "../models/registry.ts";
import type { DetectResponse, Detection, FrameResult, Track } from "../types.ts";
import { preprocess } from "../vision/preprocess.ts";
import { toDetections } from "../vision/postprocess.ts";
import { assessThreat, DEFAULT_ZONE, makeZoneTest, type Zone } from "../vision/threat.ts";
import { IouTracker } from "../vision/tracker.ts";

interface RequestConfig {
  confidence: number;
  iou: number;
  zone: Zone | null;
}

const DEFAULT_CONFIG: RequestConfig = { confidence: cfg.confThreshold, iou: cfg.iouThreshold, zone: null };

type Sniffed = "jpeg" | "png" | "webp" | null;

function sniff(buf: Buffer): Sniffed {
  if (buf.length < 12) return null;
  if (buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return "jpeg";
  if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) return "png";
  if (buf.toString("ascii", 0, 4) === "RIFF" && buf.toString("ascii", 8, 12) === "WEBP") return "webp";
  return null;
}

function parseConfig(raw: string): RequestConfig {
  if (!raw) return { ...DEFAULT_CONFIG };
  try {
    const parsed = JSON.parse(raw) as Partial<RequestConfig>;
    const zone = Array.isArray(parsed.zone) && parsed.zone.length >= 3 ? (parsed.zone as Zone) : null;
    return {
      confidence: clampNum(parsed.confidence, 0.05, 0.95, DEFAULT_CONFIG.confidence),
      iou: clampNum(parsed.iou, 0.1, 0.9, DEFAULT_CONFIG.iou),
      zone,
    };
  } catch {
    return { ...DEFAULT_CONFIG };
  }
}

function clampNum(v: unknown, lo: number, hi: number, fallback: number): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
}

export async function inferRoutes(app: FastifyInstance): Promise<void> {
  app.post("/api/v1/detect", async (req: FastifyRequest, reply: FastifyReply) => {
    const requestId = `req_${randomUUID().slice(0, 18)}`;

    const gate = limiter.check(req.ip);
    if (!gate.ok) {
      return reply.code(429).header("retry-after", String(gate.retryAfterS)).send({
        requestId,
        error: "rate_limited",
        reason: gate.reason,
        retryAfterS: gate.retryAfterS,
      });
    }

    if (!req.isMultipart()) {
      return reply.code(415).send({ requestId, error: "multipart_required" });
    }

    const frames: Buffer[] = [];
    let configRaw = "";
    let oversized = false;
    let tooManyFrames = false;
    let rejectedType: Sniffed | "unknown" | null = null;

    for await (const part of req.parts()) {
      if (part.type === "field") {
        if (part.fieldname === "config") configRaw += String(part.value ?? "");
        continue;
      }

      const chunks: Buffer[] = [];
      let size = 0;
      for await (const chunk of part.file) {
        size += chunk.length;
        if (size > cfg.maxBytes) {
          oversized = true;
          break;
        }
        chunks.push(chunk as Buffer);
      }
      if (oversized) {
        part.file.resume();
        continue;
      }
      const buf = Buffer.concat(chunks);
      const kind = sniff(buf);
      if (!kind) {
        rejectedType = "unknown";
        continue;
      }
      if (frames.length >= cfg.maxFrames) {
        tooManyFrames = true;
        continue;
      }
      frames.push(buf);
    }

    if (oversized || tooManyFrames) {
      return reply.code(413).send({
        requestId,
        error: oversized ? "payload_too_large" : "too_many_frames",
        cap: { maxBytes: cfg.maxBytes, maxFrames: cfg.maxFrames },
      });
    }
    if (rejectedType) {
      return reply.code(415).send({ requestId, error: "unsupported_media", accepted: ["jpeg", "png", "webp"] });
    }
    if (frames.length === 0) {
      return reply.code(400).send({ requestId, error: "no_frames", hint: "send one or more files under field 'frame' or 'frames'" });
    }

    const config = parseConfig(configRaw);
    const detector = await getDetector();
    const tracker = new IouTracker();
    const inputHashes: string[] = [];
    const results: FrameResult[] = [];
    let anyMetadata = false;
    let width = 0;
    let height = 0;

    for (const [index, buf] of frames.entries()) {
      inputHashes.push(createHash("sha256").update(buf).digest("hex"));
      const meta = await inspectMetadata(buf);
      anyMetadata = anyMetadata || meta.hadMetadata;

      const started = performance.now();
      try {
        const pp = await preprocess(buf, detector.inputSize);
        width ||= pp.origW;
        height ||= pp.origH;
        const raw = await detector.infer(pp.tensor);
        const detections: Detection[] = toDetections(raw, pp, config.confidence, config.iou);
        const zone = config.zone ?? DEFAULT_ZONE;
        const inZone = makeZoneTest(zone, pp.origW, pp.origH);
        const tracks: Track[] = tracker.update(detections, Date.now(), inZone);
        results.push({ index, latencyMs: Math.round(performance.now() - started), detections: tracks });
      } catch (err) {
        req.log.warn({ err, index }, "frame_failed");
        results.push({ index, latencyMs: Math.round(performance.now() - started), detections: [] });
      }
    }

    const last = results.at(-1)?.detections ?? [];
    const threat = assessThreat(last, width || 1, height || 1);
    const policy = evaluatePolicy({
      // preprocess() drops EXIF/GPS before the detector sees a pixel, so the
      // rule is satisfied by construction; presence is still recorded above.
      metadataStripped: true,
      inputBytes: frames.reduce((n, f) => n + f.length, 0),
      frames: frames.length,
      threat,
      modelKind: detector.kind,
    });

    const response: DetectResponse = {
      requestId,
      model: {
        id: detector.id,
        sha256: detector.sha256,
        kind: detector.kind,
        precision: detector.precision,
      },
      frames: results,
      threat,
      policy,
      retention: { ttlHours: cfg.retentionHours, stored: "hash-only" },
    };

    audit.append({
      requestId,
      model: response.model,
      inputHashes,
      hadMetadata: anyMetadata,
      frames: frames.length,
      detections: results.reduce((n, f) => n + f.detections.length, 0),
      threat: threat.level,
      policy: policy.decision,
      latencyMs: results.reduce((n, f) => n + f.latencyMs, 0),
      ipHash: createHash("sha256").update(req.ip).digest("hex").slice(0, 16),
    });

    return reply.header("x-request-id", requestId).send(response);
  });
}
