/**
 * Runtime configuration. Every value has a safe default so the service boots
 * with no environment at all (mock mode) and tightens up in production by env.
 */
export const cfg = {
  port: Number(process.env.PORT ?? 8080),
  host: process.env.HOST ?? "0.0.0.0",

  /** Path to exported ONNX graph. Empty -> mock detector (demo without weights). */
  modelPath: process.env.SENTINEL_MODEL_PATH ?? "",
  modelId: process.env.SENTINEL_MODEL_ID ?? "sentinel-det-v1",
  /** Expected sha256 of the graph; empty disables the check (dev only). */
  modelSha: process.env.SENTINEL_MODEL_SHA ?? "",

  confThreshold: Number(process.env.SENTINEL_CONF ?? 0.35),
  iouThreshold: Number(process.env.SENTINEL_IOU ?? 0.45),

  maxFrames: Number(process.env.SENTINEL_MAX_FRAMES ?? 8),
  maxBytes: Number(process.env.SENTINEL_MAX_BYTES ?? 2_000_000),

  ratePerMin: Number(process.env.SENTINEL_RATE_PER_MIN ?? 10),
  dailyCap: Number(process.env.SENTINEL_DAILY_CAP ?? 2000),

  retentionHours: Number(process.env.SENTINEL_RETENTION_H ?? 24),
  auditDir: process.env.SENTINEL_AUDIT_DIR ?? "audit",
  auditMaxMemory: Number(process.env.SENTINEL_AUDIT_MEMORY ?? 200),
} as const;
