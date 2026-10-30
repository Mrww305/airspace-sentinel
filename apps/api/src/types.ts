/** Shared wire types. Kept in one place so the console and API cannot drift. */

/** xyxy in original-image pixels. */
export type Box = readonly [number, number, number, number];

export interface Detection {
  label: string;
  confidence: number;
  box: Box;
}

export interface Track extends Detection {
  trackId: number;
  /** Magnitude of centre displacement between consecutive frames, px/s. */
  velocityPxS: number;
  inZone: boolean;
}

export interface FrameResult {
  index: number;
  latencyMs: number;
  detections: Track[];
}

export type ThreatLevel = "NOMINAL" | "ELEVATED" | "CRITICAL";

export interface Threat {
  level: ThreatLevel;
  reasons: string[];
}

export interface PolicyVerdict {
  decision: "allow" | "redact" | "deny";
  rules: string[];
}

export interface DetectResponse {
  requestId: string;
  model: { id: string; sha256: string; kind: "onnx" | "mock"; precision: string };
  frames: FrameResult[];
  threat: Threat;
  policy: PolicyVerdict;
  retention: { ttlHours: number; stored: "hash-only" };
}

/** Raw detector output, normalised to 0..1 boxes, before NMS and rescaling. */
export interface RawPrediction {
  /** cx, cy, w, h in 0..1 */
  boxes: Float32Array;
  scores: Float32Array;
  classes: Int32Array;
}
