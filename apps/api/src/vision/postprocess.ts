import type { Box, Detection, RawPrediction } from "../types.ts";
import type { Preprocessed } from "./preprocess.ts";

/**
 * Normalised (0..1, letterboxed) predictions -> pixel boxes in the ORIGINAL
 * image, then greedy NMS. Doing this on the server, not the client, means the
 * numbers in the audit record are the numbers the policy engine saw.
 */
export function toDetections(
  raw: RawPrediction,
  pp: Preprocessed,
  confThreshold: number,
  iouThreshold: number,
): Detection[] {
  const candidates: Detection[] = [];
  const n = raw.scores.length;

  for (let i = 0; i < n; i++) {
    const score = raw.scores[i] ?? 0;
    if (score < confThreshold) continue;

    const cx = raw.boxes[i * 4] ?? 0;
    const cy = raw.boxes[i * 4 + 1] ?? 0;
    const w = raw.boxes[i * 4 + 2] ?? 0;
    const h = raw.boxes[i * 4 + 3] ?? 0;

    const x1 = unpadX((cx - w / 2) * pp.size, pp);
    const y1 = unpadY((cy - h / 2) * pp.size, pp);
    const x2 = unpadX((cx + w / 2) * pp.size, pp);
    const y2 = unpadY((cy + h / 2) * pp.size, pp);

    const box: Box = [
      clamp(x1, 0, pp.origW),
      clamp(y1, 0, pp.origH),
      clamp(x2, 0, pp.origW),
      clamp(y2, 0, pp.origH),
    ];
    if (box[2] - box[0] < 1 || box[3] - box[1] < 1) continue;

    candidates.push({ label: "drone", confidence: round(score), box });
  }

  return nms(candidates, iouThreshold);
}

function unpadX(v: number, pp: Preprocessed): number {
  return (v - pp.padX) / pp.scale;
}

function unpadY(v: number, pp: Preprocessed): number {
  return (v - pp.padY) / pp.scale;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function round(v: number): number {
  return Math.round(v * 1000) / 1000;
}

export function iou(a: Box, b: Box): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter === 0) return 0;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

export function nms(dets: Detection[], iouThreshold: number): Detection[] {
  const sorted = [...dets].sort((a, b) => b.confidence - a.confidence);
  const keep: Detection[] = [];
  for (const det of sorted) {
    if (keep.every((k) => iou(k.box, det.box) <= iouThreshold)) keep.push(det);
  }
  return keep;
}
