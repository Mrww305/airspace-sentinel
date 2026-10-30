import type { Box, Threat, Track } from "../types.ts";

export type Point = readonly [number, number];
/** Closed polygon in normalised 0..1 image coordinates. */
export type Zone = readonly Point[];

/** Default protected airspace: the central 60% of frame. Overridable per request. */
export const DEFAULT_ZONE: Zone = [
  [0.2, 0.2],
  [0.8, 0.2],
  [0.8, 0.8],
  [0.2, 0.8],
];

export function pointInPolygon(p: Point, poly: Zone): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i] as Point;
    const [xj, yj] = poly[j] as Point;
    const intersects = yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

export function makeZoneTest(zone: Zone, width: number, height: number) {
  return (box: Box): boolean => {
    const cx = ((box[0] + box[2]) / 2) / Math.max(1, width);
    const cy = ((box[1] + box[3]) / 2) / Math.max(1, height);
    return pointInPolygon([cx, cy], zone);
  };
}

/**
 * Escalation is a rule set, not a model. Every level carries its reasons so the
 * console can show *why* — the difference between a demo and a deployable system.
 */
export function assessThreat(tracks: Track[], width: number, height: number): Threat {
  const reasons: string[] = [];
  const inZone = tracks.filter((t) => t.inZone);
  const closing = tracks.filter((t) => t.velocityPxS > Math.max(80, width * 0.08));

  if (tracks.length === 0) return { level: "NOMINAL", reasons: ["no tracks"] };

  if (inZone.length > 0) reasons.push(`${inZone.length} track(s) inside protected zone`);
  if (closing.length > 0) reasons.push(`high closing velocity on ${closing.length} track(s)`);
  if (tracks.length > 2) reasons.push(`multiple simultaneous contacts (${tracks.length})`);

  const level = inZone.length >= 2 || (inZone.length >= 1 && closing.length >= 1)
    ? "CRITICAL"
    : inZone.length === 1 || closing.length > 0
      ? "ELEVATED"
      : "NOMINAL";

  if (level === "NOMINAL" && reasons.length === 0) reasons.push("contact outside protected zone, low velocity");

  const small = tracks.filter((t) => Math.min(t.box[2] - t.box[0], t.box[3] - t.box[1]) < 32).length;
  if (small > 0) reasons.push(`${small} track(s) below 32 px — small-object confidence band`);

  return { level, reasons };
}
