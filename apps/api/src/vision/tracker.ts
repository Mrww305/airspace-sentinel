import type { Detection, Track } from "../types.ts";
import { iou } from "./postprocess.ts";

interface LiveTrack {
  id: number;
  box: Detection["box"];
  confidence: number;
  lastCentre: [number, number];
  lastTs: number;
  velocity: number;
  misses: number;
}

/**
 * SORT-lite: greedy IoU association, constant-velocity estimate, no appearance
 * model. Deliberately simple — the model card says so, rather than pretending
 * this re-identifies a UAV after occlusion.
 */
export class IouTracker {
  #tracks = new Map<number, LiveTrack>();
  #nextId = 1;
  #ttlMs: number;

  constructor(ttlMs = 1500) {
    this.#ttlMs = ttlMs;
  }

  reset(): void {
    this.#tracks.clear();
    this.#nextId = 1;
  }

  update(detections: Detection[], now: number, inZone: (box: Detection["box"]) => boolean): Track[] {
    const assigned = new Set<number>();
    const out: Track[] = [];

    for (const det of detections) {
      let best: LiveTrack | null = null;
      let bestIou = 0.3;
      for (const t of this.#tracks.values()) {
        if (assigned.has(t.id)) continue;
        const score = iou(t.box, det.box);
        if (score > bestIou) {
          bestIou = score;
          best = t;
        }
      }

      const centre: [number, number] = [(det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2];
      if (best) {
        const dt = Math.max(1, (now - best.lastTs) / 1000);
        const dx = centre[0] - best.lastCentre[0];
        const dy = centre[1] - best.lastCentre[1];
        best.velocity = Math.round((Math.hypot(dx, dy) / dt) * 10) / 10;
        best.box = det.box;
        best.confidence = det.confidence;
        best.lastCentre = centre;
        best.lastTs = now;
        best.misses = 0;
        assigned.add(best.id);
        out.push(toTrack(best, inZone(det.box)));
      } else {
        const track: LiveTrack = {
          id: this.#nextId++,
          box: det.box,
          confidence: det.confidence,
          lastCentre: centre,
          lastTs: now,
          velocity: 0,
          misses: 0,
        };
        this.#tracks.set(track.id, track);
        assigned.add(track.id);
        out.push(toTrack(track, inZone(det.box)));
      }
    }

    // age out stale tracks
    for (const [id, t] of [...this.#tracks]) {
      if (assigned.has(id)) continue;
      t.misses += 1;
      if (now - t.lastTs > this.#ttlMs) this.#tracks.delete(id);
    }

    return out;
  }
}

function toTrack(t: LiveTrack, inZone: boolean): Track {
  return {
    label: "drone",
    confidence: t.confidence,
    box: t.box,
    trackId: t.id,
    velocityPxS: t.velocity,
    inZone,
  };
}
