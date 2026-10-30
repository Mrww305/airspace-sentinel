# API — inference + governance

Node 24, Fastify, ONNX Runtime. No Python in production.

```bash
npm install            # includes onnxruntime-node + sharp binaries
cp .env.example .env   # optional; defaults are safe
npm run dev            # http://localhost:8080/health
```

Node 24 runs TypeScript directly (type stripping) — no build step, no `ts-node`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | model identity, sha256, ingest caps, audit chain status |
| `POST` | `/api/v1/detect` | multipart: 1–8 images → detections, tracks, threat, policy |
| `GET` | `/api/v1/audit/recent` | last N hash-chained audit records + chain verification |

### `POST /api/v1/detect`

```bash
curl -X POST http://localhost:8080/api/v1/detect \
  -F "frame=@uav.jpg" \
  -F "frame=@uav-2.jpg" \
  -F 'config={"confidence":0.35,"iou":0.45,"zone":[[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]]}'
```

Field names are free-form; any file part is treated as a frame. `config` is optional JSON.

Response (abridged):

```jsonc
{
  "requestId": "req_…",
  "model": { "id": "sentinel-det-v1", "sha256": "…", "kind": "onnx", "precision": "int8" },
  "frames": [{
    "index": 0, "latencyMs": 312,
    "detections": [{
      "label": "drone", "confidence": 0.87,
      "box": [412, 208, 470, 260],
      "trackId": 1, "velocityPxS": 61.2, "inZone": true
    }]
  }],
  "threat":   { "level": "ELEVATED", "reasons": ["1 track(s) inside protected zone"] },
  "policy":   { "decision": "allow", "rules": ["PASS pii-metadata-stripped", "…"] },
  "retention":{ "ttlHours": 24, "stored": "hash-only" }
}
```

Errors: `400 no_frames`, `413 payload_too_large | too_many_frames`, `415 unsupported_media | multipart_required`, `429 rate_limited` (+ `retry-after`).

## Mock mode

`SENTINEL_MODEL_PATH` empty ⇒ synthetic detector, deterministic per input. The
whole pipeline (tracking, threat rules, policy, audit chain) stays exercisable
before weights exist, and the console can be built in parallel with training.
Mock responses are marked `kind: "mock"` and can never escalate to a `deny`.

## Layout

```
src/
  config.ts              env → typed config, safe defaults
  types.ts               wire contract (shared with the console)
  models/registry.ts     ONNX session, sha256 pinning, format detection, mock
  vision/preprocess.ts   sharp: orient → letterbox → CHW normalise (drops EXIF)
  vision/postprocess.ts  unletterbox → pixel boxes → greedy NMS
  vision/tracker.ts      IoU tracker, velocity estimate
  vision/threat.ts       zone geometry + escalation rules
  gov/auditlog.ts        append-only sha256 hash chain
  gov/policy.ts          declarative rules → allow / redact / deny
  gov/ratelimit.ts       per-IP token bucket + global daily cap
  gov/exif.ts            metadata presence attestation
  routes/health.ts       health + audit inspection
  routes/infer.ts        multipart ingest → pipeline → response
```

## Design notes

**Why the server does NMS, not the client.** The numbers in the audit record must
be the numbers the policy engine saw. Client-side filtering makes the record a
claim rather than evidence.

**Why metadata is stripped before decode.** Drone imagery usually carries GPS.
Persisting coordinates of an operator is a privacy event; removing them after
inference is too late because the bytes already existed on disk.

**Why the audit chain is hash-chained and not a database table.** A row can be
edited by whoever owns the database. A chain breaks visibly. `GET /health`
returns `{ ok, brokenAt }` so tampering is detectable from outside.

**Why policy is rules, not a model.** A governance claim you cannot read is not a
governance claim.
