import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { cfg } from "../config.ts";
import type { RawPrediction } from "../types.ts";

export interface Detector {
  id: string;
  sha256: string;
  kind: "onnx" | "mock";
  precision: string;
  inputSize: number;
  infer(tensor: Float32Array): Promise<RawPrediction>;
}

type OrtSession = {
  inputNames: string[];
  outputNames: string[];
  inputMetadata: Array<{ name: string; dimensions: number[] }>;
  outputMetadata: Array<{ name: string; dimensions: number[] }>;
  run: (feeds: Record<string, unknown>) => Promise<Array<{ data: Float32Array; dims: number[] }>>;
};

let cached: Detector | null = null;

export async function getDetector(): Promise<Detector> {
  if (cached) return cached;
  cached = cfg.modelPath ? await loadOnnx(cfg.modelPath) : mockDetector();
  return cached;
}

/** sha256 of the graph — recorded in every audit record so a decision can be reproduced. */
export function fileSha256(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

async function loadOnnx(path: string): Promise<Detector> {
  const sha = fileSha256(path);
  if (cfg.modelSha && cfg.modelSha !== sha) {
    throw new Error(`model sha256 mismatch: expected ${cfg.modelSha}, got ${sha}`);
  }

  // Optional dependency: absent -> mock mode, not a crash.
  const ort = (await import("onnxruntime-node").catch(() => null)) as {
    InferenceSession: { create(p: string, o?: unknown): Promise<OrtSession> };
    Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
  } | null;
  if (!ort) throw new Error("onnxruntime-node is not installed (npm install) — refusing to serve a half-configured model");

  const session = await ort.InferenceSession.create(path, {
    executionProviders: ["cpu"],
    intraOpNumThreads: Number(process.env.SENTINEL_THREADS ?? 4),
    graphOptimizationLevel: "all",
  });

  const input = session.inputMetadata[0];
  if (!input) throw new Error("model exposes no inputs");
  const dims = input.dimensions;
  const inputSize = typeof dims[2] === "number" && dims[2] > 0 ? dims[2] : 640;
  const format = detectFormat(session);

  return {
    id: cfg.modelId,
    sha256: sha,
    kind: "onnx",
    precision: /int8/i.test(path) ? "int8" : "fp32",
    inputSize,
    async infer(tensor: Float32Array): Promise<RawPrediction> {
      const feeds = { [input.name]: new ort.Tensor("float32", tensor, [1, 3, inputSize, inputSize]) };
      const out = await session.run(feeds);
      const arrs = out.map((o) => ({ data: o.data, dims: o.dims }));
      return format === "detr" ? decodeDetr(arrs) : decodeFlat(arrs);
    },
  };
}

/**
 * Export paths differ: RF-DETR's `export()` may emit decoded rows, while a raw
 * `torch.onnx.export` emits DETR logits+boxes. Detect once at load, not per request.
 */
function detectFormat(session: OrtSession): "detr" | "flat" {
  const outs = session.outputMetadata ?? [];
  if (outs.length >= 2 && outs[0]?.dimensions.length === 3 && outs[1]?.dimensions.length === 3) return "detr";
  return "flat";
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function decodeDetr(arrs: Array<{ data: Float32Array; dims: number[] }>): RawPrediction {
  const [a, b] = arrs;
  if (!a || !b) throw new Error("expected two output tensors");
  // Identify logits (last dim = classes) vs boxes (last dim = 4).
  const aLast = a.dims[a.dims.length - 1] ?? 0;
  const [logits, boxes] = aLast > 4 ? [a, b] : [b, a];

  const numClasses = logits.dims[logits.dims.length - 1] ?? 1;
  const queries = logits.data.length / numClasses;
  const outBoxes = new Float32Array(queries * 4);
  const outScores = new Float32Array(queries);
  const outClasses = new Int32Array(queries);

  for (let q = 0; q < queries; q++) {
    let best = -Infinity;
    let bestClass = 0;
    for (let c = 0; c < numClasses; c++) {
      const p = sigmoid(logits.data[q * numClasses + c] ?? 0);
      if (p > best) {
        best = p;
        bestClass = c;
      }
    }
    outScores[q] = best;
    outClasses[q] = bestClass;
    for (let k = 0; k < 4; k++) outBoxes[q * 4 + k] = boxes.data[q * 4 + k] ?? 0;
  }
  return { boxes: outBoxes, scores: outScores, classes: outClasses };
}

function decodeFlat(arrs: Array<{ data: Float32Array; dims: number[] }>): RawPrediction {
  const t = arrs[0];
  if (!t) throw new Error("model returned no output");
  const rows = t.dims.length === 3 ? (t.dims[1] ?? 0) : (t.dims[0] ?? 0);
  const cols = t.dims[t.dims.length - 1] ?? 0;
  const stride = cols >= 6 ? cols : 6;
  const outBoxes = new Float32Array(rows * 4);
  const outScores = new Float32Array(rows);
  const outClasses = new Int32Array(rows);

  for (let r = 0; r < rows; r++) {
    const o = r * stride;
    const x1 = t.data[o] ?? 0;
    const y1 = t.data[o + 1] ?? 0;
    const w = t.data[o + 2] ?? 0;
    const h = t.data[o + 3] ?? 0;
    outScores[r] = t.data[o + 4] ?? 0;
    outClasses[r] = Math.trunc(t.data[o + 5] ?? 0);
    // flat exports are usually already xywh-normalised; normalise defensively
    outBoxes[r * 4] = x1 + w / 2;
    outBoxes[r * 4 + 1] = y1 + h / 2;
    outBoxes[r * 4 + 2] = w;
    outBoxes[r * 4 + 3] = h;
  }
  return { boxes: outBoxes, scores: outScores, classes: outClasses };
}

/**
 * Mock detector: keeps the whole pipeline (tracking, threat rules, audit chain)
 * exercisable before weights exist. Deterministic per input so demos are stable.
 */
function mockDetector(): Detector {
  return {
    id: `${cfg.modelId}-mock`,
    sha256: "mock-0000000000000000000000000000000000000000000000000000000000000000",
    kind: "mock",
    precision: "none",
    inputSize: 640,
    async infer(tensor: Float32Array): Promise<RawPrediction> {
      let h = 2166136261;
      for (let i = 0; i < tensor.length; i += 997) h = Math.imul(h ^ ((tensor[i] ?? 0) * 1e6) | 0, 16777619);
      const n = 1 + (Math.abs(h) % 3);
      const boxes = new Float32Array(n * 4);
      const scores = new Float32Array(n);
      const classes = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const seed = Math.abs(Math.imul(h, i + 7));
        boxes[i * 4] = 0.25 + ((seed % 50) / 100);
        boxes[i * 4 + 1] = 0.25 + ((seed % 37) / 100);
        boxes[i * 4 + 2] = 0.04 + ((seed % 9) / 200);
        boxes[i * 4 + 3] = 0.04 + ((seed % 7) / 200);
        scores[i] = 0.45 + ((seed % 45) / 100);
      }
      return { boxes, scores, classes };
    },
  };
}
