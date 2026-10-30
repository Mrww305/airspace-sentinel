import { cfg } from "../config.ts";
import type { PolicyVerdict, Threat } from "../types.ts";

export interface PolicyContext {
  metadataStripped: boolean;
  inputBytes: number;
  frames: number;
  threat: Threat;
  modelKind: "onnx" | "mock";
}

interface Rule {
  id: string;
  evaluate: (ctx: PolicyContext) => boolean;
}

/**
 * Declarative, inspectable rules — deliberately not another model.
 *
 * A governance claim you cannot read is not a governance claim. Each rule
 * returns a boolean and its id goes into the response and the audit record.
 */
const RULES: Rule[] = [
  { id: "pii-metadata-stripped", evaluate: (c) => c.metadataStripped },
  { id: "ingest-size-under-cap", evaluate: (c) => c.inputBytes <= cfg.maxBytes * c.frames },
  {
    id: "retention-declared",
    evaluate: () => cfg.retentionHours > 0 && cfg.retentionHours <= 24 * 30,
  },
  { id: "synthetic-model-labelled", evaluate: (c) => c.modelKind === "onnx" || c.modelKind === "mock" },
];

export function evaluatePolicy(ctx: PolicyContext): PolicyVerdict {
  const failed = RULES.filter((r) => !r.evaluate(ctx)).map((r) => `FAIL ${r.id}`);
  const passed = RULES.filter((r) => r.evaluate(ctx)).map((r) => `PASS ${r.id}`);

  // A synthetic model may be shown, never escalated. Otherwise: high threat with
  // any failed rule is denied outright.
  let decision: PolicyVerdict["decision"] = "allow";
  if (ctx.modelKind === "mock" && ctx.threat.level === "CRITICAL") decision = "redact";
  if (failed.length > 0) decision = "deny";

  return { decision, rules: [...passed, ...failed] };
}
