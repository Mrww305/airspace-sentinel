import Fastify from "fastify";
import multipart from "@fastify/multipart";
import { cfg } from "./config.ts";
import { getDetector } from "./models/registry.ts";
import { healthRoutes } from "./routes/health.ts";
import { inferRoutes } from "./routes/infer.ts";

const app = Fastify({
  logger: { level: process.env.LOG_LEVEL ?? "info" },
  trustProxy: true,
  bodyLimit: cfg.maxBytes * Math.max(1, cfg.maxFrames),
});

await app.register(multipart, {
  limits: { fileSize: cfg.maxBytes, files: cfg.maxFrames, fields: 4 },
});

await healthRoutes(app);
await inferRoutes(app);

app.setErrorHandler((err: Error, _req, reply) => {
  app.log.error({ err }, "unhandled");
  reply.code(500).send({ error: "internal_error", message: err.message });
});

// Fail fast: a misconfigured model should stop the boot, not the first request.
const detector = await getDetector();
app.log.info(`detector ${detector.id} kind=${detector.kind} sha256=${detector.sha256.slice(0, 12)}…`);

await app.listen({ port: cfg.port, host: cfg.host });
app.log.info(`airspace-sentinel listening on :${cfg.port}`);
