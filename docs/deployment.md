# Deployment — `vision.sajidafridi.com`

Target: a separate Node.js application on the existing Hostinger account, so a
crash or a bad deploy can never take down the main portfolio at
`sajidafridi.com`.

> The Hostinger MCP tooling is not available in this workspace, so these steps are
> manual (hPanel) or run through the Hostinger CLI from your machine.

## 1 · Create the subdomain

hPanel → **Domains → Subdomains** → `vision` under `sajidafridi.com`.

- Document root: `domains/vision.sajidafridi.com/public_html`
- Then **Websites → Add website / Node.js** for that subdomain, Node 22+.

## 2 · Environment

Set these as application environment variables (hPanel → Node.js → Environment).
Full replace on every change; never commit real values.

| Variable | Production value |
|---|---|
| `SENTINEL_MODEL_PATH` | `/home/u779968859/models/sentinel-det-v1.onnx` |
| `SENTINEL_MODEL_SHA` | sha256 from `benchmarks/results.json` |
| `SENTINEL_MODEL_ID` | `sentinel-det-v1` |
| `SENTINEL_CONF` | `0.35` |
| `SENTINEL_MAX_FRAMES` | `8` |
| `SENTINEL_MAX_BYTES` | `2000000` |
| `SENTINEL_RATE_PER_MIN` | `10` |
| `SENTINEL_DAILY_CAP` | `2000` |
| `SENTINEL_RETENTION_H` | `24` |
| `SENTINEL_AUDIT_DIR` | `../audit` (outside the deploy root) |
| `SENTINEL_THREADS` | `2` (shared CPU — do not starve the portfolio app) |
| `PORT` | platform-assigned |

## 3 · Weights

Never commit binaries. Pull them on the server and pin the hash:

```bash
mkdir -p ~/models && cd ~/models
# download sentinel-det-v1.onnx from the Kaggle dataset artefact
sha256sum sentinel-det-v1.onnx     # must equal SENTINEL_MODEL_SHA
```

A hash mismatch aborts boot by design — the service refuses to serve a graph it
cannot identify.

## 4 · Deploy

Connect the GitHub repo (`Mrww305/airspace-sentinel`) to the Node app for
auto-deploy on push to `main`, with root directory `apps/api`.

Build command: `npm ci && npm run build`
Start command: `npm start` (runs `node dist/server.js`) · Node 24 · root directory `apps/api`

Two install-time facts that will bite on a fresh clone or a build box with npm 11:

```bash
npm approve-scripts sharp   # npm 11 blocks install scripts by default; sharp needs its binary
npm install
```

`sharp` ships platform binaries as optional dependencies, so do **not** install
with `--omit=optional` in production — that is only for the local mock-mode
shortcut.

Then poll the deployment log in hPanel until the build reports **completed**;
before that, requests return 404/409 while the app is still being provisioned.

## 5 · Smoke test

```bash
curl -s https://vision.sajidafridi.com/health | jq
# expect: ok true, model.kind "onnx", audit.ok true

curl -s -X POST https://vision.sajidafridi.com/api/v1/detect \
  -F "frame=@docs/samples/uav.jpg" | jq '.threat, .policy'
```

Then check `kind` is `onnx` — if it says `mock`, `SENTINEL_MODEL_PATH` did not
resolve and the demo is synthetic.

## 6 · Wire it into the portfolio

- `/lab` → add **POC-04 // AIRSPACE SENTINEL** card linking the subdomain
- `/projects` → entry with the benchmark table from `benchmarks/results.json`
- JSON-LD `SoftwareApplication` + model card route for answer-engine indexing
- Cross-link from `sajidafridi.com` sitemap

## Rollback

Revert the commit and redeploy. Weights are versioned separately, so a model
regression is fixed by editing `SENTINEL_MODEL_SHA` + path, not by a code revert.
