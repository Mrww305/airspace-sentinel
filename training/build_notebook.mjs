#!/usr/bin/env node
/**
 * Generates airspace-sentinel-train.ipynb from airspace_sentinel_train.py.
 *
 * The .py is the single source of truth (jupytext `# %%` convention) so the
 * pipeline is diffable, lintable and runnable headless:
 *
 *   python airspace_sentinel_train.py     # script
 *   node build_notebook.mjs               # -> .ipynb for Kaggle
 */
import { readFileSync, writeFileSync } from "node:fs";

const SRC = new URL("./airspace_sentinel_train.py", import.meta.url);
const DST = new URL("./airspace-sentinel-train.ipynb", import.meta.url);

const lines = readFileSync(SRC, "utf8").split(/\r?\n/);
const cells = [];
let buf = [];
let kind = "code";

const flush = () => {
  // trim leading/trailing blank lines, keep interior formatting
  while (buf.length && buf[0].trim() === "") buf.shift();
  while (buf.length && buf[buf.length - 1].trim() === "") buf.pop();
  if (!buf.length) return;
  const text = buf.join("\n");
  cells.push(
    kind === "markdown"
      ? { cell_type: "markdown", metadata: {}, source: text.split("\n").map((l) => l + "\n") }
      : {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: text.split("\n").map((l) => l + "\n"),
        },
  );
  buf = [];
};

for (const line of lines) {
  const m = line.match(/^#\s*%%(?:\s*\[(\w+)\])?\s*$/);
  if (m) {
    flush();
    kind = m[1] === "markdown" ? "markdown" : "code";
    continue;
  }
  // strip the markdown comment prefix inside markdown cells
  buf.push(kind === "markdown" ? line.replace(/^# ?/, "") : line);
}
flush();

const nb = {
  cells,
  metadata: {
    kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
    language_info: { name: "python", version: "3.11" },
    kaggle: { accelerator: "nvidiaTeslaT4", dataSources: [], isGpuEnabled: true, isInternetEnabled: true },
  },
  nbformat: 4,
  nbformat_minor: 5,
};

writeFileSync(DST, JSON.stringify(nb, null, 1) + "\n");
console.log(`wrote ${cells.length} cells (${cells.filter((c) => c.cell_type === "markdown").length} md) -> airspace-sentinel-train.ipynb`);

/**
 * Kaggle CLI staging: `kaggle kernels push -p training/kaggle` uploads whatever
 * sits next to kernel-metadata.json, so the notebook is copied in under the
 * name the metadata expects.
 */
import { mkdirSync, copyFileSync } from "node:fs";

const KDIR = new URL("./kaggle/", import.meta.url);
mkdirSync(KDIR, { recursive: true });
copyFileSync(DST, new URL("airspace-sentinel-train.ipynb", KDIR));

const meta = {
  id: "mrww305/airspace-sentinel-train",
  title: "AIRSPACE SENTINEL - counter-UAS detector",
  code_file: "airspace-sentinel-train.ipynb",
  language: "python",
  kernel_type: "notebook",
  is_private: true,
  enable_gpu: true,
  enable_internet: true,
  dataset_sources: [
    "dasmehdixtr/drone-dataset-uav",
    "muki2003/yolo-drone-detection-dataset",
    "muhammadsaoodsarwar/drone-vs-bird",
  ],
  competition_sources: [],
  kernel_sources: [],
  model_sources: [],
  docker_image: "gcr.io/kaggle-gpu-images/python",
};
writeFileSync(new URL("kernel-metadata.json", KDIR), JSON.stringify(meta, null, 2) + "\n");
console.log("staged kaggle/ (metadata + notebook) -> push with: kaggle kernels push -p training/kaggle");
