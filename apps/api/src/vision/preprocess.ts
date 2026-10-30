import sharp from "sharp";

export interface Preprocessed {
  /** CHW, RGB, /255, size*size*3 floats. */
  tensor: Float32Array;
  origW: number;
  origH: number;
  scale: number;
  padX: number;
  padY: number;
  size: number;
  /** True if the upload carried EXIF/GPS — recorded, never persisted. */
  hadMetadata: boolean;
}

/**
 * Decode -> correct orientation -> letterbox -> normalise.
 *
 * `sharp` drops all metadata on output unless asked otherwise, so EXIF/GPS is
 * gone before the first byte of pixel data reaches the detector. That ordering
 * is the governance requirement, not an optimisation.
 */
export async function preprocess(buf: Buffer, size: number): Promise<Preprocessed> {
  const source = sharp(buf, { failOn: "none", limitInputPixels: 40_000_000 });
  const meta = await source.metadata();
  const hadMetadata = Boolean(meta.exif || meta.iptc || meta.xmp || meta.orientation);

  const img = sharp(buf, { failOn: "none", limitInputPixels: 40_000_000 }).rotate();
  const rotated = await img.metadata();
  const origW = rotated.width ?? 0;
  const origH = rotated.height ?? 0;
  if (!origW || !origH) throw new Error("unreadable image dimensions");

  const { data } = await img
    .resize({
      width: size,
      height: size,
      fit: "contain",
      background: { r: 114, g: 114, b: 114, alpha: 1 },
    })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const scale = Math.min(size / origW, size / origH);
  const padX = (size - Math.round(origW * scale)) / 2;
  const padY = (size - Math.round(origH * scale)) / 2;

  const plane = size * size;
  const tensor = new Float32Array(plane * 3);
  for (let i = 0, p = 0; i < plane; i++, p += 3) {
    tensor[i] = (data[p] ?? 0) / 255;
    tensor[plane + i] = (data[p + 1] ?? 0) / 255;
    tensor[plane * 2 + i] = (data[p + 2] ?? 0) / 255;
  }

  return { tensor, origW, origH, scale, padX, padY, size, hadMetadata };
}
