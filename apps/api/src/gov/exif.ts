import sharp from "sharp";

/**
 * Reports whether an upload carried metadata (EXIF/GPS/IPTC/XMP) so the audit
 * record can state that it was present and stripped. The stripping itself
 * happens in `preprocess` — sharp drops metadata unless explicitly preserved.
 */
export async function inspectMetadata(buf: Buffer): Promise<{ hadMetadata: boolean; tags: string[] }> {
  try {
    const meta = await sharp(buf, { failOn: "none" }).metadata();
    const tags: string[] = [];
    let hadMetadata = false;
    if (meta.exif) { tags.push("exif"); hadMetadata = true; }
    if (meta.iptc) { tags.push("iptc"); hadMetadata = true; }
    if (meta.xmp) { tags.push("xmp"); hadMetadata = true; }
    if (meta.orientation && meta.orientation !== 1) { tags.push("orientation"); hadMetadata = true; }
    if (meta.width && meta.height) tags.push(`${meta.width}x${meta.height}`);
    return { hadMetadata, tags };
  } catch {
    return { hadMetadata: false, tags: [] };
  }
}
