import { XMLParser } from 'fast-xml-parser';
import { fromArrayBuffer } from 'geotiff';
import type { UploadFilePayload } from './types';

const FLOAT_EPSILON = 1e-6;
const xmlParser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: '',
  trimValues: false,
});

type Rect = {
  xOff: number;
  yOff: number;
  xSize: number;
  ySize: number;
};

type VrtSource = {
  kind: 'SimpleSource' | 'ComplexSource';
  sourceFilename: string;
  sourceBand: number;
  srcRect: Rect | null;
  dstRect: Rect | null;
  sourceNoData: number | null;
};

type VrtDataset = {
  width: number;
  height: number;
  geoTransform: AffineTransform;
  bandNoData: number | null;
  sources: VrtSource[];
};

type AffineTransform = [number, number, number, number, number, number];

type DecodedGeoTiff = {
  fileName: string;
  width: number;
  height: number;
  transform: AffineTransform;
  elevations: Float32Array;
  validMask: Uint8Array;
  maxElevation: number;
  nodata: number | null;
};

export type BrowserRasterSurface = {
  resolvedRasterName: string;
  width: number;
  height: number;
  transform: AffineTransform;
  elevations: Float32Array;
  validMask: Uint8Array;
  maxElevation: number;
};

export type BrowserRasterProgress = {
  fraction: number;
  message: string;
};

export type BrowserTerrainInputKind = 'hdf' | 'dem';

type FileLookup = {
  byLowerName: Map<string, UploadFilePayload[]>;
  byLowerBasename: Map<string, UploadFilePayload[]>;
};

type XmlNode =
  | string
  | number
  | null
  | undefined
  | XmlNode[]
  | { [key: string]: XmlNode };

function lower(value: string): string {
  return value.toLowerCase();
}

function basename(value: string): string {
  const normalized = value.replace(/\\/g, '/');
  const lastSlash = normalized.lastIndexOf('/');
  return lastSlash >= 0 ? normalized.slice(lastSlash + 1) : normalized;
}

function isTiffName(value: string): boolean {
  const normalized = lower(value);
  return normalized.endsWith('.tif') || normalized.endsWith('.tiff');
}

function requireFiniteNumber(rawValue: string | null, label: string): number {
  if (rawValue === null) {
    throw new Error(`Missing ${label} in the VRT.`);
  }

  const value = Number(rawValue.trim());
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid numeric ${label} in the VRT: ${rawValue}`);
  }
  return value;
}

function parseOptionalNumber(rawValue: string | null): number | null {
  if (rawValue === null) {
    return null;
  }
  const trimmed = rawValue.trim();
  if (!trimmed) {
    return null;
  }
  const value = Number(trimmed);
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid numeric value in the VRT: ${rawValue}`);
  }
  return value;
}

function asRecord(node: XmlNode, label: string): Record<string, XmlNode> {
  if (!node || typeof node !== 'object' || Array.isArray(node)) {
    throw new Error(`Invalid ${label} in the VRT.`);
  }
  return node as Record<string, XmlNode>;
}

function asArray<T>(value: T | T[] | undefined): T[] {
  if (value === undefined) {
    return [];
  }
  return Array.isArray(value) ? value : [value];
}

function supportedKeys(record: Record<string, XmlNode>, allowedKeys: readonly string[]): string[] {
  return Object.keys(record).filter((key) => key !== '#text' && !allowedKeys.includes(key));
}

function textValue(node: XmlNode): string | null {
  if (node === null || node === undefined) {
    return null;
  }
  if (typeof node === 'string' || typeof node === 'number') {
    return String(node);
  }
  if (Array.isArray(node)) {
    for (const entry of node) {
      const value = textValue(entry);
      if (value !== null) {
        return value;
      }
    }
    return null;
  }

  const record = node as Record<string, XmlNode>;
  if (typeof record['#text'] === 'string' || typeof record['#text'] === 'number') {
    return String(record['#text']);
  }
  return null;
}

function parseRect(node: XmlNode, label: string): Rect | null {
  if (!node) {
    return null;
  }
  const element = asRecord(node, label);
  return {
    xOff: requireFiniteNumber(typeof element.xOff === 'string' ? element.xOff : textValue(element.xOff), `${label} xOff`),
    yOff: requireFiniteNumber(typeof element.yOff === 'string' ? element.yOff : textValue(element.yOff), `${label} yOff`),
    xSize: requireFiniteNumber(typeof element.xSize === 'string' ? element.xSize : textValue(element.xSize), `${label} xSize`),
    ySize: requireFiniteNumber(typeof element.ySize === 'string' ? element.ySize : textValue(element.ySize), `${label} ySize`),
  };
}

function validateAffineTransform(
  transform: AffineTransform,
  width: number,
  height: number,
  label: string,
): AffineTransform {
  const [a, b, c, d, e, f] = transform;
  if (![a, b, c, d, e, f].every((value) => Number.isFinite(value))) {
    throw new Error(`${label} does not expose a browser-readable affine transform.`);
  }
  if (Math.abs(b) > FLOAT_EPSILON || Math.abs(d) > FLOAT_EPSILON) {
    throw new Error(
      'The browser converter only supports north-up terrain rasters. Rotated or sheared transforms are not supported.',
    );
  }
  if (Math.abs(a) <= FLOAT_EPSILON || Math.abs(e) <= FLOAT_EPSILON) {
    throw new Error(`${label} has a zero pixel size, which is not supported.`);
  }

  const xSpan = Math.abs(a) * width;
  const ySpan = Math.abs(e) * height;
  if (!Number.isFinite(xSpan) || !Number.isFinite(ySpan) || xSpan <= FLOAT_EPSILON || ySpan <= FLOAT_EPSILON) {
    throw new Error(`${label} exposes an invalid raster extent in the browser converter.`);
  }

  return transform;
}

function parseGeoTransform(rawValue: string | null, width: number, height: number): AffineTransform {
  if (rawValue === null) {
    throw new Error('The VRT is missing a GeoTransform.');
  }

  const values = rawValue.split(',').map((part) => Number(part.trim()));
  if (values.length !== 6 || values.some((value) => !Number.isFinite(value))) {
    throw new Error('The VRT GeoTransform must contain exactly 6 finite numeric values.');
  }

  const [c, a, b, f, d, e] = values as AffineTransform;
  return validateAffineTransform([a, b, c, d, e, f], width, height, 'The VRT');
}

function parseVrt(xmlText: string): VrtDataset {
  let parsed: Record<string, XmlNode>;
  try {
    parsed = asRecord(xmlParser.parse(xmlText), 'VRT XML');
  } catch {
    throw new Error('The VRT XML could not be parsed.');
  }

  const rootNode = parsed.VRTDataset;
  if (!rootNode) {
    throw new Error('Unsupported VRT root element.');
  }
  const root = asRecord(rootNode, 'VRTDataset');

  const width = requireFiniteNumber(textValue(root.rasterXSize), 'VRT rasterXSize');
  const height = requireFiniteNumber(textValue(root.rasterYSize), 'VRT rasterYSize');
  const geoTransform = parseGeoTransform(textValue(root.GeoTransform), width, height);

  const bandNodes = asArray(root.VRTRasterBand);
  if (bandNodes.length !== 1) {
    throw new Error('The browser converter only supports one-band terrain VRT datasets.');
  }
  const bandNode = asRecord(bandNodes[0], 'VRTRasterBand');
  const subclass = textValue(bandNode.subClass)?.trim();
  if (subclass && subclass !== 'VRTRasterBand') {
    throw new Error(`Unsupported VRTRasterBand subclass in the browser converter: ${subclass}`);
  }

  const bandNoData = parseOptionalNumber(textValue(bandNode.NoDataValue));

  const simpleSources = asArray(bandNode.SimpleSource).map((node) => ({
    kind: 'SimpleSource' as const,
    node: asRecord(node, 'SimpleSource'),
  }));
  const complexSources = asArray(bandNode.ComplexSource).map((node) => ({
    kind: 'ComplexSource' as const,
    node: asRecord(node, 'ComplexSource'),
  }));
  const sourceNodes = [...simpleSources, ...complexSources];
  if (sourceNodes.length === 0) {
    throw new Error('The browser converter requires at least one SimpleSource or ComplexSource in the VRT.');
  }

  const unsupportedNodes = supportedKeys(bandNode, [
    'band',
    'dataType',
    'subClass',
    'Metadata',
    'ColorInterp',
    'NoDataValue',
    'Histograms',
    'SimpleSource',
    'ComplexSource',
  ]);
  if (unsupportedNodes.length > 0) {
    throw new Error(
      `Unsupported VRT band content in the browser converter: ${unsupportedNodes.join(', ')}`,
    );
  }

  const sources: VrtSource[] = sourceNodes.map(({ kind, node }) => {
    const sourceFilename = textValue(node.SourceFilename)?.trim();
    if (!sourceFilename) {
      throw new Error('A VRT source is missing SourceFilename.');
    }

    const sourceBand = parseOptionalNumber(textValue(node.SourceBand)) ?? 1;
    if (sourceBand !== 1) {
      throw new Error(`The browser converter only supports SourceBand 1. Found SourceBand ${sourceBand}.`);
    }

    const unsupportedSourceNodes = supportedKeys(node, [
      'SourceFilename',
      'SourceBand',
      'SourceProperties',
      'SrcRect',
      'DstRect',
      'NODATA',
    ]);
    if (unsupportedSourceNodes.length > 0) {
      throw new Error(
        `Unsupported VRT source content in the browser converter: ${unsupportedSourceNodes.join(', ')}`,
      );
    }

    return {
      kind,
      sourceFilename,
      sourceBand,
      srcRect: parseRect(node.SrcRect, `${kind} SrcRect`),
      dstRect: parseRect(node.DstRect, `${kind} DstRect`),
      sourceNoData: parseOptionalNumber(textValue(node.NODATA)),
    };
  });

  return {
    width,
    height,
    geoTransform,
    bandNoData,
    sources,
  };
}

function buildFileLookup(files: UploadFilePayload[]): FileLookup {
  const byLowerName = new Map<string, UploadFilePayload[]>();
  const byLowerBasename = new Map<string, UploadFilePayload[]>();

  for (const file of files) {
    const nameKey = lower(file.name);
    const basenameKey = lower(basename(file.name));
    const sameName = byLowerName.get(nameKey);
    if (sameName) {
      sameName.push(file);
    } else {
      byLowerName.set(nameKey, [file]);
    }

    const sameBasename = byLowerBasename.get(basenameKey);
    if (sameBasename) {
      sameBasename.push(file);
    } else {
      byLowerBasename.set(basenameKey, [file]);
    }
  }

  return { byLowerName, byLowerBasename };
}

function resolveSingleFile(matches: UploadFilePayload[] | undefined, missingMessage: string, duplicateLabel: string): UploadFilePayload {
  if (!matches || matches.length === 0) {
    throw new Error(missingMessage);
  }
  if (matches.length > 1) {
    throw new Error(`${duplicateLabel}: ${matches.map((file) => file.name).join(', ')}`);
  }
  return matches[0];
}

function resolveSiblingRasterUpload(files: UploadFilePayload[], hdfName: string): UploadFilePayload {
  const lookup = buildFileLookup(files);
  const lowerStem = lower(hdfName.replace(/\.hdf$/i, ''));

  const vrt = resolveSingleFile(
    lookup.byLowerName.get(`${lowerStem}.vrt`),
    '',
    'Multiple uploaded VRT files match the selected HDF',
  );
  if (vrt) {
    return vrt;
  }

  return resolveSingleFile(
    [
      ...(lookup.byLowerName.get(`${lowerStem}.tif`) ?? []),
      ...(lookup.byLowerName.get(`${lowerStem}.tiff`) ?? []),
    ],
    `Could not find a sibling raster beside ${hdfName}. Expected ${hdfName.replace(/\.hdf$/i, '.vrt')}, ${hdfName.replace(/\.hdf$/i, '.tif')}, or ${hdfName.replace(/\.hdf$/i, '.tiff')}.`,
    'Multiple uploaded TIFF files match the selected HDF',
  );
}

function getOptionalSiblingRasterUpload(files: UploadFilePayload[], hdfName: string): UploadFilePayload | null {
  const lookup = buildFileLookup(files);
  const lowerStem = lower(hdfName.replace(/\.hdf$/i, ''));
  const vrtMatches = lookup.byLowerName.get(`${lowerStem}.vrt`);
  if (vrtMatches?.length) {
    return resolveSingleFile(vrtMatches, '', 'Multiple uploaded VRT files match the selected HDF');
  }

  const tifMatches = [
    ...(lookup.byLowerName.get(`${lowerStem}.tif`) ?? []),
    ...(lookup.byLowerName.get(`${lowerStem}.tiff`) ?? []),
  ];
  if (tifMatches?.length) {
    return resolveSingleFile(tifMatches, '', 'Multiple uploaded TIFF files match the selected HDF');
  }

  return null;
}

function resolveVrtSourceUpload(sourceFilename: string, files: UploadFilePayload[]): UploadFilePayload {
  const lookup = buildFileLookup(files);
  const basenameKey = lower(basename(sourceFilename));
  const basenameMatches = lookup.byLowerBasename.get(basenameKey);
  if (basenameMatches && basenameMatches.length === 1) {
    return basenameMatches[0];
  }
  if (basenameMatches && basenameMatches.length > 1) {
    throw new Error(
      `Multiple uploaded files match the VRT source basename ${basename(sourceFilename)}: ${basenameMatches
        .map((file) => file.name)
        .join(', ')}`,
    );
  }

  const exactMatches = lookup.byLowerName.get(lower(sourceFilename));
  return resolveSingleFile(
    exactMatches,
    `The VRT references a raster source that was not uploaded: ${sourceFilename}`,
    `Multiple uploaded files match the VRT source path ${sourceFilename}`,
  );
}

async function decodeGeoTiff(file: UploadFilePayload): Promise<DecodedGeoTiff> {
  const tiff = await fromArrayBuffer(file.bytes.slice(0));
  const image = await tiff.getImage();
  if (image.getSamplesPerPixel() < 1) {
    throw new Error(`${file.name} does not contain any raster bands.`);
  }

  const origin = image.getOrigin();
  const resolution = image.getResolution();
  const transform = validateAffineTransform(
    [
      Number(resolution[0]),
      0,
      Number(origin[0]),
      0,
      Number(resolution[1]),
      Number(origin[1]),
    ],
    image.getWidth(),
    image.getHeight(),
    file.name,
  );

  const nodata = image.getGDALNoData();
  const rasterResult = await image.readRasters({
    samples: [0],
    interleave: false,
  });
  const sourceBand = rasterResult[0];
  const elevations = new Float32Array(sourceBand.length);
  const validMask = new Uint8Array(sourceBand.length);
  let maxElevation = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < sourceBand.length; index += 1) {
    const value = Number(sourceBand[index]);
    elevations[index] = value;
    const valid =
      Number.isFinite(value) &&
      (nodata === null || Math.abs(value - nodata) > FLOAT_EPSILON);
    if (valid) {
      validMask[index] = 1;
      if (value > maxElevation) {
        maxElevation = value;
      }
    } else {
      elevations[index] = Number.NaN;
    }
  }

  if (!Number.isFinite(maxElevation)) {
    throw new Error(`${file.name} does not contain any valid elevation cells.`);
  }

  return {
    fileName: file.name,
    width: image.getWidth(),
    height: image.getHeight(),
    transform,
    elevations,
    validMask,
    maxElevation,
    nodata,
  };
}

function validateRect(rect: Rect, width: number, height: number, label: string): void {
  if (
    rect.xOff < 0 ||
    rect.yOff < 0 ||
    rect.xSize <= 0 ||
    rect.ySize <= 0 ||
    rect.xOff + rect.xSize > width ||
    rect.yOff + rect.ySize > height
  ) {
    throw new Error(`${label} falls outside the raster extent.`);
  }
}

function applySource(
  destinationElevations: Float32Array,
  destinationMask: Uint8Array,
  destinationWidth: number,
  source: DecodedGeoTiff,
  sourceSpec: VrtSource,
  bandNoData: number | null,
): void {
  const srcRect: Rect = sourceSpec.srcRect ?? {
    xOff: 0,
    yOff: 0,
    xSize: source.width,
    ySize: source.height,
  };
  const dstRect: Rect = sourceSpec.dstRect ?? {
    xOff: 0,
    yOff: 0,
    xSize: source.width,
    ySize: source.height,
  };

  validateRect(srcRect, source.width, source.height, `The VRT SrcRect for ${source.fileName}`);
  validateRect(dstRect, destinationWidth, destinationElevations.length / destinationWidth, `The VRT DstRect for ${source.fileName}`);

  const sourceNoData = sourceSpec.sourceNoData;
  const sameSize = srcRect.xSize === dstRect.xSize && srcRect.ySize === dstRect.ySize;

  for (let dstY = 0; dstY < dstRect.ySize; dstY += 1) {
    const srcY = sameSize
      ? srcRect.yOff + dstY
      : srcRect.yOff +
        Math.min(
          srcRect.ySize - 1,
          Math.max(
            0,
            Math.floor(((dstY + 0.5) * srcRect.ySize) / dstRect.ySize),
          ),
        );
    const srcRowBase = srcY * source.width;
    const dstRowBase = (dstRect.yOff + dstY) * destinationWidth + dstRect.xOff;

    for (let dstX = 0; dstX < dstRect.xSize; dstX += 1) {
      const srcX = sameSize
        ? srcRect.xOff + dstX
        : srcRect.xOff +
          Math.min(
            srcRect.xSize - 1,
            Math.max(
              0,
              Math.floor(((dstX + 0.5) * srcRect.xSize) / dstRect.xSize),
            ),
          );
      const sourceIndex = srcRowBase + srcX;
      if (source.validMask[sourceIndex] === 0) {
        continue;
      }

      const value = source.elevations[sourceIndex];
      if (
        !Number.isFinite(value) ||
        (sourceNoData !== null && Math.abs(value - sourceNoData) <= FLOAT_EPSILON) ||
        (bandNoData !== null && Math.abs(value - bandNoData) <= FLOAT_EPSILON)
      ) {
        continue;
      }

      const destinationIndex = dstRowBase + dstX;
      destinationElevations[destinationIndex] = value;
      destinationMask[destinationIndex] = 1;
    }
  }
}

async function decodeVrtSurface(
  vrtFile: UploadFilePayload,
  allFiles: UploadFilePayload[],
  onStatus?: (message: string) => void,
  onProgress?: (progress: BrowserRasterProgress) => void,
): Promise<BrowserRasterSurface> {
  const report = (fraction: number, message: string): void => {
    const clampedFraction = Math.min(1, Math.max(0, fraction));
    onStatus?.(message);
    onProgress?.({ fraction: clampedFraction, message });
  };

  report(0, `Parsing terrain VRT ${vrtFile.name}...`);
  const xmlText = new TextDecoder().decode(vrtFile.bytes);
  const vrt = parseVrt(xmlText);
  const cellCount = vrt.width * vrt.height;
  const elevations = new Float32Array(cellCount);
  elevations.fill(Number.NaN);
  const validMask = new Uint8Array(cellCount);
  const sourceCache = new Map<string, Promise<DecodedGeoTiff>>();
  const totalSources = vrt.sources.length;

  for (let sourceIndex = 0; sourceIndex < vrt.sources.length; sourceIndex += 1) {
    const source = vrt.sources[sourceIndex];
    const uploadedSource = resolveVrtSourceUpload(source.sourceFilename, allFiles);
    let decoded = sourceCache.get(uploadedSource.name);
    if (!decoded) {
      report(
        sourceIndex / totalSources,
        `Reading GeoTIFF source ${uploadedSource.name} (${sourceIndex + 1}/${totalSources})...`,
      );
      decoded = decodeGeoTiff(uploadedSource);
      sourceCache.set(uploadedSource.name, decoded);
    }
    applySource(
      elevations,
      validMask,
      vrt.width,
      await decoded,
      source,
      vrt.bandNoData,
    );
    report(
      (sourceIndex + 1) / totalSources,
      `Composed raster source ${sourceIndex + 1}/${totalSources}: ${uploadedSource.name}`,
    );
  }

  let maxElevation = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < cellCount; index += 1) {
    if (validMask[index] === 0) {
      elevations[index] = Number.NaN;
      continue;
    }
    const value = elevations[index];
    if (value > maxElevation) {
      maxElevation = value;
    }
  }

  if (!Number.isFinite(maxElevation)) {
    throw new Error(`${vrtFile.name} does not resolve to any valid elevation cells in the browser converter.`);
  }

  report(1, `Resolved terrain raster ${vrtFile.name}.`);

  return {
    resolvedRasterName: vrtFile.name,
    width: vrt.width,
    height: vrt.height,
    transform: vrt.geoTransform,
    elevations,
    validMask,
    maxElevation,
  };
}

function resolveTerrainRasterUpload(files: UploadFilePayload[], hdfName: string): UploadFilePayload {
  const maybeRaster = getOptionalSiblingRasterUpload(files, hdfName);
  if (maybeRaster) {
    return maybeRaster;
  }

  throw new Error(
    `Could not find a sibling raster beside ${hdfName}. Expected ${hdfName.replace(/\.hdf$/i, '.vrt')}, ${hdfName.replace(/\.hdf$/i, '.tif')}, or ${hdfName.replace(/\.hdf$/i, '.tiff')}.`,
  );
}

function resolveDirectRasterUpload(files: UploadFilePayload[], terrainName: string): UploadFilePayload {
  const lookup = buildFileLookup(files);
  const rasterUpload = resolveSingleFile(
    lookup.byLowerName.get(lower(terrainName)),
    `The selected DEM GeoTIFF was not uploaded: ${terrainName}`,
    `Multiple uploaded DEM GeoTIFF files match ${terrainName}`,
  );
  if (!isTiffName(rasterUpload.name)) {
    throw new Error(`${terrainName} is not a supported DEM GeoTIFF. Use .tif or .tiff.`);
  }
  return rasterUpload;
}

export async function decodeTerrainRasterSurface(
  files: UploadFilePayload[],
  terrainName: string,
  inputKind: BrowserTerrainInputKind,
  onStatus?: (message: string) => void,
  onProgress?: (progress: BrowserRasterProgress) => void,
): Promise<BrowserRasterSurface> {
  const rasterUpload =
    inputKind === 'hdf'
      ? resolveTerrainRasterUpload(files, terrainName)
      : resolveDirectRasterUpload(files, terrainName);
  if (lower(rasterUpload.name).endsWith('.vrt')) {
    return decodeVrtSurface(rasterUpload, files, onStatus, onProgress);
  }

  onStatus?.(`Reading GeoTIFF raster ${rasterUpload.name}...`);
  onProgress?.({ fraction: 0, message: `Reading GeoTIFF raster ${rasterUpload.name}...` });
  const decoded = await decodeGeoTiff(rasterUpload);
  onProgress?.({ fraction: 1, message: `Resolved terrain raster ${rasterUpload.name}.` });
  return {
    resolvedRasterName: rasterUpload.name,
    width: decoded.width,
    height: decoded.height,
    transform: decoded.transform,
    elevations: decoded.elevations,
    validMask: decoded.validMask,
    maxElevation: decoded.maxElevation,
  };
}
