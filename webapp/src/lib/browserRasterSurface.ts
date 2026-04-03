import { XMLParser } from 'fast-xml-parser';
import { fromArrayBuffer } from 'geotiff';
import { buildSampleIndices } from './browserTerrainLimits';
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

export type AffineTransform = [number, number, number, number, number, number];

type GeoTiffHeader = {
  fileName: string;
  width: number;
  height: number;
  transform: AffineTransform;
  nodata: number | null;
};

type OpenedGeoTiff = GeoTiffHeader & {
  image: {
    readRasters(options: {
      samples: number[];
      interleave: false;
      window?: [number, number, number, number];
    }): Promise<ArrayLike<number>[]>;
  };
};

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
  kind?: 'full';
  resolvedRasterName: string;
  width: number;
  height: number;
  transform: AffineTransform;
  elevations: Float32Array;
  validMask: Uint8Array;
  maxElevation: number;
};

export type BrowserRefinementRegion = {
  rowStart: number;
  rowEnd: number;
  colStart: number;
  colEnd: number;
};

export type BrowserSparseRefinementTile = BrowserRefinementRegion & {
  elevations: Float32Array;
  validMask: Uint8Array;
};

export type BrowserSparseRasterSurface = {
  kind: 'sparse';
  resolvedRasterName: string;
  width: number;
  height: number;
  transform: AffineTransform;
  sampledRows: Int32Array;
  sampledCols: Int32Array;
  coarseElevations: Float32Array;
  coarseValidMask: Uint8Array;
  refinementTiles: BrowserSparseRefinementTile[];
  maxElevation: number;
};

export type BrowserTerrainSurface = BrowserRasterSurface | BrowserSparseRasterSurface;

export type BrowserRasterProbe = {
  resolvedRasterName: string;
  width: number;
  height: number;
  transform: AffineTransform;
  totalInputBytes: number;
  largestSourceCellCount: number;
  largestSourceWidth: number;
  targetCellCount: number;
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

export function parseVrt(xmlText: string): VrtDataset {
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

function totalInputBytes(files: UploadFilePayload[]): number {
  return files.reduce((total, file) => total + file.bytes.byteLength, 0);
}

async function readGeoTiffHeader(file: UploadFilePayload): Promise<GeoTiffHeader> {
  const tiff = await fromArrayBuffer(file.bytes);
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

  return {
    fileName: file.name,
    width: image.getWidth(),
    height: image.getHeight(),
    transform,
    nodata: image.getGDALNoData(),
  };
}

async function openGeoTiff(file: UploadFilePayload): Promise<OpenedGeoTiff> {
  const tiff = await fromArrayBuffer(file.bytes);
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

  return {
    fileName: file.name,
    width: image.getWidth(),
    height: image.getHeight(),
    transform,
    nodata: image.getGDALNoData(),
    image,
  };
}

function isValidElevation(
  value: number,
  nodata: number | null,
  sourceNoData: number | null = null,
  bandNoData: number | null = null,
): boolean {
  return (
    Number.isFinite(value) &&
    (nodata === null || Math.abs(value - nodata) > FLOAT_EPSILON) &&
    (sourceNoData === null || Math.abs(value - sourceNoData) > FLOAT_EPSILON) &&
    (bandNoData === null || Math.abs(value - bandNoData) > FLOAT_EPSILON)
  );
}

async function readGeoTiffWindow(
  source: OpenedGeoTiff,
  window: Rect,
): Promise<ArrayLike<number>> {
  const rasterResult = await source.image.readRasters({
    samples: [0],
    interleave: false,
    window: [window.xOff, window.yOff, window.xOff + window.xSize, window.yOff + window.ySize],
  });
  return rasterResult[0];
}

function mapDestinationOffsetToSourceOffset(
  destinationOffset: number,
  sourceSize: number,
  destinationSize: number,
  sameSize: boolean,
): number {
  if (sameSize) {
    return destinationOffset;
  }

  return Math.min(
    sourceSize - 1,
    Math.max(
      0,
      Math.floor(((destinationOffset + 0.5) * sourceSize) / destinationSize),
    ),
  );
}

function createEmptyTile(region: BrowserRefinementRegion): BrowserSparseRefinementTile {
  const tileWidth = (region.colEnd - region.colStart) + 1;
  const tileHeight = (region.rowEnd - region.rowStart) + 1;
  const elevations = new Float32Array(tileWidth * tileHeight);
  elevations.fill(Number.NaN);
  return {
    ...region,
    elevations,
    validMask: new Uint8Array(tileWidth * tileHeight),
  };
}

async function decodeGeoTiff(file: UploadFilePayload): Promise<DecodedGeoTiff> {
  const tiff = await fromArrayBuffer(file.bytes);
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

async function probeVrtSurface(
  vrtFile: UploadFilePayload,
  allFiles: UploadFilePayload[],
  onStatus?: (message: string) => void,
  onProgress?: (progress: BrowserRasterProgress) => void,
): Promise<BrowserRasterProbe> {
  const report = (fraction: number, message: string): void => {
    const clampedFraction = Math.min(1, Math.max(0, fraction));
    onStatus?.(message);
    onProgress?.({ fraction: clampedFraction, message });
  };

  report(0, `Parsing terrain VRT header ${vrtFile.name}...`);
  const xmlText = new TextDecoder().decode(vrtFile.bytes);
  const vrt = parseVrt(xmlText);
  const sourceCache = new Map<string, Promise<GeoTiffHeader>>();
  const totalSources = vrt.sources.length;
  let largestSourceCellCount = 0;
  let largestSourceWidth = 0;

  for (let sourceIndex = 0; sourceIndex < vrt.sources.length; sourceIndex += 1) {
    const source = vrt.sources[sourceIndex];
    const uploadedSource = resolveVrtSourceUpload(source.sourceFilename, allFiles);
    let header = sourceCache.get(uploadedSource.name);
    if (!header) {
      report(
        sourceIndex / totalSources,
        `Inspecting GeoTIFF header ${uploadedSource.name} (${sourceIndex + 1}/${totalSources})...`,
      );
      header = readGeoTiffHeader(uploadedSource);
      sourceCache.set(uploadedSource.name, header);
    }

    const resolvedHeader = await header;
    largestSourceCellCount = Math.max(
      largestSourceCellCount,
      resolvedHeader.width * resolvedHeader.height,
    );
    largestSourceWidth = Math.max(largestSourceWidth, resolvedHeader.width);
    report(
      (sourceIndex + 1) / totalSources,
      `Resolved raster header ${sourceIndex + 1}/${totalSources}: ${uploadedSource.name}`,
    );
  }

  return {
    kind: 'full',
    resolvedRasterName: vrtFile.name,
    width: vrt.width,
    height: vrt.height,
    transform: vrt.geoTransform,
    totalInputBytes: totalInputBytes(allFiles),
    largestSourceCellCount,
    largestSourceWidth,
    targetCellCount: vrt.width * vrt.height,
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

export async function probeTerrainRaster(
  files: UploadFilePayload[],
  terrainName: string,
  inputKind: BrowserTerrainInputKind,
  onStatus?: (message: string) => void,
  onProgress?: (progress: BrowserRasterProgress) => void,
): Promise<BrowserRasterProbe> {
  const rasterUpload =
    inputKind === 'hdf'
      ? resolveTerrainRasterUpload(files, terrainName)
      : resolveDirectRasterUpload(files, terrainName);

  if (lower(rasterUpload.name).endsWith('.vrt')) {
    return probeVrtSurface(rasterUpload, files, onStatus, onProgress);
  }

  onStatus?.(`Reading GeoTIFF header ${rasterUpload.name}...`);
  onProgress?.({ fraction: 0, message: `Reading GeoTIFF header ${rasterUpload.name}...` });
  const header = await readGeoTiffHeader(rasterUpload);
  onProgress?.({ fraction: 1, message: `Resolved terrain raster header ${rasterUpload.name}.` });
  return {
    resolvedRasterName: rasterUpload.name,
    width: header.width,
    height: header.height,
    transform: header.transform,
    totalInputBytes: totalInputBytes(files),
    largestSourceCellCount: header.width * header.height,
    largestSourceWidth: header.width,
    targetCellCount: header.width * header.height,
  };
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
    kind: 'full',
    resolvedRasterName: rasterUpload.name,
    width: decoded.width,
    height: decoded.height,
    transform: decoded.transform,
    elevations: decoded.elevations,
    validMask: decoded.validMask,
    maxElevation: decoded.maxElevation,
  };
}

type IndexedSample = {
  value: number;
  index: number;
};

type VrtTileIntersection = {
  tileIndex: number;
  rowStart: number;
  rowEnd: number;
  colStart: number;
  colEnd: number;
};

type PlannedVrtSource = {
  sourceSpec: VrtSource;
  uploadedSource: UploadFilePayload;
  coarseRows: IndexedSample[];
  coarseColumns: Array<IndexedSample & { sourceCol: number }>;
  tileIntersections: VrtTileIntersection[];
};

function createIndexedSamples(
  values: Int32Array,
  startInclusive: number,
  endExclusive: number,
): IndexedSample[] {
  const result: IndexedSample[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value >= startInclusive && value < endExclusive) {
      result.push({ value, index });
    }
  }
  return result;
}

function computeSparseMaxElevation(
  coarseElevations: Float32Array,
  coarseValidMask: Uint8Array,
  refinementTiles: BrowserSparseRefinementTile[],
): number {
  let maxElevation = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < coarseElevations.length; index += 1) {
    if (coarseValidMask[index] === 0) {
      continue;
    }
    const value = coarseElevations[index];
    if (value > maxElevation) {
      maxElevation = value;
    }
  }

  for (const tile of refinementTiles) {
    for (let index = 0; index < tile.elevations.length; index += 1) {
      if (tile.validMask[index] === 0) {
        continue;
      }
      const value = tile.elevations[index];
      if (value > maxElevation) {
        maxElevation = value;
      }
    }
  }

  return maxElevation;
}

function resolveVrtRects(
  source: OpenedGeoTiff,
  sourceSpec: VrtSource,
  destinationWidth: number,
  destinationHeight: number,
): { srcRect: Rect; dstRect: Rect; sameSize: boolean } {
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
  validateRect(dstRect, destinationWidth, destinationHeight, `The VRT DstRect for ${source.fileName}`);

  return {
    srcRect,
    dstRect,
    sameSize: srcRect.xSize === dstRect.xSize && srcRect.ySize === dstRect.ySize,
  };
}

async function decodeSparseGeoTiffSurface(
  rasterUpload: UploadFilePayload,
  sampleStep: number,
  refinementRegions: BrowserRefinementRegion[],
  onStatus?: (message: string) => void,
  onProgress?: (progress: BrowserRasterProgress) => void,
): Promise<BrowserSparseRasterSurface> {
  const report = (fraction: number, message: string): void => {
    const clampedFraction = Math.min(1, Math.max(0, fraction));
    onStatus?.(message);
    onProgress?.({ fraction: clampedFraction, message });
  };

  report(0, `Reading sparse GeoTIFF samples from ${rasterUpload.name}...`);
  const source = await openGeoTiff(rasterUpload);
  const sampledRows = Int32Array.from(buildSampleIndices(source.height, sampleStep));
  const sampledCols = Int32Array.from(buildSampleIndices(source.width, sampleStep));
  const coarseElevations = new Float32Array(sampledRows.length * sampledCols.length);
  coarseElevations.fill(Number.NaN);
  const coarseValidMask = new Uint8Array(coarseElevations.length);
  const refinementTiles = refinementRegions.map(createEmptyTile);
  const totalWork = Math.max(1, sampledRows.length + refinementTiles.length);
  const progressInterval = Math.max(1, Math.floor(totalWork / 100));
  let completedWork = 0;

  for (let rowIndex = 0; rowIndex < sampledRows.length; rowIndex += 1) {
    const row = sampledRows[rowIndex];
    const rowValues = await readGeoTiffWindow(source, {
      xOff: 0,
      yOff: row,
      xSize: source.width,
      ySize: 1,
    });
    const rowBase = rowIndex * sampledCols.length;
    for (let colIndex = 0; colIndex < sampledCols.length; colIndex += 1) {
      const value = Number(rowValues[sampledCols[colIndex]]);
      const coarseIndex = rowBase + colIndex;
      if (isValidElevation(value, source.nodata)) {
        coarseElevations[coarseIndex] = value;
        coarseValidMask[coarseIndex] = 1;
      }
    }

    completedWork += 1;
    if (completedWork === 1 || completedWork === totalWork || completedWork % progressInterval === 0) {
      report(
        completedWork / totalWork,
        `Read sparse sampled row ${rowIndex + 1}/${sampledRows.length} from ${rasterUpload.name}...`,
      );
    }
  }

  for (let tileIndex = 0; tileIndex < refinementTiles.length; tileIndex += 1) {
    const tile = refinementTiles[tileIndex];
    const tileWidth = (tile.colEnd - tile.colStart) + 1;
    const tileHeight = (tile.rowEnd - tile.rowStart) + 1;
    const tileValues = await readGeoTiffWindow(source, {
      xOff: tile.colStart,
      yOff: tile.rowStart,
      xSize: tileWidth,
      ySize: tileHeight,
    });

    for (let index = 0; index < tileValues.length; index += 1) {
      const value = Number(tileValues[index]);
      if (isValidElevation(value, source.nodata)) {
        tile.elevations[index] = value;
        tile.validMask[index] = 1;
      }
    }

    completedWork += 1;
    if (completedWork === 1 || completedWork === totalWork || completedWork % progressInterval === 0) {
      report(
        completedWork / totalWork,
        `Read sparse refinement tile ${tileIndex + 1}/${refinementTiles.length} from ${rasterUpload.name}...`,
      );
    }
  }

  const maxElevation = computeSparseMaxElevation(coarseElevations, coarseValidMask, refinementTiles);
  if (!Number.isFinite(maxElevation)) {
    throw new Error(`${rasterUpload.name} does not contain any valid elevation cells.`);
  }

  report(1, `Resolved sparse terrain raster ${rasterUpload.name}.`);
  return {
    kind: 'sparse',
    resolvedRasterName: rasterUpload.name,
    width: source.width,
    height: source.height,
    transform: source.transform,
    sampledRows,
    sampledCols,
    coarseElevations,
    coarseValidMask,
    refinementTiles,
    maxElevation,
  };
}

async function decodeSparseVrtSurface(
  vrtFile: UploadFilePayload,
  allFiles: UploadFilePayload[],
  sampleStep: number,
  refinementRegions: BrowserRefinementRegion[],
  onStatus?: (message: string) => void,
  onProgress?: (progress: BrowserRasterProgress) => void,
): Promise<BrowserSparseRasterSurface> {
  const report = (fraction: number, message: string): void => {
    const clampedFraction = Math.min(1, Math.max(0, fraction));
    onStatus?.(message);
    onProgress?.({ fraction: clampedFraction, message });
  };

  report(0, `Parsing sparse terrain VRT ${vrtFile.name}...`);
  const xmlText = new TextDecoder().decode(vrtFile.bytes);
  const vrt = parseVrt(xmlText);
  const sampledRows = Int32Array.from(buildSampleIndices(vrt.height, sampleStep));
  const sampledCols = Int32Array.from(buildSampleIndices(vrt.width, sampleStep));
  const coarseElevations = new Float32Array(sampledRows.length * sampledCols.length);
  coarseElevations.fill(Number.NaN);
  const coarseValidMask = new Uint8Array(coarseElevations.length);
  const refinementTiles = refinementRegions.map(createEmptyTile);
  const sourceCache = new Map<string, Promise<OpenedGeoTiff>>();
  const sourcePlans: PlannedVrtSource[] = [];

  for (const sourceSpec of vrt.sources) {
    const uploadedSource = resolveVrtSourceUpload(sourceSpec.sourceFilename, allFiles);
    let opened = sourceCache.get(uploadedSource.name);
    if (!opened) {
      opened = openGeoTiff(uploadedSource);
      sourceCache.set(uploadedSource.name, opened);
    }
    const resolvedSource = await opened;
    const { srcRect, dstRect, sameSize } = resolveVrtRects(resolvedSource, sourceSpec, vrt.width, vrt.height);
    const coarseRows = createIndexedSamples(sampledRows, dstRect.yOff, dstRect.yOff + dstRect.ySize);
    const coarseColumns = createIndexedSamples(sampledCols, dstRect.xOff, dstRect.xOff + dstRect.xSize)
      .map((entry) => ({
        ...entry,
        sourceCol: srcRect.xOff + mapDestinationOffsetToSourceOffset(
          entry.value - dstRect.xOff,
          srcRect.xSize,
          dstRect.xSize,
          sameSize,
        ),
      }));
    const tileIntersections: VrtTileIntersection[] = [];
    for (let tileIndex = 0; tileIndex < refinementTiles.length; tileIndex += 1) {
      const tile = refinementTiles[tileIndex];
      const rowStart = Math.max(tile.rowStart, dstRect.yOff);
      const rowEnd = Math.min(tile.rowEnd, (dstRect.yOff + dstRect.ySize) - 1);
      const colStart = Math.max(tile.colStart, dstRect.xOff);
      const colEnd = Math.min(tile.colEnd, (dstRect.xOff + dstRect.xSize) - 1);
      if (rowStart <= rowEnd && colStart <= colEnd) {
        tileIntersections.push({
          tileIndex,
          rowStart,
          rowEnd,
          colStart,
          colEnd,
        });
      }
    }

    sourcePlans.push({
      sourceSpec,
      uploadedSource,
      coarseRows,
      coarseColumns,
      tileIntersections,
    });
  }

  const totalWork = Math.max(
    1,
    sourcePlans.reduce(
      (sum, plan) => sum + plan.coarseRows.length + plan.tileIntersections.length,
      0,
    ),
  );
  const progressInterval = Math.max(1, Math.floor(totalWork / 100));
  let completedWork = 0;

  for (let sourceIndex = 0; sourceIndex < sourcePlans.length; sourceIndex += 1) {
    const plan = sourcePlans[sourceIndex];
    const source = await (sourceCache.get(plan.uploadedSource.name) as Promise<OpenedGeoTiff>);
    const { srcRect, dstRect, sameSize } = resolveVrtRects(source, plan.sourceSpec, vrt.width, vrt.height);
    const minSourceCol = plan.coarseColumns.length > 0
      ? Math.min(...plan.coarseColumns.map((entry) => entry.sourceCol))
      : 0;
    const maxSourceCol = plan.coarseColumns.length > 0
      ? Math.max(...plan.coarseColumns.map((entry) => entry.sourceCol))
      : -1;

    for (let rowPlanIndex = 0; rowPlanIndex < plan.coarseRows.length; rowPlanIndex += 1) {
      const rowEntry = plan.coarseRows[rowPlanIndex];
      const sourceRow = srcRect.yOff + mapDestinationOffsetToSourceOffset(
        rowEntry.value - dstRect.yOff,
        srcRect.ySize,
        dstRect.ySize,
        sameSize,
      );
      const rowValues = maxSourceCol >= minSourceCol
        ? await readGeoTiffWindow(source, {
          xOff: minSourceCol,
          yOff: sourceRow,
          xSize: (maxSourceCol - minSourceCol) + 1,
          ySize: 1,
        })
        : null;

      for (let colPlanIndex = 0; colPlanIndex < plan.coarseColumns.length; colPlanIndex += 1) {
        const colEntry = plan.coarseColumns[colPlanIndex];
        const destinationIndex = (rowEntry.index * sampledCols.length) + colEntry.index;
        const value = rowValues === null ? Number.NaN : Number(rowValues[colEntry.sourceCol - minSourceCol]);
        if (isValidElevation(value, source.nodata, plan.sourceSpec.sourceNoData, vrt.bandNoData)) {
          coarseElevations[destinationIndex] = value;
          coarseValidMask[destinationIndex] = 1;
        }
      }

      completedWork += 1;
      if (completedWork === 1 || completedWork === totalWork || completedWork % progressInterval === 0) {
        report(
          completedWork / totalWork,
          `Composed sparse sampled row ${rowPlanIndex + 1}/${plan.coarseRows.length} from source ${sourceIndex + 1}/${sourcePlans.length}: ${plan.uploadedSource.name}`,
        );
      }
    }

    for (let intersectionIndex = 0; intersectionIndex < plan.tileIntersections.length; intersectionIndex += 1) {
      const intersection = plan.tileIntersections[intersectionIndex];
      const tile = refinementTiles[intersection.tileIndex];
      const sourceRowStart = srcRect.yOff + mapDestinationOffsetToSourceOffset(
        intersection.rowStart - dstRect.yOff,
        srcRect.ySize,
        dstRect.ySize,
        sameSize,
      );
      const sourceRowEnd = srcRect.yOff + mapDestinationOffsetToSourceOffset(
        intersection.rowEnd - dstRect.yOff,
        srcRect.ySize,
        dstRect.ySize,
        sameSize,
      );
      const sourceColStart = srcRect.xOff + mapDestinationOffsetToSourceOffset(
        intersection.colStart - dstRect.xOff,
        srcRect.xSize,
        dstRect.xSize,
        sameSize,
      );
      const sourceColEnd = srcRect.xOff + mapDestinationOffsetToSourceOffset(
        intersection.colEnd - dstRect.xOff,
        srcRect.xSize,
        dstRect.xSize,
        sameSize,
      );
      const window = {
        xOff: Math.min(sourceColStart, sourceColEnd),
        yOff: Math.min(sourceRowStart, sourceRowEnd),
        xSize: Math.abs(sourceColEnd - sourceColStart) + 1,
        ySize: Math.abs(sourceRowEnd - sourceRowStart) + 1,
      };
      const tileValues = await readGeoTiffWindow(source, window);
      const tileWidth = (tile.colEnd - tile.colStart) + 1;

      for (let row = intersection.rowStart; row <= intersection.rowEnd; row += 1) {
        const sourceRow = srcRect.yOff + mapDestinationOffsetToSourceOffset(
          row - dstRect.yOff,
          srcRect.ySize,
          dstRect.ySize,
          sameSize,
        );
        for (let col = intersection.colStart; col <= intersection.colEnd; col += 1) {
          const sourceCol = srcRect.xOff + mapDestinationOffsetToSourceOffset(
            col - dstRect.xOff,
            srcRect.xSize,
            dstRect.xSize,
            sameSize,
          );
          const sourceIndexInWindow =
            ((sourceRow - window.yOff) * window.xSize) +
            (sourceCol - window.xOff);
          const destinationIndex =
            ((row - tile.rowStart) * tileWidth) +
            (col - tile.colStart);
          const value = Number(tileValues[sourceIndexInWindow]);
          if (isValidElevation(value, source.nodata, plan.sourceSpec.sourceNoData, vrt.bandNoData)) {
            tile.elevations[destinationIndex] = value;
            tile.validMask[destinationIndex] = 1;
          }
        }
      }

      completedWork += 1;
      if (completedWork === 1 || completedWork === totalWork || completedWork % progressInterval === 0) {
        report(
          completedWork / totalWork,
          `Composed sparse refinement tile ${intersectionIndex + 1}/${plan.tileIntersections.length} from source ${sourceIndex + 1}/${sourcePlans.length}: ${plan.uploadedSource.name}`,
        );
      }
    }
  }

  const maxElevation = computeSparseMaxElevation(coarseElevations, coarseValidMask, refinementTiles);
  if (!Number.isFinite(maxElevation)) {
    throw new Error(`${vrtFile.name} does not resolve to any valid elevation cells in the browser converter.`);
  }

  report(1, `Resolved sparse terrain raster ${vrtFile.name}.`);
  return {
    kind: 'sparse',
    resolvedRasterName: vrtFile.name,
    width: vrt.width,
    height: vrt.height,
    transform: vrt.geoTransform,
    sampledRows,
    sampledCols,
    coarseElevations,
    coarseValidMask,
    refinementTiles,
    maxElevation,
  };
}

export async function decodeTerrainRasterSurfaceForStep(
  files: UploadFilePayload[],
  terrainName: string,
  inputKind: BrowserTerrainInputKind,
  sampleStep: number,
  refinementRegions: BrowserRefinementRegion[],
  onStatus?: (message: string) => void,
  onProgress?: (progress: BrowserRasterProgress) => void,
): Promise<BrowserTerrainSurface> {
  if (sampleStep <= 1) {
    return decodeTerrainRasterSurface(files, terrainName, inputKind, onStatus, onProgress);
  }

  const rasterUpload =
    inputKind === 'hdf'
      ? resolveTerrainRasterUpload(files, terrainName)
      : resolveDirectRasterUpload(files, terrainName);
  if (lower(rasterUpload.name).endsWith('.vrt')) {
    return decodeSparseVrtSurface(
      rasterUpload,
      files,
      sampleStep,
      refinementRegions,
      onStatus,
      onProgress,
    );
  }

  return decodeSparseGeoTiffSurface(
    rasterUpload,
    sampleStep,
    refinementRegions,
    onStatus,
    onProgress,
  );
}
