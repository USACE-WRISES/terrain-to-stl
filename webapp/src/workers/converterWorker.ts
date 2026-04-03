/// <reference lib="webworker" />

import {
  decodeTerrainRasterSurfaceForStep,
  decodeTerrainRasterSurface,
  probeTerrainRaster,
  type BrowserRefinementRegion,
  type BrowserRasterProbe,
  type BrowserRasterSurface,
  type BrowserTerrainSurface,
  type BrowserTerrainInputKind,
} from '../lib/browserRasterSurface';
import {
  type AdaptiveStitchMetrics,
  buildBrowserLimits,
  buildSampleStepOptions,
  BROWSER_PEAK_WORKING_SET_LIMIT_BYTES,
  estimateAdaptiveStlUpperBoundBytes,
  estimatePeakWorkingSetBytes,
  estimateSparsePeakWorkingSetBytes,
  estimateSurfaceStlBytes,
  estimateStlUpperBoundBytes,
  evaluateSampleStep,
  buildSampleIndices,
  SAMPLE_STEP_PRESETS,
} from '../lib/browserTerrainLimits';
import type {
  BrowserLimits,
  ConversionProgress,
  ConversionProgressStep,
  ConversionResult,
  PreparedStlPayload,
  SampleStepOption,
  TerrainInspection,
  UploadFilePayload,
} from '../lib/types';

declare const self: DedicatedWorkerGlobalScope;

type InspectMessage = {
  type: 'inspect';
  files: UploadFilePayload[];
  terrainName: string;
  terrainKind: BrowserTerrainInputKind;
};

type ConvertMessage = {
  type: 'convert';
  files: UploadFilePayload[];
  terrainName: string;
  terrainKind: BrowserTerrainInputKind;
  terrainMaxElevation: number;
  topElevation: number;
  sampleStep: number;
};

type PrepareDownloadMessage = {
  type: 'prepareDownload';
};

type PrepareViewerFileMessage = {
  type: 'prepareViewerFile';
};

type ReleaseDownloadMessage = {
  type: 'releaseDownload';
};

type WorkerMessage =
  | InspectMessage
  | ConvertMessage
  | PrepareDownloadMessage
  | PrepareViewerFileMessage
  | ReleaseDownloadMessage;

type PythonInspection = {
  terrain_max_elevation: number | null;
  resolved_raster_name: string;
  stitch_point_count: number;
  stitch_triangle_count: number;
  has_populated_stitch_tin: boolean;
  stitch_component_count: number;
  adaptive_stitch_metrics: Record<string, PythonAdaptiveStitchMetrics>;
};

type PythonAdaptiveStitchMetrics = {
  coarse_cell_count: number;
  refined_cell_count: number;
  transition_triangle_count: number;
  refinement_perimeter_cell_count: number;
  refined_vertex_count: number;
  largest_refinement_vertex_count: number;
  region_count: number;
};

type PythonSparsePlan = {
  refinement_regions: Array<{
    row_start: number;
    row_end: number;
    col_start: number;
    col_end: number;
  }>;
};

type PythonConversion = {
  output_filename: string;
  terrain_max_elevation: number;
  resolved_raster_name: string;
  triangle_count: number;
  wall_triangle_count: number;
  stitch_point_count: number;
  stitch_triangle_count: number;
  stitch_bridge_triangle_count: number;
  stl_size_bytes: number;
};

type PythonModule = {
  inspect_hdf_metadata(
    sessionDir: string,
    hdfName: string,
    resolvedRasterName: string,
    rasterWidth: number,
    rasterHeight: number,
    rasterTransform: number[],
    rasterMaxElevation?: number | null,
  ): unknown;
  build_hdf_sparse_plan(
    sessionDir: string,
    hdfName: string,
    rasterWidth: number,
    rasterHeight: number,
    rasterTransform: number[],
    sampleStep: number,
  ): unknown;
  inspect_terrain(sessionDir: string, hdfName: string): unknown;
  inspect_terrain_from_surface(sessionDir: string, hdfName: string, surfaceMetaName: string): unknown;
  inspect_surface(sessionDir: string, surfaceMetaName: string): unknown;
  convert_terrain(
    sessionDir: string,
    hdfName: string,
    topElevation: number,
    sampleStep: number,
    progressCallback?: unknown,
  ): unknown;
  convert_terrain_from_surface(
    sessionDir: string,
    hdfName: string,
    topElevation: number,
    sampleStep: number,
    surfaceMetaName: string,
    terrainMaxOverride?: number | null,
    progressCallback?: unknown,
  ): unknown;
  convert_surface(
    sessionDir: string,
    terrainName: string,
    topElevation: number,
    sampleStep: number,
    surfaceMetaName: string,
    terrainMaxOverride?: number | null,
    progressCallback?: unknown,
  ): unknown;
  destroy?: () => void;
};

type PyProxyLike<T> = {
  toJs?: (options?: unknown) => T;
  destroy?: () => void;
};

type PyodideInstance = {
  FS: {
    mkdirTree(path: string): void;
    writeFile(path: string, data: Uint8Array | string): void;
    readFile(path: string, opts?: { encoding?: 'binary' | 'utf8' }): Uint8Array | string;
  };
  loadPackage(packages: string[]): Promise<void>;
  runPythonAsync(code: string): Promise<unknown>;
  pyimport(name: string): PythonModule;
};

type ActiveConversionState = {
  sessionDir: string | null;
  outputFilename: string | null;
  outputPath: string | null;
  downloadUrl: string | null;
};

let pyodidePromise: Promise<PyodideInstance> | null = null;
let sessionCounter = 0;
let activeConversion: ActiveConversionState = {
  sessionDir: null,
  outputFilename: null,
  outputPath: null,
  downloadUrl: null,
};
const PYODIDE_VERSION = '0.29.3';
const PROGRESS_RANGES: Record<ConversionProgressStep, { start: number; end: number }> = {
  'resolve-raster': { start: 0, end: 20 },
  'load-runtime': { start: 20, end: 28 },
  'load-packages': { start: 28, end: 40 },
  'load-bridge': { start: 40, end: 45 },
  'prepare-files': { start: 45, end: 50 },
  'validate-terrain': { start: 50, end: 55 },
  'write-surfaces': { start: 55, end: 84 },
  'write-stitches': { start: 84, end: 90 },
  'write-walls': { start: 90, end: 98 },
  finalize: { start: 98, end: 100 },
  complete: { start: 100, end: 100 },
};
const KNOWN_PROGRESS_STEPS = new Set<ConversionProgressStep>(Object.keys(PROGRESS_RANGES) as ConversionProgressStep[]);

function postStatus(message: string): void {
  self.postMessage({ type: 'status', payload: { message } });
}

function postProgress(payload: ConversionProgress): void {
  self.postMessage({ type: 'progress', payload });
}

function createProgressReporter() {
  let lastPercent = -1;
  let lastStep: ConversionProgressStep | null = null;

  const emit = (
    step: ConversionProgressStep,
    rawPercent: number,
    message: string,
    force = false,
  ): void => {
    const clampedPercent = Math.max(0, Math.min(100, rawPercent));
    const percent = Math.max(lastPercent, Math.round(clampedPercent));
    if (!force && percent <= lastPercent && step === lastStep) {
      return;
    }

    lastPercent = percent;
    lastStep = step;
    postProgress({ percent, step, message });
  };

  return {
    reportStage(step: ConversionProgressStep, fraction: number, message: string, force = false): void {
      const range = PROGRESS_RANGES[step];
      const normalizedFraction = Math.max(0, Math.min(1, fraction));
      const percent = range.start + ((range.end - range.start) * normalizedFraction);
      emit(step, percent, message, force);
    },
    complete(message = 'Conversion complete. Download the STL or open it in the viewer.'): void {
      emit('complete', 100, message, true);
    },
  };
}

function formatWorkerError(error: unknown): string {
  const rawMessage = error instanceof Error ? error.message : String(error);
  const lines = rawMessage
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    return 'Unknown converter error.';
  }

  const preferredPrefixes = [
    'TerrainConversionError:',
    'RasterioIOError:',
    'CPLE_AppDefinedError:',
    'Error:',
  ];
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index];
    if (preferredPrefixes.some((prefix) => line.includes(prefix))) {
      return line.replace(/^[A-Za-z_][A-Za-z0-9_.]*:\s*/, '').trim();
    }
  }

  return lines[lines.length - 1];
}

async function ensurePyodide(
  progressReporter?: ReturnType<typeof createProgressReporter>,
): Promise<PyodideInstance> {
  if (pyodidePromise) {
    progressReporter?.reportStage('load-runtime', 1, 'Pyodide runtime ready.', true);
    progressReporter?.reportStage('load-packages', 1, 'Python packages ready.', true);
    progressReporter?.reportStage('load-bridge', 1, 'Browser conversion bridge ready.', true);
    return pyodidePromise;
  }

  pyodidePromise = (async () => {
    if (progressReporter) {
      progressReporter.reportStage('load-runtime', 0, 'Loading Pyodide runtime...', true);
    } else {
      postStatus('Loading Pyodide runtime...');
    }
    const moduleUrl = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.mjs`;
    const pyodideModule = await import(/* @vite-ignore */ moduleUrl);
    const pyodide = (await pyodideModule.loadPyodide({
      indexURL: `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`,
    })) as PyodideInstance;
    progressReporter?.reportStage('load-runtime', 1, 'Pyodide runtime loaded.', true);

    if (progressReporter) {
      progressReporter.reportStage('load-packages', 0, 'Loading Python packages: numpy, h5py, rasterio...', true);
    } else {
      postStatus('Loading Python packages: numpy, h5py, rasterio...');
    }
    await pyodide.loadPackage(['numpy', 'h5py', 'rasterio']);
    progressReporter?.reportStage('load-packages', 1, 'Python packages loaded.', true);

    if (progressReporter) {
      progressReporter.reportStage('load-bridge', 0, 'Loading browser conversion bridge...', true);
    } else {
      postStatus('Loading browser conversion bridge...');
    }
    pyodide.FS.mkdirTree('/app');
    const baseUrl = new URL(import.meta.env.BASE_URL, self.location.origin);
    const [coreScript, bridgeScript] = await Promise.all([
      fetch(new URL('python/terrain_to_stl.py', baseUrl)).then((response) => response.text()),
      fetch(new URL('python/terrain_web_bridge.py', baseUrl)).then((response) => response.text()),
    ]);
    pyodide.FS.writeFile('/app/terrain_to_stl.py', coreScript);
    pyodide.FS.writeFile('/app/terrain_web_bridge.py', bridgeScript);
    await pyodide.runPythonAsync("import sys; sys.path.append('/app')");
    progressReporter?.reportStage('load-bridge', 1, 'Browser conversion bridge ready.', true);
    return pyodide;
  })();

  return pyodidePromise;
}

async function removeSessionDirectory(pyodide: PyodideInstance, sessionDir: string): Promise<void> {
  const encodedPath = JSON.stringify(sessionDir);
  await pyodide.runPythonAsync(
    `import os, shutil
path = ${encodedPath}
if os.path.isdir(path):
    shutil.rmtree(path, ignore_errors=True)
elif os.path.exists(path):
    os.remove(path)
`,
  );
}

async function safeRemoveSessionDirectory(pyodide: PyodideInstance, sessionDir: string): Promise<void> {
  try {
    await removeSessionDirectory(pyodide, sessionDir);
  } catch {
    // Best-effort cleanup should not fail the active conversion flow.
  }
}

function revokeActiveDownloadUrl(): void {
  if (activeConversion.downloadUrl) {
    URL.revokeObjectURL(activeConversion.downloadUrl);
    activeConversion.downloadUrl = null;
  }
}

async function cleanupActiveConversion(): Promise<void> {
  revokeActiveDownloadUrl();
  const sessionDir = activeConversion.sessionDir;
  activeConversion = {
    sessionDir: null,
    outputFilename: null,
    outputPath: null,
    downloadUrl: null,
  };

  if (!sessionDir || !pyodidePromise) {
    return;
  }

  const pyodide = await pyodidePromise;
  await safeRemoveSessionDirectory(pyodide, sessionDir);
}

function ensureWorkFiles(pyodide: PyodideInstance, files: UploadFilePayload[]): string {
  const sessionDir = `/work/session-${Date.now()}-${sessionCounter}`;
  sessionCounter += 1;
  pyodide.FS.mkdirTree(sessionDir);

  for (const file of files) {
    pyodide.FS.writeFile(`${sessionDir}/${file.name}`, new Uint8Array(file.bytes));
  }

  return sessionDir;
}

function selectTerrainUpload(files: UploadFilePayload[], terrainName: string): UploadFilePayload {
  const matches = files.filter((file) => file.name.toLowerCase() === terrainName.toLowerCase());
  if (matches.length === 0) {
    throw new Error(`The selected terrain file was not uploaded: ${terrainName}`);
  }
  if (matches.length > 1) {
    throw new Error(`Multiple uploaded terrain files match ${terrainName}: ${matches.map((file) => file.name).join(', ')}`);
  }
  return matches[0];
}

function writeBrowserSurfaceFiles(
  pyodide: PyodideInstance,
  sessionDir: string,
  surface: BrowserTerrainSurface,
): string {
  const metaName = 'browser_surface.meta.json';
  if (surface.kind === 'sparse') {
    const sampledRowsName = 'browser_surface.sampled_rows.i32';
    const sampledColsName = 'browser_surface.sampled_cols.i32';
    const coarseElevationsName = 'browser_surface.coarse_elevations.f32';
    const coarseValidMaskName = 'browser_surface.coarse_valid_mask.u8';
    const meta = {
      kind: 'sparse',
      resolved_raster_name: surface.resolvedRasterName,
      width: surface.width,
      height: surface.height,
      transform: surface.transform,
      max_elevation: surface.maxElevation,
      sampled_rows_file: sampledRowsName,
      sampled_cols_file: sampledColsName,
      coarse_elevations_file: coarseElevationsName,
      coarse_valid_mask_file: coarseValidMaskName,
      refinement_tiles: surface.refinementTiles.map((tile, index) => ({
        row_start: tile.rowStart,
        row_end: tile.rowEnd,
        col_start: tile.colStart,
        col_end: tile.colEnd,
        elevations_file: `browser_surface.tile_${index}.elevations.f32`,
        valid_mask_file: `browser_surface.tile_${index}.valid_mask.u8`,
      })),
    };

    pyodide.FS.writeFile(`${sessionDir}/${metaName}`, JSON.stringify(meta));
    pyodide.FS.writeFile(
      `${sessionDir}/${sampledRowsName}`,
      new Uint8Array(
        surface.sampledRows.buffer,
        surface.sampledRows.byteOffset,
        surface.sampledRows.byteLength,
      ),
    );
    pyodide.FS.writeFile(
      `${sessionDir}/${sampledColsName}`,
      new Uint8Array(
        surface.sampledCols.buffer,
        surface.sampledCols.byteOffset,
        surface.sampledCols.byteLength,
      ),
    );
    pyodide.FS.writeFile(
      `${sessionDir}/${coarseElevationsName}`,
      new Uint8Array(
        surface.coarseElevations.buffer,
        surface.coarseElevations.byteOffset,
        surface.coarseElevations.byteLength,
      ),
    );
    pyodide.FS.writeFile(
      `${sessionDir}/${coarseValidMaskName}`,
      new Uint8Array(
        surface.coarseValidMask.buffer,
        surface.coarseValidMask.byteOffset,
        surface.coarseValidMask.byteLength,
      ),
    );
    surface.refinementTiles.forEach((tile, index) => {
      pyodide.FS.writeFile(
        `${sessionDir}/browser_surface.tile_${index}.elevations.f32`,
        new Uint8Array(tile.elevations.buffer, tile.elevations.byteOffset, tile.elevations.byteLength),
      );
      pyodide.FS.writeFile(
        `${sessionDir}/browser_surface.tile_${index}.valid_mask.u8`,
        new Uint8Array(tile.validMask.buffer, tile.validMask.byteOffset, tile.validMask.byteLength),
      );
    });
    return metaName;
  }

  const elevationsName = 'browser_surface.elevations.f32';
  const validMaskName = 'browser_surface.valid_mask.u8';
  const meta = {
    kind: 'full',
    resolved_raster_name: surface.resolvedRasterName,
    width: surface.width,
    height: surface.height,
    transform: surface.transform,
    max_elevation: surface.maxElevation,
    elevations_file: elevationsName,
    valid_mask_file: validMaskName,
  };

  pyodide.FS.writeFile(`${sessionDir}/${metaName}`, JSON.stringify(meta));
  pyodide.FS.writeFile(
    `${sessionDir}/${elevationsName}`,
    new Uint8Array(
      surface.elevations.buffer,
      surface.elevations.byteOffset,
      surface.elevations.byteLength,
    ),
  );
  pyodide.FS.writeFile(
    `${sessionDir}/${validMaskName}`,
    new Uint8Array(
      surface.validMask.buffer,
      surface.validMask.byteOffset,
      surface.validMask.byteLength,
    ),
  );

  return metaName;
}

function unwrapPythonResult<T>(value: unknown): T {
  const proxy = value as PyProxyLike<T>;
  if (typeof proxy?.toJs === 'function') {
    const result = proxy.toJs({
      dict_converter: Object.fromEntries as <TValue>(
        entries: Iterable<readonly [PropertyKey, TValue]>,
      ) => Record<PropertyKey, TValue>,
    });
    proxy.destroy?.();
    return result;
  }
  return value as T;
}

type TerrainPreflight = BrowserRasterProbe & {
  estimatedPeakWorkingSetBytes: number;
};

function formatBinaryBytes(byteCount: number): string {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let value = byteCount;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 2)} ${units[unitIndex]}`;
}

function buildTerrainPreflight(probe: BrowserRasterProbe): TerrainPreflight {
  return {
    ...probe,
    estimatedPeakWorkingSetBytes: estimatePeakWorkingSetBytes(
      probe.totalInputBytes,
      probe.largestSourceCellCount,
      probe.targetCellCount,
    ),
  };
}

function toInspectionResult(
  result: PythonInspection,
  terrainMaxElevation: number,
  sampleStepOptions: SampleStepOption[],
  browserLimits: BrowserLimits,
): TerrainInspection {
  return {
    terrainMaxElevation,
    resolvedRasterName: result.resolved_raster_name,
    stitchPointCount: result.stitch_point_count,
    stitchTriangleCount: result.stitch_triangle_count,
    hasPopulatedStitchTin: result.has_populated_stitch_tin,
    browserLimits,
    sampleStepOptions,
  };
}

function toConversionResult(result: PythonConversion): ConversionResult {
  return {
    outputFilename: result.output_filename,
    terrainMaxElevation: result.terrain_max_elevation,
    resolvedRasterName: result.resolved_raster_name,
    triangleCount: result.triangle_count,
    wallTriangleCount: result.wall_triangle_count,
    stitchPointCount: result.stitch_point_count,
    stitchTriangleCount: result.stitch_triangle_count,
    stitchBridgeTriangleCount: result.stitch_bridge_triangle_count,
    stlSizeBytes: result.stl_size_bytes,
  };
}

function toAdaptiveStitchMetrics(
  raw: PythonAdaptiveStitchMetrics | undefined,
): AdaptiveStitchMetrics | null {
  if (!raw) {
    return null;
  }

  return {
    coarseCellCount: raw.coarse_cell_count,
    refinedCellCount: raw.refined_cell_count,
    transitionTriangleCount: raw.transition_triangle_count,
    refinementPerimeterCellCount: raw.refinement_perimeter_cell_count,
    refinedVertexCount: raw.refined_vertex_count,
    largestRefinementVertexCount: raw.largest_refinement_vertex_count,
    regionCount: raw.region_count,
  };
}

function estimateSparseWorkingSetForStep(
  preflight: TerrainPreflight,
  sampleStep: number,
  adaptiveMetrics: AdaptiveStitchMetrics | null = null,
): number {
  if (sampleStep <= 1) {
    return preflight.estimatedPeakWorkingSetBytes;
  }

  const sampledRowCount = buildSampleIndices(preflight.height, sampleStep).length;
  const sampledColumnCount = buildSampleIndices(preflight.width, sampleStep).length;
  const coarseVertexCount = sampledRowCount * sampledColumnCount;
  const refinedVertexCount = adaptiveMetrics?.refinedVertexCount ?? 0;
  const largestDecodeWindowCellCount = Math.max(
    preflight.largestSourceWidth,
    adaptiveMetrics?.largestRefinementVertexCount ?? 0,
  );
  return estimateSparsePeakWorkingSetBytes(
    preflight.totalInputBytes,
    largestDecodeWindowCellCount,
    coarseVertexCount + refinedVertexCount,
    sampledRowCount,
    sampledColumnCount,
  );
}

function toBrowserRefinementRegions(plan: PythonSparsePlan): BrowserRefinementRegion[] {
  return plan.refinement_regions.map((region) => ({
    rowStart: region.row_start,
    rowEnd: region.row_end,
    colStart: region.col_start,
    colEnd: region.col_end,
  }));
}

function buildUpperBoundSampleStepOptions(
  preflight: TerrainPreflight,
  inspection: PythonInspection,
): SampleStepOption[] {
  return buildSampleStepOptions(
    'upper-bound',
    (sampleStep) => {
      const adaptiveMetrics = toAdaptiveStitchMetrics(inspection.adaptive_stitch_metrics[String(sampleStep)]);
      if (inspection.has_populated_stitch_tin && adaptiveMetrics) {
        return estimateAdaptiveStlUpperBoundBytes(
          preflight.width,
          preflight.height,
          sampleStep,
          inspection.stitch_triangle_count,
          adaptiveMetrics,
        );
      }

      return estimateStlUpperBoundBytes(
        preflight.width,
        preflight.height,
        sampleStep,
        inspection.stitch_triangle_count,
      );
    },
    (sampleStep) => estimateSparseWorkingSetForStep(
      preflight,
      sampleStep,
      inspection.has_populated_stitch_tin
        ? toAdaptiveStitchMetrics(inspection.adaptive_stitch_metrics[String(sampleStep)])
        : null,
    ),
  );
}

function buildExactSampleStepOptions(
  preflight: TerrainPreflight,
  surface: BrowserRasterSurface,
): SampleStepOption[] {
  return buildSampleStepOptions(
    'exact',
    (sampleStep) => estimateSurfaceStlBytes(surface, sampleStep),
    (sampleStep) => estimateSparseWorkingSetForStep(preflight, sampleStep),
  );
}

function formatDemInspectLimitError(preflight: TerrainPreflight): string {
  return [
    'Browser inspection was blocked before raster decode.',
    `Estimated peak working set is ${formatBinaryBytes(preflight.estimatedPeakWorkingSetBytes)} (limit ${formatBinaryBytes(BROWSER_PEAK_WORKING_SET_LIMIT_BYTES)}).`,
    'Large files near these limits may still stall or crash the browser.',
    'Use a smaller raster, crop the terrain, or use the desktop Python workflow for large local runs.',
  ].join(' ');
}

function formatMissingHdfMaxError(preflight: TerrainPreflight): string {
  return [
    'The HDF does not expose a terrain max elevation in its metadata.',
    `Recovering it would require reading the full raster, and the estimated peak working set is ${formatBinaryBytes(preflight.estimatedPeakWorkingSetBytes)} (limit ${formatBinaryBytes(BROWSER_PEAK_WORKING_SET_LIMIT_BYTES)}).`,
    'Use the desktop Python workflow for this terrain or reduce the raster extent first.',
  ].join(' ');
}

function formatBrowserLimitError(
  sampleStep: number,
  estimatedSizeBytes: number,
  estimatedWorkingSetBytes: number,
  browserLimits: BrowserLimits,
): string {
  return [
    'Browser conversion was blocked before raster decode.',
    `Estimated peak working set for sample step ${sampleStep} is ${formatBinaryBytes(estimatedWorkingSetBytes)} (limit ${formatBinaryBytes(browserLimits.peakWorkingSetLimitBytes)}).`,
    `Estimated STL upper bound for sample step ${sampleStep} is ${formatBinaryBytes(estimatedSizeBytes)} (limit ${formatBinaryBytes(browserLimits.stlSizeLimitBytes)}).`,
    'Files near these limits may still stall or crash the browser.',
    'Use a smaller cropped terrain, a larger sample step, or the desktop Python workflow for large local runs.',
  ].join(' ');
}

async function readHdfMetadata(
  pyodide: PyodideInstance,
  sessionDir: string,
  hdfName: string,
  resolvedRasterName: string,
  rasterWidth: number,
  rasterHeight: number,
  rasterTransform: number[],
  rasterMaxElevation: number | null = null,
): Promise<PythonInspection> {
  const bridge = pyodide.pyimport('terrain_web_bridge');
  try {
    return unwrapPythonResult<PythonInspection>(
      bridge.inspect_hdf_metadata(
        sessionDir,
        hdfName,
        resolvedRasterName,
        rasterWidth,
        rasterHeight,
        rasterTransform,
        rasterMaxElevation,
      ),
    );
  } finally {
    bridge.destroy?.();
  }
}

async function readHdfSparsePlan(
  pyodide: PyodideInstance,
  sessionDir: string,
  hdfName: string,
  rasterWidth: number,
  rasterHeight: number,
  rasterTransform: number[],
  sampleStep: number,
): Promise<PythonSparsePlan> {
  const bridge = pyodide.pyimport('terrain_web_bridge');
  try {
    return unwrapPythonResult<PythonSparsePlan>(
      bridge.build_hdf_sparse_plan(
        sessionDir,
        hdfName,
        rasterWidth,
        rasterHeight,
        rasterTransform,
        sampleStep,
      ),
    );
  } finally {
    bridge.destroy?.();
  }
}

async function handleInspect(message: InspectMessage): Promise<void> {
  await cleanupActiveConversion();
  postStatus('Inspecting terrain raster headers...');
  const probe = await probeTerrainRaster(
    message.files,
    message.terrainName,
    message.terrainKind,
    postStatus,
  );
  const preflight = buildTerrainPreflight(probe);

  if (message.terrainKind === 'dem') {
    if (preflight.estimatedPeakWorkingSetBytes > BROWSER_PEAK_WORKING_SET_LIMIT_BYTES) {
      throw new Error(formatDemInspectLimitError(preflight));
    }

    postStatus('Reading terrain raster metadata...');
    const surface = await decodeTerrainRasterSurface(
      message.files,
      message.terrainName,
      message.terrainKind,
      postStatus,
    );
    const raw: PythonInspection = {
      terrain_max_elevation: surface.maxElevation,
      resolved_raster_name: surface.resolvedRasterName,
      stitch_point_count: 0,
      stitch_triangle_count: 0,
      has_populated_stitch_tin: false,
      stitch_component_count: 0,
      adaptive_stitch_metrics: {},
    };
    const sampleStepOptions = buildExactSampleStepOptions(preflight, surface);
    const browserLimits = buildBrowserLimits(
      preflight.width,
      preflight.height,
      preflight.totalInputBytes,
      preflight.estimatedPeakWorkingSetBytes,
      sampleStepOptions,
    );
    self.postMessage({
      type: 'inspected',
      payload: toInspectionResult(raw, surface.maxElevation, sampleStepOptions, browserLimits),
    });
    return;
  }

  const pyodide = await ensurePyodide();
  postStatus('Reading terrain HDF metadata...');
  const sessionDir = ensureWorkFiles(pyodide, [selectTerrainUpload(message.files, message.terrainName)]);
  try {
    let raw = await readHdfMetadata(
      pyodide,
      sessionDir,
      message.terrainName,
      preflight.resolvedRasterName,
      preflight.width,
      preflight.height,
      Array.from(preflight.transform),
    );
    let terrainMaxElevation = raw.terrain_max_elevation;

    if (terrainMaxElevation === null) {
      if (preflight.estimatedPeakWorkingSetBytes > BROWSER_PEAK_WORKING_SET_LIMIT_BYTES) {
        throw new Error(formatMissingHdfMaxError(preflight));
      }

      postStatus('Reading terrain raster metadata to recover terrain max elevation...');
      const surface = await decodeTerrainRasterSurface(
        message.files,
        message.terrainName,
        message.terrainKind,
        postStatus,
      );
      raw = await readHdfMetadata(
        pyodide,
        sessionDir,
        message.terrainName,
        preflight.resolvedRasterName,
        preflight.width,
        preflight.height,
        Array.from(preflight.transform),
        surface.maxElevation,
      );
      terrainMaxElevation = raw.terrain_max_elevation;
    }

    if (terrainMaxElevation === null) {
      throw new Error('The browser inspector could not determine the terrain max elevation.');
    }

    const sampleStepOptions = buildUpperBoundSampleStepOptions(preflight, raw);
    const browserLimits = buildBrowserLimits(
      preflight.width,
      preflight.height,
      preflight.totalInputBytes,
      preflight.estimatedPeakWorkingSetBytes,
      sampleStepOptions,
    );
    self.postMessage({
      type: 'inspected',
      payload: toInspectionResult(raw, terrainMaxElevation, sampleStepOptions, browserLimits),
    });
  } finally {
    await safeRemoveSessionDirectory(pyodide, sessionDir);
  }
}

async function handleConvert(message: ConvertMessage): Promise<void> {
  await cleanupActiveConversion();
  const progressReporter = createProgressReporter();
  progressReporter.reportStage('resolve-raster', 0, 'Inspecting terrain raster headers...', true);
  const probe = await probeTerrainRaster(
    message.files,
    message.terrainName,
    message.terrainKind,
    undefined,
    ({ fraction, message: progressMessage }) => {
      progressReporter.reportStage(
        'resolve-raster',
        fraction * 0.15,
        progressMessage,
        fraction === 0 || fraction === 1,
      );
    },
  );
  const preflight = buildTerrainPreflight(probe);

  let pyodide: PyodideInstance | null = null;
  let sessionDir: string | null = null;
  let bridge: PythonModule | null = null;
  let keepSession = false;
  let hdfInspection: PythonInspection | null = null;

  try {
    if (message.terrainKind === 'hdf') {
      pyodide = await ensurePyodide(progressReporter);
      progressReporter.reportStage('prepare-files', 0, 'Reading terrain HDF metadata...', true);
      sessionDir = ensureWorkFiles(pyodide, [selectTerrainUpload(message.files, message.terrainName)]);
      hdfInspection = await readHdfMetadata(
        pyodide,
        sessionDir,
        message.terrainName,
        preflight.resolvedRasterName,
        preflight.width,
        preflight.height,
        Array.from(preflight.transform),
      );
    }

    const sampleStepOptions =
      message.terrainKind === 'hdf' && hdfInspection
        ? buildUpperBoundSampleStepOptions(preflight, hdfInspection)
        : buildSampleStepOptions(
            'upper-bound',
            (sampleStep) => estimateStlUpperBoundBytes(preflight.width, preflight.height, sampleStep, 0),
            (sampleStep) => estimateSparseWorkingSetForStep(preflight, sampleStep),
          );
    const selectedSampleStep =
      sampleStepOptions.find((option) => option.value === message.sampleStep)
      ?? evaluateSampleStep(
        message.sampleStep,
        'upper-bound',
        estimateStlUpperBoundBytes(
          preflight.width,
          preflight.height,
          message.sampleStep,
          hdfInspection?.stitch_triangle_count ?? 0,
        ),
        estimateSparseWorkingSetForStep(
          preflight,
          message.sampleStep,
          hdfInspection?.has_populated_stitch_tin
            ? toAdaptiveStitchMetrics(hdfInspection.adaptive_stitch_metrics[String(message.sampleStep)])
            : null,
        ),
      );
    const browserLimits = buildBrowserLimits(
      preflight.width,
      preflight.height,
      preflight.totalInputBytes,
      preflight.estimatedPeakWorkingSetBytes,
      sampleStepOptions,
    );

    if (selectedSampleStep.disabled) {
      throw new Error(
        formatBrowserLimitError(
          message.sampleStep,
          selectedSampleStep.estimatedSizeBytes ?? 0,
          selectedSampleStep.estimatedWorkingSetBytes ?? preflight.estimatedPeakWorkingSetBytes,
          browserLimits,
        ),
      );
    }

    progressReporter.reportStage('resolve-raster', 0.15, 'Browser limits checked.', true);
    let refinementRegions: BrowserRefinementRegion[] = [];
    if (
      message.sampleStep > 1 &&
      message.terrainKind === 'hdf' &&
      pyodide &&
      sessionDir &&
      hdfInspection?.has_populated_stitch_tin
    ) {
      progressReporter.reportStage('resolve-raster', 0.15, 'Planning sparse stitch-aware refinement regions...', true);
      const sparsePlan = await readHdfSparsePlan(
        pyodide,
        sessionDir,
        message.terrainName,
        preflight.width,
        preflight.height,
        Array.from(preflight.transform),
        message.sampleStep,
      );
      refinementRegions = toBrowserRefinementRegions(sparsePlan);
    }

    const decodeMessage =
      message.sampleStep > 1
        ? refinementRegions.length > 0
          ? 'Reading sparse terrain samples and stitch refinement windows into browser memory...'
          : 'Reading sparse terrain samples into browser memory...'
        : 'Reading terrain raster into browser memory...';
    progressReporter.reportStage('resolve-raster', 0.15, decodeMessage, true);
    const surface = await decodeTerrainRasterSurfaceForStep(
      message.files,
      message.terrainName,
      message.terrainKind,
      message.sampleStep,
      refinementRegions,
      undefined,
      ({ fraction, message: progressMessage }) => {
        progressReporter.reportStage(
          'resolve-raster',
          0.15 + (fraction * 0.85),
          progressMessage,
          fraction === 0 || fraction === 1,
        );
      },
    );
    progressReporter.reportStage('resolve-raster', 1, 'Resolved terrain raster in the browser.', true);

    if (!pyodide) {
      pyodide = await ensurePyodide(progressReporter);
    }
    if (!sessionDir) {
      sessionDir = ensureWorkFiles(pyodide, []);
    }

    progressReporter.reportStage('prepare-files', 0, 'Preparing terrain files...', true);
    const surfaceMetaName = writeBrowserSurfaceFiles(pyodide, sessionDir, surface);
    progressReporter.reportStage('prepare-files', 1, 'Prepared terrain files for conversion.', true);
    bridge = pyodide.pyimport('terrain_web_bridge');

    const pythonProgressCallback = (
      step: string,
      completed: number,
      total: number,
      progressMessage: string,
    ): void => {
      if (!KNOWN_PROGRESS_STEPS.has(step as ConversionProgressStep) || step === 'complete') {
        return;
      }
      const normalizedTotal = total <= 0 ? 1 : total;
      const fraction = Math.min(1, Math.max(0, completed / normalizedTotal));
      progressReporter.reportStage(step as ConversionProgressStep, fraction, progressMessage);
    };

    const raw =
      message.terrainKind === 'hdf'
        ? unwrapPythonResult<PythonConversion>(
            bridge.convert_terrain_from_surface(
              sessionDir,
              message.terrainName,
              message.topElevation,
              message.sampleStep,
              surfaceMetaName,
              message.terrainMaxElevation,
              pythonProgressCallback,
            ),
          )
        : unwrapPythonResult<PythonConversion>(
            bridge.convert_surface(
              sessionDir,
              message.terrainName,
              message.topElevation,
              message.sampleStep,
              surfaceMetaName,
              message.terrainMaxElevation,
              pythonProgressCallback,
            ),
          );
    const result = toConversionResult(raw);
    activeConversion = {
      sessionDir,
      outputFilename: raw.output_filename,
      outputPath: `${sessionDir}/${raw.output_filename}`,
      downloadUrl: null,
    };
    keepSession = true;
    progressReporter.complete();
    self.postMessage({
      type: 'converted',
      payload: result,
    });
  } finally {
    bridge?.destroy?.();
    if (!keepSession && pyodide && sessionDir) {
      await safeRemoveSessionDirectory(pyodide, sessionDir);
    }
  }
}

async function handlePrepareDownload(): Promise<void> {
  if (!activeConversion.outputPath || !activeConversion.outputFilename) {
    throw new Error('No completed STL is available to download.');
  }

  if (activeConversion.downloadUrl) {
    self.postMessage({
      type: 'downloadReady',
      downloadUrl: activeConversion.downloadUrl,
      filename: activeConversion.outputFilename,
    });
    return;
  }

  postStatus('Preparing STL download...');
  const pyodide = await ensurePyodide();
  const stlBytes = pyodide.FS.readFile(activeConversion.outputPath, { encoding: 'binary' });
  if (!(stlBytes instanceof Uint8Array)) {
    throw new Error('The STL file could not be prepared for download.');
  }

  const downloadBytes = new Uint8Array(stlBytes);
  const downloadUrl = URL.createObjectURL(new Blob([downloadBytes], { type: 'model/stl' }));
  activeConversion.downloadUrl = downloadUrl;

  postStatus('STL download ready.');
  self.postMessage({
    type: 'downloadReady',
    downloadUrl,
    filename: activeConversion.outputFilename,
  });
}

async function handlePrepareViewerFile(): Promise<void> {
  if (!activeConversion.outputPath || !activeConversion.outputFilename) {
    throw new Error('No completed STL is available to open in the viewer.');
  }

  postStatus('Preparing STL for viewer...');
  const pyodide = await ensurePyodide();
  const stlBytes = pyodide.FS.readFile(activeConversion.outputPath, { encoding: 'binary' });
  if (!(stlBytes instanceof Uint8Array)) {
    throw new Error('The STL file could not be prepared for the viewer.');
  }

  const preparedBytes = new Uint8Array(stlBytes);
  const payload: PreparedStlPayload = {
    filename: activeConversion.outputFilename,
    bytes: preparedBytes.buffer,
  };

  postStatus('STL ready for viewer.');
  self.postMessage(
    {
      type: 'viewerFileReady',
      payload,
    },
    [payload.bytes],
  );
}

async function handleReleaseDownload(): Promise<void> {
  await cleanupActiveConversion();
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  try {
    if (event.data.type === 'inspect') {
      await handleInspect(event.data);
      return;
    }

    if (event.data.type === 'prepareDownload') {
      await handlePrepareDownload();
      return;
    }

    if (event.data.type === 'prepareViewerFile') {
      await handlePrepareViewerFile();
      return;
    }

    if (event.data.type === 'releaseDownload') {
      await handleReleaseDownload();
      return;
    }

    await handleConvert(event.data);
  } catch (error) {
    const message = formatWorkerError(error);
    self.postMessage({ type: 'error', error: message });
  }
};

export {};
