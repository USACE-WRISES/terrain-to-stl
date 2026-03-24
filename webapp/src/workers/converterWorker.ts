/// <reference lib="webworker" />

import {
  decodeTerrainRasterSurface,
  type BrowserRasterSurface,
  type BrowserTerrainInputKind,
} from '../lib/browserRasterSurface';
import type {
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
  terrain_max_elevation: number;
  resolved_raster_name: string;
  stitch_point_count: number;
  stitch_triangle_count: number;
  has_populated_stitch_tin: boolean;
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
    progressCallback?: unknown,
  ): unknown;
  convert_surface(
    sessionDir: string,
    terrainName: string,
    topElevation: number,
    sampleStep: number,
    surfaceMetaName: string,
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
const SAMPLE_STEP_PRESETS = [1, 2, 4, 8, 16] as const;
const SAMPLE_STEP_DISABLED_REASON = 'requires step 1 for stitch-aware terrain';
const STL_HEADER_BYTES = 84;
const STL_TRIANGLE_BYTES = 50;
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

function toInspectionResult(
  result: PythonInspection,
  sampleStepOptions: SampleStepOption[],
): TerrainInspection {
  return {
    terrainMaxElevation: result.terrain_max_elevation,
    resolvedRasterName: result.resolved_raster_name,
    stitchPointCount: result.stitch_point_count,
    stitchTriangleCount: result.stitch_triangle_count,
    hasPopulatedStitchTin: result.has_populated_stitch_tin,
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
  surface: BrowserRasterSurface,
): string {
  const metaName = 'browser_surface.meta.json';
  const elevationsName = 'browser_surface.elevations.f32';
  const validMaskName = 'browser_surface.valid_mask.u8';
  const meta = {
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

function buildSampleIndices(size: number, step: number): number[] {
  const indices: number[] = [];
  for (let index = 0; index < size; index += step) {
    indices.push(index);
  }
  if (indices.length === 0 || indices[indices.length - 1] !== size - 1) {
    indices.push(size - 1);
  }
  return indices;
}

function encodeVertex(width: number, row: number, col: number): number {
  return (row * width) + col;
}

function updateBoundaryEdges(boundaryEdges: Set<string>, a: number, b: number): void {
  const key = a < b ? `${a}:${b}` : `${b}:${a}`;
  if (boundaryEdges.has(key)) {
    boundaryEdges.delete(key);
    return;
  }
  boundaryEdges.add(key);
}

function isValidSurfaceCell(surface: BrowserRasterSurface, row: number, col: number): boolean {
  return surface.validMask[(row * surface.width) + col] !== 0;
}

function estimateSampleStepSizeMb(surface: BrowserRasterSurface, sampleStep: number): number {
  const sampledRows = buildSampleIndices(surface.height, sampleStep);
  const sampledCols = buildSampleIndices(surface.width, sampleStep);
  const boundaryEdges = new Set<string>();
  let rasterTriangleCount = 0;

  for (let rowPairIndex = 0; rowPairIndex < sampledRows.length - 1; rowPairIndex += 1) {
    const row0 = sampledRows[rowPairIndex];
    const row1 = sampledRows[rowPairIndex + 1];

    for (let colIndex = 0; colIndex < sampledCols.length - 1; colIndex += 1) {
      const col0 = sampledCols[colIndex];
      const col1 = sampledCols[colIndex + 1];
      const a = encodeVertex(surface.width, row0, col0);
      const b = encodeVertex(surface.width, row1, col0);
      const c = encodeVertex(surface.width, row1, col1);
      const d = encodeVertex(surface.width, row0, col1);

      if (
        isValidSurfaceCell(surface, row0, col0) &&
        isValidSurfaceCell(surface, row1, col0) &&
        isValidSurfaceCell(surface, row1, col1)
      ) {
        updateBoundaryEdges(boundaryEdges, a, b);
        updateBoundaryEdges(boundaryEdges, b, c);
        updateBoundaryEdges(boundaryEdges, c, a);
        rasterTriangleCount += 1;
      }

      if (
        isValidSurfaceCell(surface, row0, col0) &&
        isValidSurfaceCell(surface, row1, col1) &&
        isValidSurfaceCell(surface, row0, col1)
      ) {
        updateBoundaryEdges(boundaryEdges, a, c);
        updateBoundaryEdges(boundaryEdges, c, d);
        updateBoundaryEdges(boundaryEdges, d, a);
        rasterTriangleCount += 1;
      }
    }
  }

  const totalTriangles = (rasterTriangleCount * 2) + (boundaryEdges.size * 2);
  return (STL_HEADER_BYTES + (totalTriangles * STL_TRIANGLE_BYTES)) / (1024 * 1024);
}

function buildSampleStepOptions(
  surface: BrowserRasterSurface,
  hasPopulatedStitchTin: boolean,
): SampleStepOption[] {
  return SAMPLE_STEP_PRESETS.map((value) => {
    if (hasPopulatedStitchTin && value !== 1) {
      return {
        value,
        estimatedSizeMb: null,
        disabled: true,
        reason: SAMPLE_STEP_DISABLED_REASON,
      };
    }

    return {
      value,
      estimatedSizeMb: estimateSampleStepSizeMb(surface, value),
      disabled: false,
      reason: null,
    };
  });
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

async function handleInspect(message: InspectMessage): Promise<void> {
  await cleanupActiveConversion();
  postStatus('Resolving terrain raster in the browser...');
  const surface = await decodeTerrainRasterSurface(
    message.files,
    message.terrainName,
    message.terrainKind,
    postStatus,
  );
  const pyodide = await ensurePyodide();
  postStatus(
    message.terrainKind === 'hdf'
      ? 'Reading terrain HDF metadata and stitch arrays...'
      : 'Reading terrain raster metadata...',
  );
  const sessionFiles =
    message.terrainKind === 'hdf' ? [selectTerrainUpload(message.files, message.terrainName)] : [];
  const sessionDir = ensureWorkFiles(pyodide, sessionFiles);
  const surfaceMetaName = writeBrowserSurfaceFiles(pyodide, sessionDir, surface);
  const bridge = pyodide.pyimport('terrain_web_bridge');
  try {
    const raw =
      message.terrainKind === 'hdf'
        ? unwrapPythonResult<PythonInspection>(
            bridge.inspect_terrain_from_surface(sessionDir, message.terrainName, surfaceMetaName),
          )
        : unwrapPythonResult<PythonInspection>(bridge.inspect_surface(sessionDir, surfaceMetaName));
    self.postMessage({
      type: 'inspected',
      payload: toInspectionResult(
        raw,
        buildSampleStepOptions(surface, raw.has_populated_stitch_tin),
      ),
    });
  } finally {
    bridge.destroy?.();
    await safeRemoveSessionDirectory(pyodide, sessionDir);
  }
}

async function handleConvert(message: ConvertMessage): Promise<void> {
  await cleanupActiveConversion();
  const progressReporter = createProgressReporter();
  progressReporter.reportStage('resolve-raster', 0, 'Resolving terrain raster in the browser...', true);
  const surface = await decodeTerrainRasterSurface(
    message.files,
    message.terrainName,
    message.terrainKind,
    undefined,
    ({ fraction, message: progressMessage }) => {
      progressReporter.reportStage('resolve-raster', fraction, progressMessage, fraction === 0 || fraction === 1);
    },
  );
  const pyodide = await ensurePyodide(progressReporter);
  progressReporter.reportStage('prepare-files', 0, 'Preparing terrain files...', true);
  const sessionFiles =
    message.terrainKind === 'hdf' ? [selectTerrainUpload(message.files, message.terrainName)] : [];
  const sessionDir = ensureWorkFiles(pyodide, sessionFiles);
  const surfaceMetaName = writeBrowserSurfaceFiles(pyodide, sessionDir, surface);
  progressReporter.reportStage('prepare-files', 1, 'Prepared terrain files for conversion.', true);
  const bridge = pyodide.pyimport('terrain_web_bridge');
  let keepSession = false;
  try {
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
    bridge.destroy?.();
    if (!keepSession) {
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
