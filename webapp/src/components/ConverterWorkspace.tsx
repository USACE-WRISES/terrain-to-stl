import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { SAMPLE_STEP_PRESETS } from '../lib/browserTerrainLimits';
import {
  DESKTOP_WORKFLOW_STEPS,
  DESKTOP_WORKFLOW_WARNING,
  getDesktopWorkflowRecommendation,
  resolveDesktopWorkflowUrl,
} from '../lib/desktopWorkflow';
import type {
  ConversionProgress,
  ConversionProgressStep,
  ConversionResult,
  PreparedStlPayload,
  SampleStepOption,
  TerrainInspection,
  UploadFilePayload,
} from '../lib/types';
import { fetchExampleTerrainFiles } from '../lib/exampleAssets';

const PROGRESS_STEP_ORDER: Array<Exclude<ConversionProgressStep, 'complete'>> = [
  'resolve-raster',
  'load-runtime',
  'load-packages',
  'load-bridge',
  'prepare-files',
  'validate-terrain',
  'write-surfaces',
  'write-stitches',
  'write-walls',
  'finalize',
];

const PROGRESS_STEP_LABELS: Record<Exclude<ConversionProgressStep, 'complete'>, string> = {
  'resolve-raster': 'Resolve terrain raster',
  'load-runtime': 'Load Pyodide runtime',
  'load-packages': 'Load Python packages',
  'load-bridge': 'Load browser bridge',
  'prepare-files': 'Prepare conversion files',
  'validate-terrain': 'Validate terrain',
  'write-surfaces': 'Write top and bottom surfaces',
  'write-stitches': 'Write stitch bridge triangles',
  'write-walls': 'Write boundary walls',
  finalize: 'Finalize STL file',
};
const INITIAL_CONVERTER_STATUS = 'Select a terrain HDF with its raster files, or a standalone DEM GeoTIFF.';
const SAMPLE_STEP_TOOLTIP =
  'Raster sample step controls how densely the raster is sampled for the STL. 1 uses every raster cell for the most detail. Higher values skip cells to reduce detail and usually shrink the STL. The converter still includes the last raster row and column. Stitch-aware terrains use local refinement around stitch components and support preset steps 1, 2, 4, 8, 16, and 32.';
const DEFAULT_SAMPLE_STEP_OPTIONS: SampleStepOption[] = SAMPLE_STEP_PRESETS.map((value) => ({
  value,
  estimatedSizeBytes: null,
  estimatedWorkingSetBytes: null,
  estimateKind: 'upper-bound',
  disabled: false,
  reason: null,
}));

type ConverterWorkspaceProps = {
  variant?: 'page' | 'modal';
  onOpenInViewer?: (file: File) => void | Promise<void>;
};

type TerrainInputSelection =
  | {
      kind: 'hdf' | 'dem';
      file: File;
    }
  | {
      kind: null;
      file: null;
      error: string | null;
    };

function formatBytes(byteCount: number): string {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = byteCount;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 2)} ${units[unitIndex]}`;
}

function isGeoTiffFile(file: File): boolean {
  const lowerName = file.name.toLowerCase();
  return lowerName.endsWith('.tif') || lowerName.endsWith('.tiff');
}

function formatSampleStepOptionLabel(option: SampleStepOption): string {
  if (option.estimatedSizeBytes !== null) {
    const estimateLabel =
      option.estimateKind === 'upper-bound'
        ? `up to ${formatBytes(option.estimatedSizeBytes)}`
        : `~${formatBytes(option.estimatedSizeBytes)}`;
    const workingSetLabel =
      option.estimatedWorkingSetBytes === null
        ? null
        : `working set ${formatBytes(option.estimatedWorkingSetBytes)}`;
    const combinedLabel = workingSetLabel === null ? estimateLabel : `${estimateLabel}; ${workingSetLabel}`;
    if (option.disabled && option.reason) {
      return `${option.value} (${combinedLabel}; ${option.reason})`;
    }
    return `${option.value} (${combinedLabel})`;
  }
  if (option.disabled && option.reason) {
    return `${option.value} (${option.reason})`;
  }
  return String(option.value);
}

export function ConverterWorkspace({
  variant = 'page',
  onOpenInViewer,
}: ConverterWorkspaceProps) {
  const workerRef = useRef<Worker | null>(null);
  const downloadUrlRef = useRef<string | null>(null);
  const sampleStepTooltipId = useId();
  const sampleStepSelectId = useId();

  const [status, setStatus] = useState(INITIAL_CONVERTER_STATUS);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [inspection, setInspection] = useState<TerrainInspection | null>(null);
  const [conversion, setConversion] = useState<ConversionResult | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [downloadBusy, setDownloadBusy] = useState(false);
  const [exampleLoadBusy, setExampleLoadBusy] = useState(false);
  const [openInViewerBusy, setOpenInViewerBusy] = useState(false);
  const [progress, setProgress] = useState<ConversionProgress | null>(null);
  const [topElevation, setTopElevation] = useState('');
  const [sampleStep, setSampleStep] = useState('1');
  const [working, setWorking] = useState(false);

  const terrainInput = useMemo<TerrainInputSelection>(() => {
    const hdfFiles = uploadedFiles.filter((file) => file.name.toLowerCase().endsWith('.hdf'));
    if (hdfFiles.length > 1) {
      return {
        kind: null,
        file: null,
        error: `Upload only one terrain HDF at a time. Found: ${hdfFiles.map((file) => file.name).join(', ')}`,
      };
    }
    if (hdfFiles.length === 1) {
      return { kind: 'hdf', file: hdfFiles[0] };
    }

    const demFiles = uploadedFiles.filter(isGeoTiffFile);
    if (demFiles.length > 1) {
      return {
        kind: null,
        file: null,
        error: 'When no HDF is uploaded, choose exactly one DEM GeoTIFF (.tif or .tiff).',
      };
    }
    if (demFiles.length === 1) {
      return { kind: 'dem', file: demFiles[0] };
    }

    return { kind: null, file: null, error: null };
  },
    [uploadedFiles],
  );
  const sampleStepOptions = inspection?.sampleStepOptions ?? DEFAULT_SAMPLE_STEP_OPTIONS;
  const selectedSampleStepOption = sampleStepOptions.find((option) => String(option.value) === sampleStep) ?? null;
  const terrainInputError = terrainInput.kind === null ? terrainInput.error : null;
  const outputActionBusy = downloadBusy || openInViewerBusy;
  const controlsBusy = working || outputActionBusy || exampleLoadBusy;
  const canConvert = Boolean(
    terrainInput.file &&
    inspection &&
    !controlsBusy &&
    selectedSampleStepOption &&
    !selectedSampleStepOption.disabled,
  );

  useEffect(() => {
    const selectedOption = sampleStepOptions.find((option) => String(option.value) === sampleStep);
    if (selectedOption && !selectedOption.disabled) {
      return;
    }

    const nextOption = sampleStepOptions.find((option) => !option.disabled) ?? sampleStepOptions[0];
    if (nextOption && String(nextOption.value) !== sampleStep) {
      setSampleStep(String(nextOption.value));
    }
  }, [sampleStep, sampleStepOptions]);

  function triggerDownload(url: string, filename: string): void {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.rel = 'noopener';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  }

  function clearDownloadUrl(notifyWorker = true): void {
    if (notifyWorker && workerRef.current) {
      workerRef.current.postMessage({ type: 'releaseDownload' });
    }
    if (downloadUrlRef.current) {
      URL.revokeObjectURL(downloadUrlRef.current);
      downloadUrlRef.current = null;
    }
    setDownloadUrl(null);
    setDownloadBusy(false);
  }

  function clearConverterOutputState(notifyWorker = true): void {
    clearDownloadUrl(notifyWorker);
    setConversion(null);
    setProgress(null);
    setError(null);
    setStatus(INITIAL_CONVERTER_STATUS);
    setWorking(false);
    setOpenInViewerBusy(false);
  }

  async function openPreparedStlInViewer(payload: PreparedStlPayload): Promise<void> {
    if (!onOpenInViewer) {
      setOpenInViewerBusy(false);
      return;
    }

    try {
      setError(null);
      setStatus('Opening STL in viewer...');
      const file = new File([payload.bytes], payload.filename, { type: 'model/stl' });
      await onOpenInViewer(file);
    } catch (viewerError) {
      setError(viewerError instanceof Error ? viewerError.message : String(viewerError));
      setStatus('The STL could not be opened in the viewer.');
    } finally {
      setOpenInViewerBusy(false);
    }
  }

  useEffect(() => {
    const worker = new Worker(new URL('../workers/converterWorker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent) => {
      const { type, payload, error: workerError, downloadUrl: workerDownloadUrl, filename } = event.data as {
        type: string;
        payload?:
          | TerrainInspection
          | ConversionResult
          | ConversionProgress
          | PreparedStlPayload
          | { message: string };
        error?: string;
        downloadUrl?: string;
        filename?: string;
      };

      if (type === 'status') {
        setStatus((payload as { message: string }).message);
        return;
      }
      if (type === 'progress') {
        const nextProgress = payload as ConversionProgress;
        setProgress(nextProgress);
        setStatus(nextProgress.message);
        return;
      }
      if (type === 'inspected') {
        const nextInspection = payload as TerrainInspection;
        setInspection(nextInspection);
        setTopElevation(nextInspection.terrainMaxElevation.toFixed(3));
        setError(null);
        setWorking(false);
        setStatus('Terrain inspection complete.');
        return;
      }
      if (type === 'converted') {
        const nextConversion = payload as ConversionResult;
        setConversion(nextConversion);
        clearDownloadUrl(false);
        setError(null);
        setWorking(false);
        setStatus('Conversion complete. Download the STL or open it in the viewer.');
        return;
      }
      if (type === 'downloadReady') {
        if (!workerDownloadUrl || !filename) {
          setError('The STL download could not be prepared.');
          setDownloadBusy(false);
          return;
        }
        downloadUrlRef.current = workerDownloadUrl;
        setDownloadUrl(workerDownloadUrl);
        setError(null);
        setDownloadBusy(false);
        setStatus('STL download ready.');
        triggerDownload(workerDownloadUrl, filename);
        return;
      }
      if (type === 'viewerFileReady') {
        if (!payload || !('filename' in payload) || !('bytes' in payload)) {
          setError('The STL could not be opened in the viewer.');
          setOpenInViewerBusy(false);
          return;
        }
        void openPreparedStlInViewer(payload as PreparedStlPayload);
        return;
      }
      if (type === 'error') {
        setError(workerError ?? 'Unknown converter error.');
        setWorking(false);
        setDownloadBusy(false);
        setOpenInViewerBusy(false);
      }
    };

    return () => {
      clearConverterOutputState();
      worker.terminate();
    };
  }, []);

  async function makePayload(files: File[]): Promise<{
    files: UploadFilePayload[];
    transferables: ArrayBuffer[];
  }> {
    const payloadFiles = await Promise.all(
      files.map(async (file) => ({
        name: file.name,
        bytes: await file.arrayBuffer(),
      })),
    );
    return {
      files: payloadFiles,
      transferables: payloadFiles.map((file) => file.bytes),
    };
  }

  async function inspectTerrain(): Promise<void> {
    if (!workerRef.current || !terrainInput.file || !terrainInput.kind) {
      setError(terrainInputError ?? 'Choose a terrain HDF or DEM GeoTIFF first.');
      return;
    }
    setInspection(null);
    clearConverterOutputState();
    setWorking(true);
    setStatus('Reading terrain metadata...');
    const { files, transferables } = await makePayload(uploadedFiles);
    workerRef.current.postMessage(
      {
        type: 'inspect',
        files,
        terrainName: terrainInput.file.name,
        terrainKind: terrainInput.kind,
      },
      transferables,
    );
  }

  async function convertTerrain(): Promise<void> {
    if (!workerRef.current || !terrainInput.file || !terrainInput.kind) {
      setError(terrainInputError ?? 'Choose a terrain HDF or DEM GeoTIFF first.');
      return;
    }
    if (!inspection) {
      setError('Inspect Terrain before converting to STL.');
      return;
    }
    const parsedTop = Number(topElevation);
    const parsedStep = Number(sampleStep);
    if (!Number.isFinite(parsedTop)) {
      setError('Enter a numeric top elevation.');
      return;
    }
    if (!Number.isInteger(parsedStep) || parsedStep < 1) {
      setError('Sample step must be an integer greater than or equal to 1.');
      return;
    }

    clearConverterOutputState();
    setWorking(true);
    setProgress({
      percent: 0,
      step: 'resolve-raster',
      message: 'Starting terrain to STL conversion...',
    });
    setStatus('Starting terrain to STL conversion...');
    const { files, transferables } = await makePayload(uploadedFiles);
    workerRef.current.postMessage(
      {
        type: 'convert',
        files,
        terrainName: terrainInput.file.name,
        terrainKind: terrainInput.kind,
        terrainMaxElevation: inspection.terrainMaxElevation,
        topElevation: parsedTop,
        sampleStep: parsedStep,
      },
      transferables,
    );
  }

  function prepareDownload(): void {
    if (!workerRef.current || !conversion) {
      return;
    }
    if (downloadUrl) {
      triggerDownload(downloadUrl, conversion.outputFilename);
      return;
    }

    setDownloadBusy(true);
    setError(null);
    setStatus('Preparing STL download...');
    workerRef.current.postMessage({ type: 'prepareDownload' });
  }

  function openInViewer(): void {
    if (!workerRef.current || !conversion || !onOpenInViewer) {
      return;
    }

    setOpenInViewerBusy(true);
    setError(null);
    setStatus('Preparing STL for viewer...');
    workerRef.current.postMessage({ type: 'prepareViewerFile' });
  }

  async function loadExampleTerrain(): Promise<void> {
    setExampleLoadBusy(true);
    setError(null);
    setStatus('Loading example terrain files...');

    try {
      const exampleFiles = await fetchExampleTerrainFiles();
      clearConverterOutputState();
      setUploadedFiles(exampleFiles);
      setInspection(null);
      setTopElevation('');
      setSampleStep('1');
      setStatus('Example terrain files ready. Click Inspect Terrain to continue.');
    } catch (exampleError) {
      setError(exampleError instanceof Error ? exampleError.message : String(exampleError));
      setStatus('The example terrain files could not be loaded.');
    } finally {
      setExampleLoadBusy(false);
    }
  }

  const currentProgressStepIndex =
    progress?.step === 'complete'
      ? PROGRESS_STEP_ORDER.length
      : progress
        ? PROGRESS_STEP_ORDER.indexOf(progress.step as Exclude<ConversionProgressStep, 'complete'>)
        : -1;
  const title = 'Terrain to STL';
  const description =
    variant === 'modal'
      ? 'Inspect terrain inputs and generate an STL without leaving the viewer.'
      : 'Upload an HEC-RAS terrain HDF with its sibling .vrt/.tif/.tiff files, or upload one standalone DEM GeoTIFF. Conversion runs in Pyodide inside a Web Worker so the app stays static-hostable on GitHub Pages.';
  const progressSummaryLabel =
    progress?.step === 'complete'
      ? 'Conversion complete'
      : progress
        ? PROGRESS_STEP_LABELS[progress.step as Exclude<ConversionProgressStep, 'complete'>]
        : null;
  const desktopWorkflowUrl = resolveDesktopWorkflowUrl({
    override: import.meta.env.VITE_DESKTOP_WORKFLOW_URL,
    hostname: globalThis.location?.hostname,
    baseUrl: import.meta.env.BASE_URL,
  });
  const desktopWorkflowRecommendation =
    inspection === null ? null : getDesktopWorkflowRecommendation(inspection.browserLimits);

  function renderDesktopWorkflowAction(label: string, className = 'button-link button-secondary'): JSX.Element | null {
    if (!desktopWorkflowUrl) {
      return null;
    }

    return (
      <a
        className={className}
        href={desktopWorkflowUrl}
        rel="noopener noreferrer"
      >
        {label}
      </a>
    );
  }

  function renderDesktopWorkflowCallout(location: 'status' | 'details'): JSX.Element | null {
    if (!desktopWorkflowRecommendation) {
      return null;
    }

    const showSteps = location === 'status';
    return (
      <div className={`desktop-workflow-callout ${desktopWorkflowRecommendation.severity}`}>
        <p className="desktop-workflow-callout-title">{desktopWorkflowRecommendation.headline}</p>
        <p>{desktopWorkflowRecommendation.summary}</p>
        {renderDesktopWorkflowAction(
          showSteps ? 'Download Desktop Workflow ZIP' : 'Download Desktop Workflow',
          'button-link desktop-workflow-download-link',
        )}
        {showSteps ? (
          <ol className="desktop-workflow-steps">
            {DESKTOP_WORKFLOW_STEPS.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
        ) : null}
        <p className="muted">{DESKTOP_WORKFLOW_WARNING}</p>
      </div>
    );
  }

  return (
    <section className={`converter-workspace ${variant === 'modal' ? 'compact' : 'page'}`}>
      <div className="panel converter-panel">
        <div className="section-heading">
          <div className="converter-title-row">
            <h2>{title}</h2>
            <button
              type="button"
              className="button-secondary button-compact"
              onClick={() => void loadExampleTerrain()}
              disabled={controlsBusy}
            >
              {exampleLoadBusy ? 'Loading Example...' : 'Load Example to Convert'}
            </button>
          </div>
          <p className="muted">{description}</p>
        </div>

        <div className="desktop-workflow-panel">
          <div>
            <strong>Large files or offline runs</strong>
            <p className="muted">
              Download the portable Windows desktop workflow to convert large terrains offline.
              This v1 bundle is unsigned, so Windows may still warn on some systems until a future code-signed release ships.
            </p>
          </div>
          {renderDesktopWorkflowAction('Download Desktop Workflow')}
        </div>

        <label className="field">
          <span>Terrain files</span>
          <input
            type="file"
            multiple
            disabled={controlsBusy}
            onChange={(event) => {
              setUploadedFiles(Array.from(event.target.files ?? []));
              setInspection(null);
              setTopElevation('');
              setSampleStep('1');
              clearConverterOutputState();
            }}
          />
        </label>

        <div className="file-list">
          {uploadedFiles.length === 0 ? <p className="muted">No files selected yet.</p> : null}
          {uploadedFiles.map((file) => (
            <div key={`${file.name}-${file.size}`} className="file-row">
              <span>{file.name}</span>
              <span>{formatBytes(file.size)}</span>
            </div>
          ))}
        </div>

        <div className="button-row wrap">
          <button
            type="button"
            onClick={() => void inspectTerrain()}
            disabled={!terrainInput.file || controlsBusy}
          >
            Inspect Terrain
          </button>
          <button
            type="button"
            onClick={() => void convertTerrain()}
            disabled={!canConvert}
          >
            Convert to STL
          </button>
        </div>

        <div className="inline-grid">
          <label className="field compact">
            <span>STL Top Elevation (&gt;= terrain max value)</span>
            <input
              type="number"
              value={topElevation}
              onChange={(event) => setTopElevation(event.target.value)}
            />
          </label>

          <div className="field compact">
            <div className="field-label-row">
              <label htmlFor={sampleStepSelectId}>Raster sample step</label>
              <span className="field-tooltip-anchor">
                <button
                  type="button"
                  className="field-info-button"
                  aria-label="Explain raster sample step"
                  aria-describedby={sampleStepTooltipId}
                >
                  i
                </button>
                <span role="tooltip" id={sampleStepTooltipId} className="field-tooltip">
                  {SAMPLE_STEP_TOOLTIP}
                </span>
              </span>
            </div>
            <select
              id={sampleStepSelectId}
              value={sampleStep}
                    onChange={(event) => setSampleStep(event.target.value)}
              disabled={controlsBusy || !inspection}
            >
              {sampleStepOptions.map((option) => (
                <option key={option.value} value={option.value} disabled={option.disabled}>
                  {formatSampleStepOptionLabel(option)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="status-box">
          <strong>Status</strong>
          <p>{status}</p>
          {terrainInput.kind ? (
            <p className="muted">
              Input mode: {terrainInput.kind === 'hdf' ? `RAS Terrain HDF (${terrainInput.file.name})` : `DEM GeoTIFF (${terrainInput.file.name})`}
            </p>
          ) : null}
          {!terrainInput.kind && terrainInputError ? <p className="error-text">{terrainInputError}</p> : null}
          {renderDesktopWorkflowCallout('status')}
          {progress ? (
            <div className="progress-panel">
              <div className="progress-summary">
                <strong>{progress.percent}%</strong>
                <span>{progressSummaryLabel}</span>
              </div>
              <div
                className="progress-bar"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progress.percent}
              >
                <div
                  className="progress-bar-fill"
                  style={{ width: `${progress.percent}%` }}
                />
              </div>
              <ol className="progress-steps">
                {PROGRESS_STEP_ORDER.map((step, index) => {
                  const stepState =
                    currentProgressStepIndex > index
                      ? 'completed'
                      : currentProgressStepIndex === index
                        ? 'current'
                        : 'pending';

                  return (
                    <li key={step} className={`progress-step ${stepState}`}>
                      <span className="progress-step-dot" aria-hidden="true" />
                      <span>{PROGRESS_STEP_LABELS[step]}</span>
                    </li>
                  );
                })}
              </ol>
            </div>
          ) : null}
          {error ? <p className="error-text">{error}</p> : null}
        </div>
      </div>

      <div className="panel converter-panel">
        <div className="section-heading">
          <h2>{variant === 'modal' ? 'Details' : 'Conversion Details'}</h2>
          <p className="muted">
            {variant === 'modal'
              ? 'Inspection results and STL download details appear here.'
              : 'Inspection results and generated STL metadata appear here.'}
          </p>
        </div>

        {inspection ? (
          <>
            <dl className="details-grid">
              <div>
                <dt>Terrain max elevation</dt>
                <dd>{inspection.terrainMaxElevation.toFixed(6)}</dd>
              </div>
              <div>
                <dt>Resolved raster</dt>
                <dd>{inspection.resolvedRasterName}</dd>
              </div>
              <div>
                <dt>Stitch points</dt>
                <dd>{inspection.stitchPointCount}</dd>
              </div>
              <div>
                <dt>Stitch triangles</dt>
                <dd>{inspection.stitchTriangleCount}</dd>
              </div>
              <div>
                <dt>Populated stitch TIN</dt>
                <dd>{inspection.hasPopulatedStitchTin ? 'Yes' : 'No'}</dd>
              </div>
            </dl>

            <div className="browser-limits-panel">
              <div className="section-heading compact">
                <h3>Browser Limits</h3>
                <p className="muted">
                  Inspect uses raster headers first. Conversion is allowed only when the selected sample step stays inside the browser memory and STL size limits.
                </p>
              </div>

              <dl className="details-grid">
                <div>
                  <dt>Raster dimensions</dt>
                  <dd>{inspection.browserLimits.rasterWidth.toLocaleString()} x {inspection.browserLimits.rasterHeight.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Total uploaded size</dt>
                  <dd>{formatBytes(inspection.browserLimits.totalInputBytes)}</dd>
                </div>
                <div>
                  <dt>Full-decode / step 1 working set</dt>
                  <dd>{formatBytes(inspection.browserLimits.estimatedPeakWorkingSetBytes)}</dd>
                </div>
                <div>
                  <dt>Working-set limit</dt>
                  <dd>{formatBytes(inspection.browserLimits.peakWorkingSetLimitBytes)}</dd>
                </div>
                <div>
                  <dt>Browser STL limit</dt>
                  <dd>{formatBytes(inspection.browserLimits.stlSizeLimitBytes)}</dd>
                </div>
              </dl>

              {selectedSampleStepOption?.estimatedWorkingSetBytes !== null ? (
                <p className="muted">
                  Selected step {selectedSampleStepOption.value} working set: {formatBytes(selectedSampleStepOption.estimatedWorkingSetBytes)}.
                </p>
              ) : null}

              {inspection.browserLimits.blockingReasons.length > 0 ? (
                <div className="limit-callout error">
                  {inspection.browserLimits.blockingReasons.map((reason) => (
                    <p key={reason}>{reason}</p>
                  ))}
                </div>
              ) : null}

              {inspection.browserLimits.nearLimit ? (
                <div className="limit-callout warning">
                  <p>Large files near these limits may still stall or crash the browser.</p>
                </div>
              ) : null}

              {renderDesktopWorkflowCallout('details')}

              <div className="sample-step-limit-list">
                {inspection.sampleStepOptions.map((option) => (
                  <div key={option.value} className={`sample-step-limit-row ${option.disabled ? 'blocked' : 'allowed'}`}>
                    <span>Step {option.value}</span>
                    <span>
                      {option.estimatedSizeBytes === null
                        ? 'No size estimate'
                        : option.estimateKind === 'upper-bound'
                          ? `Up to ${formatBytes(option.estimatedSizeBytes)}`
                          : `~${formatBytes(option.estimatedSizeBytes)}`}
                    </span>
                    <span>
                      {option.estimatedWorkingSetBytes === null
                        ? 'No working-set estimate'
                        : `Working set ${formatBytes(option.estimatedWorkingSetBytes)}`}
                    </span>
                    <span>{option.disabled ? option.reason ?? 'Blocked' : 'Allowed in browser'}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          <p className="muted">Run "Inspect Terrain" to read the terrain metadata first.</p>
        )}

        {conversion ? (
          <>
            <dl className="details-grid">
              <div>
                <dt>Output STL</dt>
                <dd>{conversion.outputFilename}</dd>
              </div>
              <div>
                <dt>Total triangles</dt>
                <dd>{conversion.triangleCount.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Boundary wall triangles</dt>
                <dd>{conversion.wallTriangleCount.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Bridge triangles</dt>
                <dd>{conversion.stitchBridgeTriangleCount.toLocaleString()}</dd>
              </div>
              <div>
                <dt>STL size</dt>
                <dd>{formatBytes(conversion.stlSizeBytes)}</dd>
              </div>
            </dl>

            <div className="converter-result-actions">
              <button type="button" onClick={prepareDownload} disabled={outputActionBusy}>
                {downloadBusy ? 'Preparing Download...' : `Download ${conversion.outputFilename}`}
              </button>
              {onOpenInViewer ? (
                <button type="button" onClick={openInViewer} disabled={outputActionBusy}>
                  {openInViewerBusy ? 'Opening in Viewer...' : 'Open in Viewer'}
                </button>
              ) : null}
            </div>
          </>
        ) : (
          <p className="muted">Converted STL details will appear here.</p>
        )}
      </div>
    </section>
  );
}
