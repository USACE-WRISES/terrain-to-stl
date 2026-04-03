import { useEffect, useMemo, useRef, useState, type DragEvent as ReactDragEvent } from 'react';
import type { ActiveElement, ChartOptions, ScriptableContext } from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import { BROWSER_STL_SIZE_LIMIT_BYTES } from '../lib/browserTerrainLimits';
import {
  BROWSER_VIEWER_STL_LIMIT_ERROR_MESSAGE,
  getViewerStlBrowserLimitResult,
  readViewerStlArrayBufferForBrowserViewer,
  type ViewerStlBrowserLimitResult,
} from '../lib/browserStlLimits';
import { ConverterWorkspace } from '../components/ConverterWorkspace';
import { MeshScene } from '../components/MeshScene';
import {
  DESKTOP_WORKFLOW_STEPS,
  DESKTOP_WORKFLOW_WARNING,
  resolveDesktopWorkflowUrl,
} from '../lib/desktopWorkflow';
import { fetchExampleStlFile } from '../lib/exampleAssets';
import type {
  ActiveView,
  ClipAxis,
  ClipKeep,
  ClipState,
  MeshBounds,
  MeshLoadResult,
  ProfileResult,
  ViewPreset,
  ViewRequest,
} from '../lib/types';

const DEFAULT_OPACITY = 0.8;
const DEFAULT_VERTICAL_EXAGGERATION = 3;
const INITIAL_STATUS = 'Load an STL to inspect it in the browser.';
const INITIAL_VIEW_REQUEST: ViewRequest = { preset: 'top', id: 0 };
const VERTICAL_EXAGGERATION_OPTIONS = [1, 2, 3, 5, 10];

function formatMaybe(value: number | null): string {
  return value === null ? 'n/a' : value.toFixed(2);
}

function formatViewLabel(view: ActiveView): string {
  return view === 'custom' ? 'Custom' : `${view[0].toUpperCase()}${view.slice(1)}`;
}

function mid(bounds: MeshBounds, axis: 0 | 1 | 2): number {
  return (bounds.min[axis] + bounds.max[axis]) * 0.5;
}

function defaultProfileLine(bounds: MeshBounds): [[number, number], [number, number]] {
  return [
    [bounds.min[0] + bounds.size[0] * 0.12, bounds.min[1] + bounds.size[1] * 0.18],
    [bounds.max[0] - bounds.size[0] * 0.12, bounds.max[1] - bounds.size[1] * 0.18],
  ];
}

function axisBounds(bounds: MeshBounds, axis: ClipAxis): [number, number] {
  if (axis === 'x') return [bounds.min[0], bounds.max[0]];
  if (axis === 'y') return [bounds.min[1], bounds.max[1]];
  return [bounds.min[2], bounds.max[2]];
}

function isFileDrag(event: ReactDragEvent<HTMLElement>): boolean {
  return Array.from(event.dataTransfer.types).includes('Files');
}

function isStlFile(file: File): boolean {
  return file.name.toLowerCase().endsWith('.stl');
}

function triggerDownload(url: string, filename: string): void {
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = 'noopener';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function formatFileSizeMb(byteCount: number): string {
  return `${(byteCount / (1024 * 1024)).toFixed(byteCount >= 10 * 1024 * 1024 ? 1 : 2)} MB`;
}

function formatBinaryBytes(byteCount: number): string {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = byteCount;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 2)} ${units[unitIndex]}`;
}

export function ViewerPage() {
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const dragDepthRef = useRef(0);
  const pendingMeshFileRef = useRef<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [status, setStatus] = useState(INITIAL_STATUS);
  const [error, setError] = useState<string | null>(null);
  const [meshName, setMeshName] = useState<string | null>(null);
  const [loadedStlFile, setLoadedStlFile] = useState<File | null>(null);
  const [selectedStlFile, setSelectedStlFile] = useState<File | null>(null);
  const [viewerLimitResult, setViewerLimitResult] = useState<ViewerStlBrowserLimitResult | null>(null);
  const [bounds, setBounds] = useState<MeshBounds | null>(null);
  const [previewPositions, setPreviewPositions] = useState<Float32Array | null>(null);
  const [triangleCount, setTriangleCount] = useState(0);
  const [previewTriangleCount, setPreviewTriangleCount] = useState(0);
  const [renderMode, setRenderMode] = useState<'preview' | 'exact'>('preview');

  const [viewRequest, setViewRequest] = useState<ViewRequest>(INITIAL_VIEW_REQUEST);
  const [activeView, setActiveView] = useState<ActiveView>('top');
  const [opacity, setOpacity] = useState(DEFAULT_OPACITY);
  const [verticalExaggeration, setVerticalExaggeration] = useState(DEFAULT_VERTICAL_EXAGGERATION);
  const [showTopShell, setShowTopShell] = useState(false);
  const [wireframe, setWireframe] = useState(false);
  const [clipEnabled, setClipEnabled] = useState(false);
  const [clipAxis, setClipAxis] = useState<ClipAxis>('x');
  const [clipKeep, setClipKeep] = useState<ClipKeep>('greater');
  const [clipValue, setClipValue] = useState(0);
  const [profileEnabled, setProfileEnabled] = useState(false);
  const [profileStart, setProfileStart] = useState<[number, number] | null>(null);
  const [profileEnd, setProfileEnd] = useState<[number, number] | null>(null);
  const [pickMode, setPickMode] = useState<'start' | 'end' | null>(null);
  const [exactValuesEnabled, setExactValuesEnabled] = useState(false);
  const [profileResult, setProfileResult] = useState<ProfileResult | null>(null);
  const [profileBusy, setProfileBusy] = useState(false);
  const [hoveredProfileMarker, setHoveredProfileMarker] = useState<{
    sampleIndex: number;
    dataset: 'bottom' | 'top';
  } | null>(null);
  const [isProfileDrawerMinimized, setIsProfileDrawerMinimized] = useState(false);
  const [isConverterModalOpen, setIsConverterModalOpen] = useState(false);
  const [isProfileToolOpen, setIsProfileToolOpen] = useState(false);
  const [isClipToolOpen, setIsClipToolOpen] = useState(false);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [isStageDragActive, setIsStageDragActive] = useState(false);
  const [viewerExampleBusy, setViewerExampleBusy] = useState(false);
  const profileMode = showTopShell ? 'full' : 'bottom';
  const hasMesh = Boolean(bounds && previewPositions);
  const renderModeLabel = hasMesh
    ? renderMode === 'preview'
      ? `Preview (${previewTriangleCount.toLocaleString()} triangles)`
      : 'Exact'
    : 'n/a';
  const viewLabel = hasMesh ? formatViewLabel(activeView) : 'n/a';
  const shellModeLabel = hasMesh ? (profileMode === 'bottom' ? 'Bottom Only' : 'Full Shell') : 'n/a';
  const pickModeLabel = hasMesh ? (pickMode ? `Set ${pickMode}` : 'Off') : 'n/a';
  const profileSourceLabel = hasMesh && profileEnabled ? (exactValuesEnabled ? 'Exact' : 'Preview') : 'n/a';
  const profileDistanceLabel = profileResult ? profileResult.horizontalDistance.toFixed(2) : 'n/a';
  const profileDrawerSourceLabel = exactValuesEnabled ? 'Exact' : 'Preview';
  const selectedStlFileLabel = selectedStlFile?.name ?? 'No STL selected';
  const loadedFileSizeLabel = loadedStlFile ? formatFileSizeMb(loadedStlFile.size) : 'n/a';
  const browserStlLimitLabel = formatBinaryBytes(BROWSER_STL_SIZE_LIMIT_BYTES);
  const viewerCenterClassName = [
    'viewer-center',
    profileEnabled
      ? isProfileDrawerMinimized
        ? 'viewer-center-profile-minimized'
        : 'viewer-center-profile-expanded'
      : '',
  ]
    .filter(Boolean)
    .join(' ');
  const desktopWorkflowUrl = resolveDesktopWorkflowUrl({
    override: import.meta.env.VITE_DESKTOP_WORKFLOW_URL,
    hostname: globalThis.location?.hostname,
    baseUrl: import.meta.env.BASE_URL,
  });

  const clipState = useMemo<ClipState>(
    () => ({
      enabled: clipEnabled,
      axis: clipAxis,
      value: clipValue,
      keep: clipKeep,
    }),
    [clipAxis, clipEnabled, clipKeep, clipValue],
  );

  useEffect(() => {
    const worker = new Worker(new URL('../workers/geometryWorker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent) => {
      const { type, payload, error: workerError, requestId } = event.data as {
        type: string;
        payload?: MeshLoadResult | ProfileResult | { message: string };
        error?: string;
        requestId?: number;
      };

      if (type === 'status') {
        setStatus((payload as { message: string }).message);
        return;
      }
      if (type === 'disposed') {
        setStatus(INITIAL_STATUS);
        return;
      }
      if (type === 'meshLoaded') {
        const result = payload as MeshLoadResult;
        const positions = new Float32Array(result.previewPositions);
        const nextLoadedFile = pendingMeshFileRef.current;
        setBounds(result.bounds);
        setMeshName(result.meshName);
        setLoadedStlFile(nextLoadedFile);
        setSelectedStlFile(nextLoadedFile);
        pendingMeshFileRef.current = null;
        setPreviewPositions(positions);
        setTriangleCount(result.triangleCount);
        setPreviewTriangleCount(result.previewTriangleCount);
        setRenderMode(result.renderMode);
        setViewRequest((current) => ({ preset: 'top', id: current.id + 1 }));
        setActiveView('top');
        setOpacity(DEFAULT_OPACITY);
        setVerticalExaggeration(DEFAULT_VERTICAL_EXAGGERATION);
        setShowTopShell(false);
        setWireframe(false);
        setClipEnabled(false);
        setIsClipToolOpen(false);
        setClipAxis('x');
        setClipKeep('greater');
        setClipValue(mid(result.bounds, 0));
        setProfileEnabled(false);
        setIsProfileToolOpen(false);
        setIsProfileDrawerMinimized(false);
        setExactValuesEnabled(false);
        const [start, end] = defaultProfileLine(result.bounds);
        setProfileStart(start);
        setProfileEnd(end);
        setProfileResult(null);
        setHoveredProfileMarker(null);
        setPickMode(null);
        setError(null);
        setStatus(
          result.renderMode === 'preview'
            ? 'STL loaded. Preview mesh ready in the browser viewer.'
            : 'STL loaded. Exact mesh ready in the browser viewer.',
        );
        setProfileBusy(false);
        return;
      }
      if (type === 'profileComputed') {
        if (requestId !== requestIdRef.current) {
          return;
        }
        setProfileResult(payload as ProfileResult);
        setProfileBusy(false);
        return;
      }
      if (type === 'error') {
        pendingMeshFileRef.current = null;
        setError(workerError ?? 'Unknown viewer error.');
        setProfileBusy(false);
      }
    };

    return () => {
      worker.terminate();
    };
  }, []);

  useEffect(() => {
    if (!bounds) {
      return;
    }
    const axisIndex = clipAxis === 'x' ? 0 : clipAxis === 'y' ? 1 : 2;
    const [minimum, maximum] = axisBounds(bounds, clipAxis);
    setClipValue((current) => Math.min(maximum, Math.max(minimum, current || mid(bounds, axisIndex))));
  }, [bounds, clipAxis]);

  useEffect(() => {
    if (!workerRef.current || !profileEnabled || !profileStart || !profileEnd) {
      return;
    }

    setProfileBusy(true);
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    const timeoutId = window.setTimeout(() => {
      workerRef.current?.postMessage({
        type: 'computeProfile',
        requestId,
        start: profileStart,
        end: profileEnd,
        clip: clipState,
        mode: profileMode,
        exact: exactValuesEnabled,
      });
    }, 100);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [clipState, exactValuesEnabled, profileEnabled, profileEnd, profileMode, profileStart]);

  useEffect(() => {
    if (profileEnabled) {
      setIsProfileDrawerMinimized(false);
      return;
    }
    setIsProfileDrawerMinimized(false);
    setPickMode(null);
  }, [profileEnabled]);

  useEffect(() => {
    setHoveredProfileMarker(null);
  }, [profileBusy, profileEnabled, profileResult, profileStart, profileEnd]);

  useEffect(() => {
    if (!isConverterModalOpen) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeConverterModal();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isConverterModalOpen]);

  function resetViewerState(disposeMesh: boolean): void {
    requestIdRef.current += 1;
    if (disposeMesh) {
      workerRef.current?.postMessage({ type: 'disposeMesh' });
    }
    setStatus(INITIAL_STATUS);
    setError(null);
    setMeshName(null);
    setLoadedStlFile(null);
    setSelectedStlFile(null);
    setViewerLimitResult(null);
    setBounds(null);
    setPreviewPositions(null);
    setTriangleCount(0);
    setPreviewTriangleCount(0);
    setRenderMode('preview');
    setViewRequest((current) => ({ preset: 'top', id: current.id + 1 }));
    setActiveView('top');
    setOpacity(DEFAULT_OPACITY);
    setVerticalExaggeration(DEFAULT_VERTICAL_EXAGGERATION);
    setShowTopShell(false);
    setWireframe(false);
    setClipEnabled(false);
    setIsClipToolOpen(false);
    setClipAxis('x');
    setClipKeep('greater');
    setClipValue(0);
    setProfileEnabled(false);
    setIsProfileToolOpen(false);
    setExactValuesEnabled(false);
    setProfileStart(null);
    setProfileEnd(null);
    setPickMode(null);
    setProfileResult(null);
    setProfileBusy(false);
    setHoveredProfileMarker(null);
    setIsProfileDrawerMinimized(false);
    setFileInputKey((current) => current + 1);
    pendingMeshFileRef.current = null;
  }

  function renderDesktopWorkflowAction(label: string): JSX.Element | null {
    if (!desktopWorkflowUrl) {
      return null;
    }

    return (
      <a
        className="button-link desktop-workflow-download-link"
        href={desktopWorkflowUrl}
        rel="noopener noreferrer"
      >
        {label}
      </a>
    );
  }

  function renderViewerLimitCallout(): JSX.Element | null {
    if (!viewerLimitResult?.recommendation) {
      return null;
    }

    const fileDetails = selectedStlFile
      ? `Selected file: ${selectedStlFile.name} (${formatBinaryBytes(selectedStlFile.size)}).`
      : null;

    return (
      <div className={`desktop-workflow-callout ${viewerLimitResult.recommendation.severity}`}>
        <p className="desktop-workflow-callout-title">{viewerLimitResult.recommendation.headline}</p>
        <p>{viewerLimitResult.recommendation.summary}</p>
        <p className="muted">
          Browser STL limit: {formatBinaryBytes(viewerLimitResult.limitBytes)}.
          {fileDetails ? ` ${fileDetails}` : ''}
        </p>
        {renderDesktopWorkflowAction(
          viewerLimitResult.recommendation.severity === 'blocked'
            ? 'Download Desktop Workflow ZIP'
            : 'Download Desktop Workflow',
        )}
        {viewerLimitResult.recommendation.severity === 'blocked' ? (
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

  async function loadMesh(file: File): Promise<void> {
    if (!workerRef.current) {
      return;
    }
    if (!isStlFile(file)) {
      throw new Error('The viewer only accepts .stl files.');
    }
    const nextViewerLimitResult = getViewerStlBrowserLimitResult(file.size);
    setSelectedStlFile(file);
    setViewerLimitResult(nextViewerLimitResult.recommendation ? nextViewerLimitResult : null);
    setStatus(
      nextViewerLimitResult.blocked
        ? 'The selected STL is too large for the browser viewer.'
        : nextViewerLimitResult.nearLimit
          ? 'Loading large STL into browser worker...'
          : 'Loading STL into browser worker...',
    );
    setError(null);
    pendingMeshFileRef.current = file;
    try {
      const { bytes } = await readViewerStlArrayBufferForBrowserViewer(file);
      workerRef.current.postMessage(
        {
          type: 'loadMesh',
          name: file.name,
          bytes,
        },
        [bytes],
      );
    } catch (loadError) {
      pendingMeshFileRef.current = null;
      if (loadError instanceof Error && loadError.message === BROWSER_VIEWER_STL_LIMIT_ERROR_MESSAGE) {
        setStatus('The selected STL is too large for the browser viewer.');
      }
      throw loadError;
    }
  }

  function resetClip(): void {
    if (!bounds) {
      return;
    }
    setClipEnabled(false);
    setClipAxis('x');
    setClipKeep('greater');
    setClipValue(mid(bounds, 0));
  }

  function resetProfile(): void {
    if (!bounds) {
      return;
    }
    const [start, end] = defaultProfileLine(bounds);
    setProfileStart(start);
    setProfileEnd(end);
    setProfileResult(null);
    setHoveredProfileMarker(null);
    setPickMode(null);
  }

  function requestView(preset: ViewPreset): void {
    setViewRequest((current) => ({ preset, id: current.id + 1 }));
    setActiveView(preset);
  }

  function clearAll(): void {
    if (!hasMesh) {
      return;
    }
    resetViewerState(true);
  }

  function closeConverterModal(): void {
    setIsConverterModalOpen(false);
  }

  function handlePickPoint(mode: 'start' | 'end', point: [number, number]): void {
    if (mode === 'start') {
      setProfileStart(point);
    } else {
      setProfileEnd(point);
    }
    setHoveredProfileMarker(null);
    setPickMode(null);
  }

  function resetStageDragState(): void {
    dragDepthRef.current = 0;
    setIsStageDragActive(false);
  }

  function downloadLoadedStl(): void {
    if (!loadedStlFile) {
      return;
    }

    const objectUrl = URL.createObjectURL(loadedStlFile);
    triggerDownload(objectUrl, loadedStlFile.name);
    window.setTimeout(() => {
      URL.revokeObjectURL(objectUrl);
    }, 0);
  }

  async function openConvertedStlInViewer(file: File): Promise<void> {
    await loadMesh(file);
    closeConverterModal();
  }

  function openViewerFilePicker(): void {
    fileInputRef.current?.click();
  }

  async function loadExampleMesh(): Promise<void> {
    setViewerExampleBusy(true);
    setError(null);

    try {
      const exampleFile = await fetchExampleStlFile();
      await loadMesh(exampleFile);
    } catch (exampleError) {
      setError(exampleError instanceof Error ? exampleError.message : String(exampleError));
      setStatus('The example STL could not be loaded.');
    } finally {
      setViewerExampleBusy(false);
    }
  }

  function handleStageDragEnter(event: ReactDragEvent<HTMLDivElement>): void {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    dragDepthRef.current += 1;
    setIsStageDragActive(true);
  }

  function handleStageDragOver(event: ReactDragEvent<HTMLDivElement>): void {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
    if (!isStageDragActive) {
      setIsStageDragActive(true);
    }
  }

  function handleStageDragLeave(event: ReactDragEvent<HTMLDivElement>): void {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setIsStageDragActive(false);
    }
  }

  async function handleStageDrop(event: ReactDragEvent<HTMLDivElement>): Promise<void> {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    resetStageDragState();

    const droppedFiles = Array.from(event.dataTransfer.files ?? []);
    if (droppedFiles.length !== 1) {
      setError('Drop exactly one .stl file into the viewer.');
      return;
    }

    const [file] = droppedFiles;
    if (!isStlFile(file)) {
      setError('Drop exactly one .stl file into the viewer.');
      return;
    }

    try {
      await loadMesh(file);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : String(loadError));
    }
  }

  const hoveredProfileMapPoint = useMemo<[number, number] | null>(() => {
    if (!profileResult || !hoveredProfileMarker) {
      return null;
    }

    const sampleCount = Math.max(profileResult.sampleCount, 1);
    const clampedIndex = Math.max(0, Math.min(sampleCount - 1, hoveredProfileMarker.sampleIndex));
    const t = sampleCount === 1 ? 0 : clampedIndex / (sampleCount - 1);

    return [
      profileResult.start[0] + (profileResult.end[0] - profileResult.start[0]) * t,
      profileResult.start[1] + (profileResult.end[1] - profileResult.start[1]) * t,
    ];
  }, [hoveredProfileMarker, profileResult]);

  const chartData = useMemo(() => {
    if (!profileResult) {
      return null;
    }

    const markerRadius = (
      dataset: 'bottom' | 'top',
      context: ScriptableContext<'line'>,
    ): number =>
      hoveredProfileMarker?.dataset === dataset && hoveredProfileMarker.sampleIndex === context.dataIndex
        ? 5
        : 0;
    const markerBorderWidth = (
      dataset: 'bottom' | 'top',
      context: ScriptableContext<'line'>,
    ): number =>
      hoveredProfileMarker?.dataset === dataset && hoveredProfileMarker.sampleIndex === context.dataIndex
        ? 2
        : 0;

    const labels = profileResult.bottomDistances.map((distance) => distance.toFixed(1));
    const datasets = [
      {
        label: `${profileResult.source === 'exact' ? 'Exact' : 'Preview'} bottom`,
        data: profileResult.bottomValues,
        borderColor: '#ff6b5f',
        backgroundColor: 'rgba(255, 107, 95, 0.12)',
        spanGaps: false,
        pointHitRadius: 18,
        pointBackgroundColor: '#ff4d4d',
        pointBorderColor: '#fff4f2',
        pointBorderWidth: (context: ScriptableContext<'line'>) => markerBorderWidth('bottom', context),
        pointRadius: (context: ScriptableContext<'line'>) => markerRadius('bottom', context),
        pointHoverRadius: 0,
        borderWidth: 2,
      },
    ];

    if (profileMode === 'full') {
      datasets.push({
        label: `${profileResult.source === 'exact' ? 'Exact' : 'Preview'} top`,
        data: profileResult.topValues,
        borderColor: '#75a6ff',
        backgroundColor: 'rgba(117, 166, 255, 0.12)',
        spanGaps: false,
        pointHitRadius: 18,
        pointBackgroundColor: '#ff4d4d',
        pointBorderColor: '#fff4f2',
        pointBorderWidth: (context: ScriptableContext<'line'>) => markerBorderWidth('top', context),
        pointRadius: (context: ScriptableContext<'line'>) => markerRadius('top', context),
        pointHoverRadius: 0,
        borderWidth: 2,
      });
    }

    return { labels, datasets };
  }, [hoveredProfileMarker, profileMode, profileResult]);

  const chartOptions = useMemo<ChartOptions<'line'>>(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { intersect: false, mode: 'nearest' as const },
      onHover: (_event, activeElements: ActiveElement[]) => {
        const nextActive = activeElements[0];
        if (!nextActive) {
          setHoveredProfileMarker(null);
          return;
        }

        setHoveredProfileMarker({
          sampleIndex: nextActive.index,
          dataset: nextActive.datasetIndex === 1 && profileMode === 'full' ? 'top' : 'bottom',
        });
      },
      scales: {
        x: {
          title: { display: true, text: 'Distance', color: '#d7e2f2' },
          ticks: { color: '#9caecb' },
          grid: { color: 'rgba(130, 152, 188, 0.14)' },
        },
        y: {
          title: { display: true, text: 'Elevation', color: '#d7e2f2' },
          ticks: { color: '#9caecb' },
          grid: { color: 'rgba(130, 152, 188, 0.14)' },
        },
      },
      plugins: {
        legend: {
          display: true,
          labels: { color: '#d7e2f2' },
        },
        tooltip: {
          intersect: false,
        },
      },
    }),
    [profileMode],
  );

  return (
    <>
      <section className="viewer-studio">
        <aside className="viewer-sidebar viewer-sidebar-left">
          <div className="section-heading viewer-sidebar-title">
            <h2>Terrain to STL</h2>
          </div>

          <div className="sidebar-divider" />

          <p className="muted viewer-sidebar-copy">Convert terrain (DEM,RAS HDF) to STL.</p>

          <div className="button-row wrap">
            <button
              type="button"
              className="button-compact viewer-open-converter-button"
              onClick={() => setIsConverterModalOpen(true)}
            >
              Open Converter
            </button>
          </div>

          <div className="sidebar-divider" />

          <div className="field">
            <div className="field-label-row">
              <span>Upload STL to view</span>
              <button
                type="button"
                className="button-secondary button-compact"
                onClick={() => void loadExampleMesh()}
                disabled={viewerExampleBusy}
              >
                {viewerExampleBusy ? 'Loading Example...' : 'Load Example'}
              </button>
            </div>
            <div className="viewer-upload-control">
              <button
                type="button"
                className="button-secondary button-compact"
                onClick={openViewerFilePicker}
              >
                Choose File
              </button>
              <div className="viewer-upload-filename" title={selectedStlFileLabel}>
                {selectedStlFileLabel}
              </div>
            </div>
            <p className="muted">Browser STL limit: {browserStlLimitLabel}. Use the desktop workflow for larger local inspection.</p>
            <input
              ref={fileInputRef}
              key={fileInputKey}
              type="file"
              accept=".stl"
              className="viewer-hidden-file-input"
              onChange={(event) => {
                const file = event.target.files?.[0];
                event.target.value = '';
                if (file) {
                  void loadMesh(file).catch((loadError) =>
                    setError(loadError instanceof Error ? loadError.message : String(loadError)),
                  );
                }
              }}
            />
          </div>

          <div className="status-box">
            <strong>Status</strong>
            <p>{status}</p>
            {renderViewerLimitCallout()}
            {meshName ? (
              <dl className="sidebar-meta-list">
                <div>
                  <dt>Loaded file</dt>
                  <dd>{meshName}</dd>
                </div>
                <div>
                  <dt>File size</dt>
                  <dd>{loadedFileSizeLabel}</dd>
                </div>
                <div>
                  <dt>Exact triangles</dt>
                  <dd>{triangleCount.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Preview triangles</dt>
                  <dd>{previewTriangleCount.toLocaleString()}</dd>
                </div>
              </dl>
            ) : null}
            <div className="sidebar-divider" />
            <dl className="sidebar-meta-grid">
              <div>
                <dt>Render mode</dt>
                <dd>{renderModeLabel}</dd>
              </div>
              <div>
                <dt>View</dt>
                <dd>{viewLabel}</dd>
              </div>
              <div>
                <dt>Shell mode</dt>
                <dd>{shellModeLabel}</dd>
              </div>
              <div>
                <dt>Pick mode</dt>
                <dd>{pickModeLabel}</dd>
              </div>
              <div>
                <dt>Profile source</dt>
                <dd>{profileSourceLabel}</dd>
              </div>
              <div>
                <dt>Distance</dt>
                <dd>{profileDistanceLabel}</dd>
              </div>
            </dl>
            {error ? <p className="error-text">{error}</p> : null}
          </div>

          <button
            type="button"
            className="button-secondary"
            onClick={downloadLoadedStl}
            disabled={!loadedStlFile}
          >
            Download STL
          </button>

          <button type="button" className="button-secondary" onClick={clearAll} disabled={!hasMesh}>
            Clear All
          </button>
        </aside>

        <section className={viewerCenterClassName}>
          <div
            className={`viewer-stage-shell ${isStageDragActive ? 'drag-active' : ''}`}
            onDragEnter={handleStageDragEnter}
            onDragOver={handleStageDragOver}
            onDragLeave={handleStageDragLeave}
            onDrop={(event) => {
              void handleStageDrop(event);
            }}
          >
            <MeshScene
              positions={previewPositions}
              bounds={bounds}
              opacity={opacity}
              verticalExaggeration={verticalExaggeration}
              wireframe={wireframe}
              clip={clipState}
              viewRequest={viewRequest}
              activeView={activeView}
              showTopShell={showTopShell}
              profileEnabled={profileEnabled}
              profileStart={profileStart}
              profileEnd={profileEnd}
              hoveredProfileMapPoint={hoveredProfileMapPoint}
              pickMode={pickMode}
              onPickPoint={handlePickPoint}
              onRequestView={requestView}
              onActiveViewChange={setActiveView}
            />
            {isStageDragActive ? (
              <div className="viewer-drop-overlay" aria-hidden="true">
                <strong>Drop one STL here</strong>
                <span>It will replace the current mesh.</span>
              </div>
            ) : null}
          </div>

          {profileEnabled ? (
            isProfileDrawerMinimized ? (
              <button
                type="button"
                className="viewer-profile-tab"
                onClick={() => setIsProfileDrawerMinimized(false)}
              >
                Open Profile Drawer
              </button>
            ) : (
              <div className="viewer-profile-drawer">
                <div className="viewer-profile-drawer-header">
                  <div className="viewer-profile-heading">
                    <div className="viewer-profile-title-row">
                      <strong>{`Profile (Source: ${profileDrawerSourceLabel})`}</strong>
                      <span className="field-tooltip-anchor">
                        <button
                          type="button"
                          className="field-info-button"
                          aria-label="Explain profile hover marker"
                        >
                          i
                        </button>
                        <span role="tooltip" className="field-tooltip">
                          Hover over the profile view to see location on the profile line.
                        </span>
                      </span>
                    </div>
                    <span>
                      {profileBusy
                        ? 'Computing profile...'
                        : profileStart && profileEnd
                          ? 'Adjust the line to resample the terrain section.'
                          : 'Pick start and end points to sample a terrain section.'}
                    </span>
                  </div>
                  <div className="viewer-profile-header-actions">
                    <button
                      type="button"
                      className={exactValuesEnabled ? 'viewer-toggle-button active' : 'viewer-toggle-button'}
                      onClick={() => setExactValuesEnabled((current) => !current)}
                      disabled={!bounds}
                    >
                      {`Exact Values: ${exactValuesEnabled ? 'On' : 'Off'}`}
                    </button>
                    <button
                      type="button"
                      className="button-secondary"
                      onClick={() => setIsProfileDrawerMinimized(true)}
                    >
                      Minimize
                    </button>
                  </div>
                </div>

                <div className="viewer-profile-drawer-body">
                  <div
                    className="viewer-profile-chart"
                    onMouseLeave={() => setHoveredProfileMarker(null)}
                  >
                    {chartData ? (
                      <Line data={chartData} options={chartOptions} />
                    ) : (
                      <p className="muted">Enable the profile tool to compute a terrain section.</p>
                    )}
                  </div>

                  <dl className="viewer-profile-metrics">
                    <div>
                      <dt>Horizontal distance</dt>
                      <dd>{profileResult ? profileResult.horizontalDistance.toFixed(2) : 'n/a'}</dd>
                    </div>
                    <div>
                      <dt>Bottom min/max</dt>
                      <dd>
                        {profileResult
                          ? `${formatMaybe(profileResult.bottomMin)} / ${formatMaybe(profileResult.bottomMax)}`
                          : 'n/a'}
                      </dd>
                    </div>
                    {profileMode === 'full' ? (
                      <div>
                        <dt>Top min/max</dt>
                        <dd>
                          {profileResult
                            ? `${formatMaybe(profileResult.topMin)} / ${formatMaybe(profileResult.topMax)}`
                            : 'n/a'}
                        </dd>
                      </div>
                    ) : null}
                    <div>
                      <dt>Profile points</dt>
                      <dd>{profileStart && profileEnd ? 'Ready' : 'Pick start and end'}</dd>
                    </div>
                  </dl>
                </div>
              </div>
            )
          ) : null}
        </section>

        <aside className="viewer-sidebar viewer-sidebar-right">
          <div className="sidebar-section">
            <div className="section-heading compact">
              <h3>Inspect</h3>
            </div>

            <div className={`sidebar-expander ${isProfileToolOpen ? 'open' : ''}`}>
              <button
                type="button"
                className="sidebar-expander-toggle"
                onClick={() => setIsProfileToolOpen((current) => !current)}
                aria-expanded={isProfileToolOpen}
              >
                <span>Profile Tool</span>
                <span className="sidebar-expander-icon">{isProfileToolOpen ? '−' : '+'}</span>
              </button>

              {isProfileToolOpen ? (
                <div className="sidebar-expander-body">
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={profileEnabled}
                      onChange={(event) => {
                        const nextEnabled = event.target.checked;
                        setProfileEnabled(nextEnabled);
                        if (nextEnabled) {
                          setIsProfileToolOpen(true);
                        }
                      }}
                      disabled={!bounds}
                    />
                    <span>Enable Profile Tool</span>
                  </label>

                  <div className="button-row wrap">
                    <button type="button" onClick={() => setPickMode('start')} disabled={!bounds || !profileEnabled}>
                      Pick Start
                    </button>
                    <button type="button" onClick={() => setPickMode('end')} disabled={!bounds || !profileEnabled}>
                      Pick End
                    </button>
                    <button type="button" className="button-secondary" onClick={resetProfile} disabled={!bounds}>
                      Reset Profile
                    </button>
                  </div>
                </div>
              ) : null}
            </div>

            <div className={`sidebar-expander ${isClipToolOpen ? 'open' : ''}`}>
              <button
                type="button"
                className="sidebar-expander-toggle"
                onClick={() => setIsClipToolOpen((current) => !current)}
                aria-expanded={isClipToolOpen}
              >
                <span>Clip Tool</span>
                <span className="sidebar-expander-icon">{isClipToolOpen ? '−' : '+'}</span>
              </button>

              {isClipToolOpen ? (
                <div className="sidebar-expander-body">
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={clipEnabled}
                      onChange={(event) => {
                        const nextEnabled = event.target.checked;
                        setClipEnabled(nextEnabled);
                        if (nextEnabled) {
                          setIsClipToolOpen(true);
                        }
                      }}
                      disabled={!bounds}
                    />
                    <span>Enable Clip Tool</span>
                  </label>

                  <div className="inline-grid">
                    <label className="field compact">
                      <span>Axis</span>
                      <select
                        value={clipAxis}
                        onChange={(event) => setClipAxis(event.target.value as ClipAxis)}
                        disabled={!bounds}
                      >
                        <option value="x">X</option>
                        <option value="y">Y</option>
                        <option value="z">Z</option>
                      </select>
                    </label>

                    <label className="field compact">
                      <span>Keep</span>
                      <select
                        value={clipKeep}
                        onChange={(event) => setClipKeep(event.target.value as ClipKeep)}
                        disabled={!bounds}
                      >
                        <option value="greater">Greater</option>
                        <option value="less">Less</option>
                      </select>
                    </label>
                  </div>

                  <label className="field">
                    <span>Clip value</span>
                    <input
                      type="range"
                      min={bounds ? axisBounds(bounds, clipAxis)[0] : 0}
                      max={bounds ? axisBounds(bounds, clipAxis)[1] : 1}
                      step="0.01"
                      value={clipValue}
                      onChange={(event) => setClipValue(Number(event.target.value))}
                      disabled={!bounds}
                    />
                  </label>

                  <button
                    type="button"
                    className="button-secondary"
                    onClick={resetClip}
                    disabled={!bounds}
                  >
                    Reset Clip
                  </button>
                </div>
              ) : null}
            </div>
          </div>

          <div className="sidebar-section">
            <div className="section-heading compact">
              <h3>Display</h3>
              <p className="muted">Tune shell visibility and surface presentation.</p>
            </div>

            <label className="field">
              <span>Vertical exaggeration</span>
              <select
                value={verticalExaggeration}
                onChange={(event) => setVerticalExaggeration(Number(event.target.value))}
                disabled={!bounds}
              >
                {VERTICAL_EXAGGERATION_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}x
                  </option>
                ))}
              </select>
            </label>

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={showTopShell}
                onChange={(event) => setShowTopShell(event.target.checked)}
                disabled={!bounds}
              />
              <span>Show Top Shell</span>
            </label>

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={wireframe}
                onChange={(event) => setWireframe(event.target.checked)}
                disabled={!bounds}
              />
              <span>Wireframe</span>
            </label>

            <label className="field">
              <span>Surface opacity: {opacity.toFixed(2)}</span>
              <input
                type="range"
                min="0.15"
                max="1"
                step="0.01"
                value={opacity}
                onChange={(event) => setOpacity(Number(event.target.value))}
                disabled={!bounds}
              />
            </label>
          </div>
        </aside>
      </section>

      {isConverterModalOpen ? (
        <div
          className="modal-backdrop"
          onClick={closeConverterModal}
          role="presentation"
        >
          <div
            className="modal-shell"
            role="dialog"
            aria-modal="true"
            aria-label="Terrain to STL Converter"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header modal-header-compact">
              <button
                type="button"
                className="button-secondary"
                onClick={closeConverterModal}
              >
                Close
              </button>
            </div>

            <div className="modal-body">
              <ConverterWorkspace variant="modal" onOpenInViewer={openConvertedStlInViewer} />
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
