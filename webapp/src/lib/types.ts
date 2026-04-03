export type UploadFilePayload = {
  name: string;
  bytes: ArrayBuffer;
};

export type ConversionProgressStep =
  | 'resolve-raster'
  | 'load-runtime'
  | 'load-packages'
  | 'load-bridge'
  | 'prepare-files'
  | 'validate-terrain'
  | 'write-surfaces'
  | 'write-stitches'
  | 'write-walls'
  | 'finalize'
  | 'complete';

export type ConversionProgress = {
  percent: number;
  step: ConversionProgressStep;
  message: string;
};

export type ProfileMode = 'bottom' | 'full';
export type ProfileSource = 'preview' | 'exact';
export type ClipAxis = 'x' | 'y' | 'z';
export type ClipKeep = 'greater' | 'less';
export type ViewPreset = 'top' | 'bottom' | 'isometric';
export type ActiveView = ViewPreset | 'custom';

export type ViewRequest = {
  preset: ViewPreset;
  id: number;
};

export type ClipState = {
  enabled: boolean;
  axis: ClipAxis;
  value: number;
  keep: ClipKeep;
};

export type MeshBounds = {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  size: [number, number, number];
};

export type SampleStepOption = {
  value: number;
  estimatedSizeBytes: number | null;
  estimatedWorkingSetBytes: number | null;
  estimateKind: 'exact' | 'upper-bound';
  disabled: boolean;
  reason: string | null;
};

export type BrowserLimits = {
  rasterWidth: number;
  rasterHeight: number;
  totalInputBytes: number;
  estimatedPeakWorkingSetBytes: number;
  peakWorkingSetLimitBytes: number;
  stlSizeLimitBytes: number;
  nearLimit: boolean;
  blockingReasons: string[];
};

export type TerrainInspection = {
  terrainMaxElevation: number;
  resolvedRasterName: string;
  stitchPointCount: number;
  stitchTriangleCount: number;
  hasPopulatedStitchTin: boolean;
  browserLimits: BrowserLimits;
  sampleStepOptions: SampleStepOption[];
};

export type ConversionResult = {
  outputFilename: string;
  terrainMaxElevation: number;
  resolvedRasterName: string;
  triangleCount: number;
  wallTriangleCount: number;
  stitchPointCount: number;
  stitchTriangleCount: number;
  stitchBridgeTriangleCount: number;
  stlSizeBytes: number;
};

export type PreparedStlPayload = {
  filename: string;
  bytes: ArrayBuffer;
};

export type MeshLoadResult = {
  meshName: string;
  bounds: MeshBounds;
  triangleCount: number;
  previewTriangleCount: number;
  renderMode: 'preview' | 'exact';
  previewPositions: ArrayBuffer;
};

export type ProfileResult = {
  source: ProfileSource;
  sampleCount: number;
  horizontalDistance: number;
  start: [number, number];
  end: [number, number];
  bottomDistances: number[];
  bottomValues: Array<number | null>;
  topDistances: number[];
  topValues: Array<number | null>;
  bottomMin: number | null;
  bottomMax: number | null;
  topMin: number | null;
  topMax: number | null;
};

export type WorkerStatus = {
  message: string;
};
