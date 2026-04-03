import type { BrowserLimits, SampleStepOption } from './types';

export const SAMPLE_STEP_PRESETS = [1, 2, 4, 8, 16, 32] as const;
export const BROWSER_PEAK_WORKING_SET_LIMIT_BYTES = Math.floor(2.5 * 1024 * 1024 * 1024);
export const BROWSER_STL_SIZE_LIMIT_BYTES = 256 * 1024 * 1024;
export const BROWSER_LIMIT_WARNING_FRACTION = 0.8;
export const WORKING_SET_LIMIT_REASON = 'exceeds browser working-set limit';
export const STL_SIZE_LIMIT_REASON = 'exceeds browser STL size limit';

const STL_HEADER_BYTES = 84;
const STL_TRIANGLE_BYTES = 50;

type EstimateKind = SampleStepOption['estimateKind'];

export type SurfaceLike = {
  width: number;
  height: number;
  validMask: ArrayLike<number | boolean>;
};

export type SampleStepEvaluation = {
  value: number;
  estimatedSizeBytes: number | null;
  estimatedWorkingSetBytes: number | null;
  estimateKind: EstimateKind;
  disabled: boolean;
  reason: string | null;
};

export type AdaptiveStitchMetrics = {
  coarseCellCount: number;
  refinedCellCount: number;
  transitionTriangleCount: number;
  refinementPerimeterCellCount: number;
  refinedVertexCount: number;
  largestRefinementVertexCount: number;
  regionCount: number;
};

function isNearLimit(value: number, limit: number): boolean {
  return value >= (limit * BROWSER_LIMIT_WARNING_FRACTION);
}

function appendReason(reasons: string[], reason: string): void {
  if (!reasons.includes(reason)) {
    reasons.push(reason);
  }
}

export function buildSampleIndices(size: number, step: number): number[] {
  const indices: number[] = [];
  for (let index = 0; index < size; index += step) {
    indices.push(index);
  }
  if (indices.length === 0 || indices[indices.length - 1] !== size - 1) {
    indices.push(size - 1);
  }
  return indices;
}

export function encodeVertex(width: number, row: number, col: number): number {
  return (row * width) + col;
}

export function updateBoundaryEdges(boundaryEdges: Set<string>, a: number, b: number): void {
  const key = a < b ? `${a}:${b}` : `${b}:${a}`;
  if (boundaryEdges.has(key)) {
    boundaryEdges.delete(key);
    return;
  }
  boundaryEdges.add(key);
}

export function estimatePeakWorkingSetBytes(
  totalInputBytes: number,
  largestSourceCellCount: number,
  targetCellCount: number,
): number {
  return Math.ceil((totalInputBytes + (largestSourceCellCount * 4) + (targetCellCount * 10)) * 1.2);
}

export function estimateSparsePeakWorkingSetBytes(
  totalInputBytes: number,
  largestDecodeWindowCellCount: number,
  storedCellCount: number,
  sampledRowCount = 0,
  sampledColumnCount = 0,
): number {
  return Math.ceil(
    (
      totalInputBytes +
      (largestDecodeWindowCellCount * 4) +
      (storedCellCount * 10) +
      ((sampledRowCount + sampledColumnCount) * 4)
    ) * 1.2,
  );
}

export function estimateStlUpperBoundBytes(
  width: number,
  height: number,
  sampleStep: number,
  stitchTriangleUpperBound: number,
): number {
  const sampledRowCount = buildSampleIndices(height, sampleStep).length;
  const sampledColumnCount = buildSampleIndices(width, sampleStep).length;
  const topBottomTriangles = 4 * (sampledRowCount - 1) * (sampledColumnCount - 1);
  const wallTriangles = 4 * ((sampledRowCount - 1) + (sampledColumnCount - 1));
  const totalTriangles = topBottomTriangles + wallTriangles + stitchTriangleUpperBound;
  return STL_HEADER_BYTES + (totalTriangles * STL_TRIANGLE_BYTES);
}

export function estimateAdaptiveStlUpperBoundBytes(
  width: number,
  height: number,
  sampleStep: number,
  stitchTriangleUpperBound: number,
  metrics: AdaptiveStitchMetrics,
): number {
  const sampledRowCount = buildSampleIndices(height, sampleStep).length;
  const sampledColumnCount = buildSampleIndices(width, sampleStep).length;
  const topBottomTriangles =
    (4 * (metrics.coarseCellCount + metrics.refinedCellCount)) +
    (2 * metrics.transitionTriangleCount) +
    (2 * stitchTriangleUpperBound);
  const boundaryEdgeUpperBound =
    (2 * ((sampledRowCount - 1) + (sampledColumnCount - 1))) +
    metrics.refinementPerimeterCellCount;
  const wallTriangles = boundaryEdgeUpperBound * 2;
  return STL_HEADER_BYTES + ((topBottomTriangles + wallTriangles) * STL_TRIANGLE_BYTES);
}

export function estimateSurfaceStlBytes(
  surface: SurfaceLike,
  sampleStep: number,
): number {
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
        Boolean(surface.validMask[(row0 * surface.width) + col0]) &&
        Boolean(surface.validMask[(row1 * surface.width) + col0]) &&
        Boolean(surface.validMask[(row1 * surface.width) + col1])
      ) {
        updateBoundaryEdges(boundaryEdges, a, b);
        updateBoundaryEdges(boundaryEdges, b, c);
        updateBoundaryEdges(boundaryEdges, c, a);
        rasterTriangleCount += 1;
      }

      if (
        Boolean(surface.validMask[(row0 * surface.width) + col0]) &&
        Boolean(surface.validMask[(row1 * surface.width) + col1]) &&
        Boolean(surface.validMask[(row0 * surface.width) + col1])
      ) {
        updateBoundaryEdges(boundaryEdges, a, c);
        updateBoundaryEdges(boundaryEdges, c, d);
        updateBoundaryEdges(boundaryEdges, d, a);
        rasterTriangleCount += 1;
      }
    }
  }

  const totalTriangles = (rasterTriangleCount * 2) + (boundaryEdges.size * 2);
  return STL_HEADER_BYTES + (totalTriangles * STL_TRIANGLE_BYTES);
}

export function evaluateSampleStep(
  value: number,
  estimateKind: EstimateKind,
  estimatedSizeBytes: number,
  estimatedWorkingSetBytes: number,
): SampleStepEvaluation {
  const reasons: string[] = [];
  if (estimatedWorkingSetBytes > BROWSER_PEAK_WORKING_SET_LIMIT_BYTES) {
    appendReason(reasons, WORKING_SET_LIMIT_REASON);
  }
  if (estimatedSizeBytes > BROWSER_STL_SIZE_LIMIT_BYTES) {
    appendReason(reasons, STL_SIZE_LIMIT_REASON);
  }

  return {
    value,
    estimatedSizeBytes,
    estimatedWorkingSetBytes,
    estimateKind,
    disabled: reasons.length > 0,
    reason: reasons.length > 0 ? reasons.join('; ') : null,
  };
}

export function buildSampleStepOptions(
  estimateKind: EstimateKind,
  estimateSizeBytes: (sampleStep: number) => number,
  estimateWorkingSetBytes: (sampleStep: number) => number,
): SampleStepOption[] {
  return SAMPLE_STEP_PRESETS.map((value) => evaluateSampleStep(
    value,
    estimateKind,
    estimateSizeBytes(value),
    estimateWorkingSetBytes(value),
  ));
}

export function buildBrowserLimits(
  rasterWidth: number,
  rasterHeight: number,
  totalInputBytes: number,
  estimatedPeakWorkingSetBytes: number,
  sampleStepOptions: SampleStepOption[],
): BrowserLimits {
  const blockingReasons: string[] = [];

  if (sampleStepOptions.length > 0 && sampleStepOptions.every((option) => option.disabled)) {
    appendReason(
      blockingReasons,
      'Browser conversion is blocked for every available sample step.',
    );
  }

  const nearLimit =
    isNearLimit(estimatedPeakWorkingSetBytes, BROWSER_PEAK_WORKING_SET_LIMIT_BYTES) ||
    sampleStepOptions.some(
      (option) =>
        (
          (option.estimatedSizeBytes !== null &&
            isNearLimit(option.estimatedSizeBytes, BROWSER_STL_SIZE_LIMIT_BYTES)) ||
          (option.estimatedWorkingSetBytes !== null &&
            isNearLimit(option.estimatedWorkingSetBytes, BROWSER_PEAK_WORKING_SET_LIMIT_BYTES))
        ),
    );

  return {
    rasterWidth,
    rasterHeight,
    totalInputBytes,
    estimatedPeakWorkingSetBytes,
    peakWorkingSetLimitBytes: BROWSER_PEAK_WORKING_SET_LIMIT_BYTES,
    stlSizeLimitBytes: BROWSER_STL_SIZE_LIMIT_BYTES,
    nearLimit,
    blockingReasons,
  };
}
