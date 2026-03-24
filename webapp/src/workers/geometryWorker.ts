/// <reference lib="webworker" />

import type {
  ClipState,
  MeshBounds,
  MeshLoadResult,
  ProfileMode,
  ProfileResult,
  ProfileSource,
} from '../lib/types';

declare const self: DedicatedWorkerGlobalScope;

type LoadMeshMessage = {
  type: 'loadMesh';
  name: string;
  bytes: ArrayBuffer;
};

type ComputeProfileMessage = {
  type: 'computeProfile';
  requestId: number;
  start: [number, number];
  end: [number, number];
  clip: ClipState;
  mode: ProfileMode;
  exact: boolean;
};

type DisposeMeshMessage = {
  type: 'disposeMesh';
};

type WorkerMessage = LoadMeshMessage | ComputeProfileMessage | DisposeMeshMessage;

type SpatialIndex = {
  columns: number;
  rows: number;
  minX: number;
  minY: number;
  cellWidth: number;
  cellHeight: number;
  cells: Map<number, number[]>;
};

type WorkerMeshState = {
  meshName: string;
  bounds: MeshBounds;
  exactPositions: Float32Array;
  previewPositions: Float32Array;
  renderMode: 'preview' | 'exact';
  previewIndex: SpatialIndex;
  exactIndex: SpatialIndex | null;
};

const PREVIEW_TRIANGLE_LIMIT = 300_000;
const PREVIEW_TARGET_TRIANGLES = 120_000;
const EPSILON = 1e-6;

let meshState: WorkerMeshState | null = null;

function postStatus(message: string): void {
  self.postMessage({ type: 'status', payload: { message } });
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function parseAsciiStl(text: string): Float32Array {
  const vertices: number[] = [];
  const vertexPattern = /vertex\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)/gi;
  let match = vertexPattern.exec(text);
  while (match) {
    vertices.push(Number(match[1]), Number(match[2]), Number(match[3]));
    match = vertexPattern.exec(text);
  }

  if (vertices.length === 0 || vertices.length % 9 !== 0) {
    throw new Error('The STL file could not be parsed as ASCII STL.');
  }

  return new Float32Array(vertices);
}

function parseBinaryStl(buffer: ArrayBuffer): Float32Array | null {
  if (buffer.byteLength < 84) {
    return null;
  }
  const view = new DataView(buffer);
  const triangleCount = view.getUint32(80, true);
  const expectedSize = 84 + triangleCount * 50;
  if (expectedSize !== buffer.byteLength) {
    return null;
  }

  const positions = new Float32Array(triangleCount * 9);
  let inputOffset = 84;
  let outputOffset = 0;
  for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex += 1) {
    inputOffset += 12;
    for (let vertexIndex = 0; vertexIndex < 3; vertexIndex += 1) {
      positions[outputOffset] = view.getFloat32(inputOffset, true);
      positions[outputOffset + 1] = view.getFloat32(inputOffset + 4, true);
      positions[outputOffset + 2] = view.getFloat32(inputOffset + 8, true);
      outputOffset += 3;
      inputOffset += 12;
    }
    inputOffset += 2;
  }

  return positions;
}

function parseStl(buffer: ArrayBuffer): Float32Array {
  const binary = parseBinaryStl(buffer);
  if (binary) {
    return binary;
  }

  const decoder = new TextDecoder();
  return parseAsciiStl(decoder.decode(buffer));
}

function buildBounds(positions: Float32Array): MeshBounds {
  if (positions.length === 0) {
    throw new Error('The STL file does not contain any triangles.');
  }

  let minX = positions[0];
  let minY = positions[1];
  let minZ = positions[2];
  let maxX = positions[0];
  let maxY = positions[1];
  let maxZ = positions[2];

  for (let index = 3; index < positions.length; index += 3) {
    const x = positions[index];
    const y = positions[index + 1];
    const z = positions[index + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }

  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    center: [(minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5],
    size: [maxX - minX, maxY - minY, maxZ - minZ],
  };
}

function buildPreviewPositions(positions: Float32Array, bounds: MeshBounds): Float32Array {
  const triangleCount = positions.length / 9;
  if (triangleCount <= PREVIEW_TRIANGLE_LIMIT) {
    return positions.slice();
  }

  const xRange = Math.max(bounds.size[0], EPSILON);
  const yRange = Math.max(bounds.size[1], EPSILON);
  const zRange = Math.max(bounds.size[2], EPSILON);
  const maxRange = Math.max(xRange, yRange);
  const xDivisions = clamp(Math.round((128 * xRange) / maxRange), 48, 128);
  const yDivisions = clamp(Math.round((128 * yRange) / maxRange), 48, 128);
  const zDivisions = 32;
  const quantizedVertexMap = new Map<number, number>();
  const vertexSumX: number[] = [];
  const vertexSumY: number[] = [];
  const vertexSumZ: number[] = [];
  const vertexCount: number[] = [];
  const triangleSet = new Set<bigint>();
  const triangleVertices: number[] = [];
  const xScale = xDivisions + 1;
  const yScale = yDivisions + 1;
  const zScale = zDivisions + 1;
  const vertexIdBitShift = 20n;

  const getVertexId = (x: number, y: number, z: number): number => {
    const qx = clamp(Math.round(((x - bounds.min[0]) / xRange) * xDivisions), 0, xDivisions);
    const qy = clamp(Math.round(((y - bounds.min[1]) / yRange) * yDivisions), 0, yDivisions);
    const qz = clamp(Math.round(((z - bounds.min[2]) / zRange) * zDivisions), 0, zDivisions);
    const key = qx + xScale * (qy + yScale * qz);
    const existing = quantizedVertexMap.get(key);
    if (existing !== undefined) {
      vertexSumX[existing] += x;
      vertexSumY[existing] += y;
      vertexSumZ[existing] += z;
      vertexCount[existing] += 1;
      return existing;
    }
    const id = vertexSumX.length;
    vertexSumX.push(x);
    vertexSumY.push(y);
    vertexSumZ.push(z);
    vertexCount.push(1);
    quantizedVertexMap.set(key, id);
    return id;
  };

  const makeTriangleKey = (a: number, b: number, c: number): bigint => {
    let x = a;
    let y = b;
    let z = c;
    if (x > y) {
      const temp = x;
      x = y;
      y = temp;
    }
    if (y > z) {
      const temp = y;
      y = z;
      z = temp;
    }
    if (x > y) {
      const temp = x;
      x = y;
      y = temp;
    }
    return (BigInt(x) << (vertexIdBitShift * 2n)) | (BigInt(y) << vertexIdBitShift) | BigInt(z);
  };

  for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex += 1) {
    const offset = triangleIndex * 9;
    const a = getVertexId(positions[offset], positions[offset + 1], positions[offset + 2]);
    const b = getVertexId(positions[offset + 3], positions[offset + 4], positions[offset + 5]);
    const c = getVertexId(positions[offset + 6], positions[offset + 7], positions[offset + 8]);

    if (a === b || b === c || a === c) {
      continue;
    }

    const triangleKey = makeTriangleKey(a, b, c);
    if (triangleSet.has(triangleKey)) {
      continue;
    }
    triangleSet.add(triangleKey);
    triangleVertices.push(a, b, c);
  }

  const representativePositions = new Float32Array(vertexSumX.length * 3);
  for (let vertexId = 0; vertexId < vertexSumX.length; vertexId += 1) {
    representativePositions[vertexId * 3] = vertexSumX[vertexId] / vertexCount[vertexId];
    representativePositions[vertexId * 3 + 1] = vertexSumY[vertexId] / vertexCount[vertexId];
    representativePositions[vertexId * 3 + 2] = vertexSumZ[vertexId] / vertexCount[vertexId];
  }

  const previewPositions = new Float32Array(triangleVertices.length * 3);
  let writeOffset = 0;
  for (let index = 0; index < triangleVertices.length; index += 3) {
    const a = triangleVertices[index];
    const b = triangleVertices[index + 1];
    const c = triangleVertices[index + 2];

    const ax = representativePositions[a * 3];
    const ay = representativePositions[a * 3 + 1];
    const az = representativePositions[a * 3 + 2];
    const bx = representativePositions[b * 3];
    const by = representativePositions[b * 3 + 1];
    const bz = representativePositions[b * 3 + 2];
    const cx = representativePositions[c * 3];
    const cy = representativePositions[c * 3 + 1];
    const cz = representativePositions[c * 3 + 2];

    const abx = bx - ax;
    const aby = by - ay;
    const abz = bz - az;
    const acx = cx - ax;
    const acy = cy - ay;
    const acz = cz - az;
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    if (Math.abs(nx) <= EPSILON && Math.abs(ny) <= EPSILON && Math.abs(nz) <= EPSILON) {
      continue;
    }

    previewPositions[writeOffset] = ax;
    previewPositions[writeOffset + 1] = ay;
    previewPositions[writeOffset + 2] = az;
    previewPositions[writeOffset + 3] = bx;
    previewPositions[writeOffset + 4] = by;
    previewPositions[writeOffset + 5] = bz;
    previewPositions[writeOffset + 6] = cx;
    previewPositions[writeOffset + 7] = cy;
    previewPositions[writeOffset + 8] = cz;
    writeOffset += 9;
  }

  return previewPositions.slice(0, writeOffset);
}

function buildSpatialIndex(positions: Float32Array, bounds: MeshBounds): SpatialIndex {
  const triangleCount = positions.length / 9;
  const cellEstimate = clamp(Math.round(Math.sqrt(triangleCount / 128)), 32, 128);
  const columns = cellEstimate;
  const rows = cellEstimate;
  const cellWidth = Math.max(bounds.size[0] / columns, EPSILON);
  const cellHeight = Math.max(bounds.size[1] / rows, EPSILON);
  const cells = new Map<number, number[]>();

  for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex += 1) {
    const offset = triangleIndex * 9;
    const x0 = positions[offset];
    const y0 = positions[offset + 1];
    const x1 = positions[offset + 3];
    const y1 = positions[offset + 4];
    const x2 = positions[offset + 6];
    const y2 = positions[offset + 7];

    const minColumn = clamp(Math.floor((Math.min(x0, x1, x2) - bounds.min[0]) / cellWidth), 0, columns - 1);
    const maxColumn = clamp(Math.floor((Math.max(x0, x1, x2) - bounds.min[0]) / cellWidth), 0, columns - 1);
    const minRow = clamp(Math.floor((Math.min(y0, y1, y2) - bounds.min[1]) / cellHeight), 0, rows - 1);
    const maxRow = clamp(Math.floor((Math.max(y0, y1, y2) - bounds.min[1]) / cellHeight), 0, rows - 1);

    for (let row = minRow; row <= maxRow; row += 1) {
      for (let column = minColumn; column <= maxColumn; column += 1) {
        const cellKey = row * columns + column;
        const bucket = cells.get(cellKey);
        if (bucket) {
          bucket.push(triangleIndex);
        } else {
          cells.set(cellKey, [triangleIndex]);
        }
      }
    }
  }

  return {
    columns,
    rows,
    minX: bounds.min[0],
    minY: bounds.min[1],
    cellWidth,
    cellHeight,
    cells,
  };
}

function triangleVisibleByClip(
  x0: number,
  y0: number,
  z0: number,
  x1: number,
  y1: number,
  z1: number,
  x2: number,
  y2: number,
  z2: number,
  clip: ClipState,
): boolean {
  if (!clip.enabled) {
    return true;
  }

  const axisIndex = clip.axis === 'x' ? 0 : clip.axis === 'y' ? 1 : 2;
  const keepGreater = clip.keep === 'greater';
  const values = [
    axisIndex === 0 ? x0 : axisIndex === 1 ? y0 : z0,
    axisIndex === 0 ? x1 : axisIndex === 1 ? y1 : z1,
    axisIndex === 0 ? x2 : axisIndex === 1 ? y2 : z2,
  ];

  return values.some((value) =>
    keepGreater ? value >= clip.value - EPSILON : value <= clip.value + EPSILON,
  );
}

function pointInTriangleAndZ(
  x: number,
  y: number,
  x0: number,
  y0: number,
  z0: number,
  x1: number,
  y1: number,
  z1: number,
  x2: number,
  y2: number,
  z2: number,
): number | null {
  const determinant = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
  if (Math.abs(determinant) <= EPSILON) {
    return null;
  }

  const a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / determinant;
  const b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / determinant;
  const c = 1 - a - b;
  if (a < -EPSILON || b < -EPSILON || c < -EPSILON) {
    return null;
  }

  return a * z0 + b * z1 + c * z2;
}

function summarize(values: Array<number | null>): [number | null, number | null] {
  let minValue = Number.POSITIVE_INFINITY;
  let maxValue = Number.NEGATIVE_INFINITY;
  let found = false;

  for (const value of values) {
    if (value === null) {
      continue;
    }
    found = true;
    if (value < minValue) minValue = value;
    if (value > maxValue) maxValue = value;
  }

  if (!found) {
    return [null, null];
  }

  return [minValue, maxValue];
}

function computeProfile(
  positions: Float32Array,
  index: SpatialIndex,
  start: [number, number],
  end: [number, number],
  clip: ClipState,
  mode: ProfileMode,
  source: ProfileSource,
): ProfileResult {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const horizontalDistance = Math.hypot(dx, dy);
  const sampleCount = source === 'exact' ? 768 : 320;
  const bottomDistances: number[] = [];
  const bottomValues: Array<number | null> = [];
  const topDistances: number[] = [];
  const topValues: Array<number | null> = [];

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    const t = sampleCount === 1 ? 0 : sampleIndex / (sampleCount - 1);
    const x = start[0] + dx * t;
    const y = start[1] + dy * t;
    const distance = horizontalDistance * t;

    const column = clamp(Math.floor((x - index.minX) / index.cellWidth), 0, index.columns - 1);
    const row = clamp(Math.floor((y - index.minY) / index.cellHeight), 0, index.rows - 1);
    const cellTriangles = index.cells.get(row * index.columns + column) ?? [];

    let minZ = Number.POSITIVE_INFINITY;
    let maxZ = Number.NEGATIVE_INFINITY;
    let found = false;

    for (const triangleIndex of cellTriangles) {
      const offset = triangleIndex * 9;
      const x0 = positions[offset];
      const y0 = positions[offset + 1];
      const z0 = positions[offset + 2];
      const x1 = positions[offset + 3];
      const y1 = positions[offset + 4];
      const z1 = positions[offset + 5];
      const x2 = positions[offset + 6];
      const y2 = positions[offset + 7];
      const z2 = positions[offset + 8];

      if (
        !triangleVisibleByClip(
          x0,
          y0,
          z0,
          x1,
          y1,
          z1,
          x2,
          y2,
          z2,
          clip,
        )
      ) {
        continue;
      }

      const z = pointInTriangleAndZ(x, y, x0, y0, z0, x1, y1, z1, x2, y2, z2);
      if (z === null) {
        continue;
      }

      found = true;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }

    bottomDistances.push(distance);
    bottomValues.push(found ? minZ : null);
    if (mode === 'full') {
      topDistances.push(distance);
      topValues.push(found ? maxZ : null);
    }
  }

  const [bottomMin, bottomMax] = summarize(bottomValues);
  const [topMin, topMax] = mode === 'full' ? summarize(topValues) : [null, null];

  return {
    source,
    sampleCount,
    horizontalDistance,
    start,
    end,
    bottomDistances,
    bottomValues,
    topDistances,
    topValues,
    bottomMin,
    bottomMax,
    topMin,
    topMax,
  };
}

function handleLoadMesh(message: LoadMeshMessage): void {
  postStatus('Parsing STL mesh...');
  const exactPositions = parseStl(message.bytes);
  const bounds = buildBounds(exactPositions);

  postStatus('Building preview mesh...');
  const previewPositions = buildPreviewPositions(exactPositions, bounds);
  const previewIndex = buildSpatialIndex(previewPositions, bounds);

  meshState = {
    meshName: message.name,
    bounds,
    exactPositions,
    previewPositions,
    renderMode: previewPositions.length === exactPositions.length ? 'exact' : 'preview',
    previewIndex,
    exactIndex: null,
  };

  const previewBuffer = previewPositions.slice().buffer;
  const result: MeshLoadResult = {
    meshName: message.name,
    bounds,
    triangleCount: exactPositions.length / 9,
    previewTriangleCount: previewPositions.length / 9,
    renderMode: meshState.renderMode,
    previewPositions: previewBuffer,
  };

  self.postMessage(
    {
      type: 'meshLoaded',
      payload: result,
    },
    [previewBuffer],
  );
}

function handleComputeProfile(message: ComputeProfileMessage): void {
  if (!meshState) {
    throw new Error('No STL mesh is loaded.');
  }

  const source: ProfileSource = message.exact ? 'exact' : 'preview';
  const positions = message.exact ? meshState.exactPositions : meshState.previewPositions;
  let index = message.exact ? meshState.exactIndex : meshState.previewIndex;
  if (!index) {
    postStatus('Building exact mesh lookup...');
    index = buildSpatialIndex(meshState.exactPositions, meshState.bounds);
    meshState.exactIndex = index;
  }

  postStatus(message.exact ? 'Computing exact profile...' : 'Computing preview profile...');
  const profile = computeProfile(
    positions,
    index,
    message.start,
    message.end,
    message.clip,
    message.mode,
    source,
  );
  self.postMessage({
    type: 'profileComputed',
    requestId: message.requestId,
    payload: profile,
  });
}

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  try {
    if (event.data.type === 'loadMesh') {
      handleLoadMesh(event.data);
      return;
    }
    if (event.data.type === 'disposeMesh') {
      meshState = null;
      self.postMessage({ type: 'disposed' });
      return;
    }
    handleComputeProfile(event.data);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ type: 'error', error: message });
  }
};

export {};
