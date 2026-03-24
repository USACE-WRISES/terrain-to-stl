import type { MeshBounds } from './types';

type TerrainLegendStop = {
  t: number;
  hex: string;
};

const TERRAIN_COLOR_HEXES = [
  '#2b78c2',
  '#2ea34a',
  '#9fca3c',
  '#e6df2a',
  '#f1bb1e',
  '#df7d12',
  '#bb2d22',
  '#7b7b7b',
  '#b4b4b4',
  '#dedede',
] as const;

export const TERRAIN_LEGEND_STOPS: readonly TerrainLegendStop[] = TERRAIN_COLOR_HEXES.map(
  (hex, index, colors) => ({
    t: index / (colors.length - 1),
    hex,
  }),
);

const TERRAIN_STOPS = [
  { t: 0.0, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[0]) },
  { t: 1 / 9, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[1]) },
  { t: 2 / 9, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[2]) },
  { t: 3 / 9, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[3]) },
  { t: 4 / 9, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[4]) },
  { t: 5 / 9, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[5]) },
  { t: 6 / 9, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[6]) },
  { t: 7 / 9, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[7]) },
  { t: 8 / 9, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[8]) },
  { t: 1.0, color: hexToLinearRgb(TERRAIN_COLOR_HEXES[9]) },
] as const;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function srgbChannelToLinear(channel: number): number {
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
}

function hexToLinearRgb(hex: string): [number, number, number] {
  const value = hex.replace('#', '');
  if (value.length !== 6) {
    throw new Error(`Expected 6-digit hex color, received "${hex}".`);
  }

  const r = parseInt(value.slice(0, 2), 16) / 255;
  const g = parseInt(value.slice(2, 4), 16) / 255;
  const b = parseInt(value.slice(4, 6), 16) / 255;
  return [srgbChannelToLinear(r), srgbChannelToLinear(g), srgbChannelToLinear(b)];
}

function interpolateChannel(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function colorAt(normalizedT: number): [number, number, number] {
  const t = clamp(normalizedT, 0, 1);
  for (let index = 0; index < TERRAIN_STOPS.length - 1; index += 1) {
    const left = TERRAIN_STOPS[index];
    const right = TERRAIN_STOPS[index + 1];
    if (t > right.t) {
      continue;
    }
    const localT = right.t === left.t ? 0 : (t - left.t) / (right.t - left.t);
    return [
      interpolateChannel(left.color[0], right.color[0], localT),
      interpolateChannel(left.color[1], right.color[1], localT),
      interpolateChannel(left.color[2], right.color[2], localT),
    ];
  }
  const last = TERRAIN_STOPS[TERRAIN_STOPS.length - 1];
  return [last.color[0], last.color[1], last.color[2]];
}

export function buildTerrainColorArray(
  positions: Float32Array,
  bounds: MeshBounds,
): Float32Array {
  const colors = new Float32Array(positions.length);
  const minZ = bounds.min[2];
  const zRange = Math.max(bounds.max[2] - bounds.min[2], 1e-6);

  for (let index = 0; index < positions.length; index += 3) {
    const z = positions[index + 2];
    const [r, g, b] = colorAt((z - minZ) / zRange);
    colors[index] = r;
    colors[index + 1] = g;
    colors[index + 2] = b;
  }

  return colors;
}
