import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  BROWSER_PEAK_WORKING_SET_LIMIT_BYTES,
  BROWSER_STL_SIZE_LIMIT_BYTES,
  buildSampleIndices,
  buildBrowserLimits,
  buildSampleStepOptions,
  estimateAdaptiveStlUpperBoundBytes,
  estimatePeakWorkingSetBytes,
  estimateSparsePeakWorkingSetBytes,
  estimateSurfaceStlBytes,
  estimateStlUpperBoundBytes,
  SAMPLE_STEP_PRESETS,
} from './browserTerrainLimits';
import type { UploadFilePayload } from './types';
import { decodeTerrainRasterSurface, decodeTerrainRasterSurfaceForStep, parseVrt } from './browserRasterSurface';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

function repoPath(...segments: string[]): string {
  return path.join(repoRoot, ...segments);
}

function readTextFile(...segments: string[]): string {
  return fs.readFileSync(repoPath(...segments), 'utf8');
}

function fileSize(...segments: string[]): number {
  return fs.statSync(repoPath(...segments)).size;
}

function uploadFile(...segments: string[]): UploadFilePayload {
  const absolutePath = repoPath(...segments);
  const buffer = fs.readFileSync(absolutePath);
  return {
    name: path.basename(absolutePath),
    bytes: buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  };
}

describe('browser terrain limits', () => {
  it('parses the bundled example VRT header', () => {
    const vrt = parseVrt(readTextFile('example', 'example.vrt'));
    expect(vrt.width).toBe(946);
    expect(vrt.height).toBe(664);
  });

  it('parses the Steve VRT header without decoding pixels', () => {
    const vrt = parseVrt(readTextFile('example', 'Steve', 'Terrain.vrt'));
    expect(vrt.width).toBe(19928);
    expect(vrt.height).toBe(15018);
  });

  it('computes the exact shell size for a small fully valid surface', () => {
    const validMask = new Uint8Array(9).fill(1);
    expect(estimateSurfaceStlBytes({ width: 3, height: 3, validMask }, 1)).toBe(1684);
  });

  it('matches the full DEM decode exactly at sampled points for sparse high-step conversion', async () => {
    const files = [uploadFile('example', 'example.tif')];
    const full = await decodeTerrainRasterSurface(files, 'example.tif', 'dem');
    const sparse = await decodeTerrainRasterSurfaceForStep(files, 'example.tif', 'dem', 8, []);
    expect(sparse.kind).toBe('sparse');

    const sampledRows = buildSampleIndices(full.height, 8);
    const sampledCols = buildSampleIndices(full.width, 8);
    for (let rowIndex = 0; rowIndex < sampledRows.length; rowIndex += 1) {
      for (let colIndex = 0; colIndex < sampledCols.length; colIndex += 1) {
        const row = sampledRows[rowIndex];
        const col = sampledCols[colIndex];
        const fullIndex = (row * full.width) + col;
        const sparseIndex = (rowIndex * sparse.sampledCols.length) + colIndex;
        expect(sparse.coarseValidMask[sparseIndex]).toBe(full.validMask[fullIndex]);
        if (full.validMask[fullIndex] === 1) {
          expect(sparse.coarseElevations[sparseIndex]).toBe(full.elevations[fullIndex]);
        }
      }
    }
  });

  it('blocks the Steve terrain under the browser limits', () => {
    const steveInputBytes =
      fileSize('example', 'Steve', 'Terrain.hdf') +
      fileSize('example', 'Steve', 'Terrain.vrt') +
      fileSize('example', 'Steve', 'Terrain.HMiltonPIR_USGS2018_2half_grid_NAD83_2011_SPFLW_NAVD88_USFeet.tif');
    const steveCellCount = 19928 * 15018;
    const estimatedPeakWorkingSetBytes = estimatePeakWorkingSetBytes(
      steveInputBytes,
      steveCellCount,
      steveCellCount,
    );

    expect(estimatedPeakWorkingSetBytes).toBeGreaterThan(BROWSER_PEAK_WORKING_SET_LIMIT_BYTES);

    const sparseStepSixteenWorkingSetBytes = estimateSparsePeakWorkingSetBytes(
      steveInputBytes,
      19928,
      (Math.ceil(15018 / 16) * Math.ceil(19928 / 16)) + 125000,
      Math.ceil(15018 / 16),
      Math.ceil(19928 / 16),
    );
    expect(sparseStepSixteenWorkingSetBytes).toBeLessThan(BROWSER_PEAK_WORKING_SET_LIMIT_BYTES);

    const sampleStepOptions = buildSampleStepOptions(
      'upper-bound',
      (sampleStep) => estimateStlUpperBoundBytes(19928, 15018, sampleStep, 908),
      (sampleStep) => sampleStep === 1 ? estimatedPeakWorkingSetBytes : sparseStepSixteenWorkingSetBytes,
    );
    const stepOne = sampleStepOptions.find((option) => option.value === 1);
    const stepSixteen = sampleStepOptions.find((option) => option.value === 16);
    expect(stepOne?.disabled).toBe(true);
    expect(stepOne?.reason).toContain('browser working-set limit');
    expect(stepOne?.reason).toContain('browser STL size limit');
    expect(stepSixteen?.disabled).toBe(false);
    expect(stepSixteen?.estimatedWorkingSetBytes).toBeLessThan(BROWSER_PEAK_WORKING_SET_LIMIT_BYTES);

    const browserLimits = buildBrowserLimits(
      19928,
      15018,
      steveInputBytes,
      estimatedPeakWorkingSetBytes,
      sampleStepOptions,
    );

    expect(browserLimits.blockingReasons).toEqual([]);
    expect(browserLimits.nearLimit).toBe(true);
    expect(stepOne?.estimatedSizeBytes).toBeGreaterThan(BROWSER_STL_SIZE_LIMIT_BYTES);
  });

  it('keeps the smaller example inside the browser limits', () => {
    const inputBytes =
      fileSize('example', 'example.hdf') +
      fileSize('example', 'example.vrt') +
      fileSize('example', 'example.tif');
    const cellCount = 946 * 664;
    const estimatedPeakWorkingSetBytes = estimatePeakWorkingSetBytes(inputBytes, cellCount, cellCount);

    expect(estimatedPeakWorkingSetBytes).toBeLessThan(BROWSER_PEAK_WORKING_SET_LIMIT_BYTES);

    const sampleStepOptions = buildSampleStepOptions(
      'upper-bound',
      (sampleStep) => estimateStlUpperBoundBytes(946, 664, sampleStep, 0),
      () => estimatedPeakWorkingSetBytes,
    );

    expect(sampleStepOptions.every((option) => !option.disabled)).toBe(true);
    expect(sampleStepOptions[0]?.estimatedSizeBytes).toBeLessThan(BROWSER_STL_SIZE_LIMIT_BYTES);
  });

  it('blocks a sample step when the estimated STL size exceeds the browser cap', () => {
    const sampleStepOptions = buildSampleStepOptions(
      'upper-bound',
      () => BROWSER_STL_SIZE_LIMIT_BYTES + 1,
      () => BROWSER_PEAK_WORKING_SET_LIMIT_BYTES - 1,
    );

    expect(sampleStepOptions[0]?.disabled).toBe(true);
    expect(sampleStepOptions[0]?.reason).toContain('browser STL size limit');
  });

  it('supports stitched preset step 32 and enables higher adaptive steps once the browser STL cap is satisfied', () => {
    expect(SAMPLE_STEP_PRESETS).toEqual([1, 2, 4, 8, 16, 32]);

    const adaptiveMetricsByStep = {
      1: {
        coarseCellCount: 299243759,
        refinedCellCount: 0,
        transitionTriangleCount: 0,
        refinementPerimeterCellCount: 0,
        refinedVertexCount: 0,
        largestRefinementVertexCount: 0,
        regionCount: 0,
      },
      2: {
        coarseCellCount: 74816741,
        refinedCellCount: 2706,
        transitionTriangleCount: 6772,
        refinementPerimeterCellCount: 4522,
        refinedVertexCount: 3200,
        largestRefinementVertexCount: 3200,
        regionCount: 1,
      },
      4: {
        coarseCellCount: 18704822,
        refinedCellCount: 9004,
        transitionTriangleCount: 10110,
        refinementPerimeterCellCount: 8134,
        refinedVertexCount: 10120,
        largestRefinementVertexCount: 10120,
        regionCount: 1,
      },
      8: {
        coarseCellCount: 4675689,
        refinedCellCount: 32472,
        transitionTriangleCount: 17074,
        refinementPerimeterCellCount: 15382,
        refinedVertexCount: 35790,
        largestRefinementVertexCount: 35790,
        regionCount: 1,
      },
      16: {
        coarseCellCount: 1167686,
        refinedCellCount: 121920,
        transitionTriangleCount: 30774,
        refinementPerimeterCellCount: 29782,
        refinedVertexCount: 129660,
        largestRefinementVertexCount: 129660,
        regionCount: 1,
      },
      32: {
        coarseCellCount: 290615,
        refinedCellCount: 463936,
        transitionTriangleCount: 56633,
        refinementPerimeterCellCount: 58070,
        refinedVertexCount: 478812,
        largestRefinementVertexCount: 478812,
        regionCount: 1,
      },
    } as const;

    const sampleStepOptions = buildSampleStepOptions(
      'upper-bound',
      (sampleStep) => estimateAdaptiveStlUpperBoundBytes(
        19928,
        15018,
        sampleStep,
        908,
        adaptiveMetricsByStep[sampleStep as keyof typeof adaptiveMetricsByStep],
      ),
      (sampleStep) => sampleStep === 1
        ? BROWSER_PEAK_WORKING_SET_LIMIT_BYTES + 1
        : estimateSparsePeakWorkingSetBytes(
          fileSize('example', 'Steve', 'Terrain.hdf') +
            fileSize('example', 'Steve', 'Terrain.vrt') +
            fileSize('example', 'Steve', 'Terrain.HMiltonPIR_USGS2018_2half_grid_NAD83_2011_SPFLW_NAVD88_USFeet.tif'),
          19928,
          (Math.ceil(15018 / sampleStep) * Math.ceil(19928 / sampleStep)) +
            adaptiveMetricsByStep[sampleStep as keyof typeof adaptiveMetricsByStep].refinedVertexCount,
          Math.ceil(15018 / sampleStep),
          Math.ceil(19928 / sampleStep),
        ),
    );

    expect(sampleStepOptions.find((option) => option.value === 1)?.disabled).toBe(true);
    expect(sampleStepOptions.find((option) => option.value === 1)?.reason).toContain('browser working-set limit');
    expect(sampleStepOptions.find((option) => option.value === 4)?.disabled).toBe(true);
    expect(sampleStepOptions.find((option) => option.value === 8)?.disabled).toBe(true);
    expect(sampleStepOptions.find((option) => option.value === 16)?.disabled).toBe(false);
    expect(sampleStepOptions.find((option) => option.value === 32)?.disabled).toBe(false);
    expect(sampleStepOptions.find((option) => option.value === 16)?.estimatedSizeBytes).toBeLessThan(BROWSER_STL_SIZE_LIMIT_BYTES);
    expect(sampleStepOptions.find((option) => option.value === 32)?.estimatedWorkingSetBytes).toBeLessThan(BROWSER_PEAK_WORKING_SET_LIMIT_BYTES);
  });
});
