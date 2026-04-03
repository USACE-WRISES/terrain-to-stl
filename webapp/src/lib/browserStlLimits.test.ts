import { describe, expect, it, vi } from 'vitest';
import { BROWSER_STL_SIZE_LIMIT_BYTES } from './browserTerrainLimits';
import {
  BROWSER_VIEWER_STL_LIMIT_ERROR_MESSAGE,
  getViewerStlBrowserLimitResult,
  readViewerStlArrayBufferForBrowserViewer,
} from './browserStlLimits';

describe('browser STL viewer limits', () => {
  it('blocks files above the browser STL limit', () => {
    const result = getViewerStlBrowserLimitResult(BROWSER_STL_SIZE_LIMIT_BYTES + 1);

    expect(result.blocked).toBe(true);
    expect(result.recommendation?.severity).toBe('blocked');
    expect(result.recommendation?.headline).toContain('too large');
  });

  it('warns for files near the browser STL limit', () => {
    const result = getViewerStlBrowserLimitResult(Math.floor(BROWSER_STL_SIZE_LIMIT_BYTES * 0.8));

    expect(result.blocked).toBe(false);
    expect(result.nearLimit).toBe(true);
    expect(result.recommendation?.severity).toBe('warning');
  });

  it('does not read file bytes when the STL is blocked', async () => {
    const arrayBuffer = vi.fn(async () => new ArrayBuffer(16));

    await expect(readViewerStlArrayBufferForBrowserViewer({
      size: BROWSER_STL_SIZE_LIMIT_BYTES + 1,
      arrayBuffer,
    })).rejects.toThrow(BROWSER_VIEWER_STL_LIMIT_ERROR_MESSAGE);

    expect(arrayBuffer).not.toHaveBeenCalled();
  });

  it('reads file bytes for allowed STL files', async () => {
    const bytes = new ArrayBuffer(32);
    const arrayBuffer = vi.fn(async () => bytes);

    const result = await readViewerStlArrayBufferForBrowserViewer({
      size: 1024,
      arrayBuffer,
    });

    expect(arrayBuffer).toHaveBeenCalledTimes(1);
    expect(result.bytes).toBe(bytes);
    expect(result.limitResult.blocked).toBe(false);
    expect(result.limitResult.recommendation).toBeNull();
  });
});
