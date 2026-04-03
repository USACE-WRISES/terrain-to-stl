import { BROWSER_LIMIT_WARNING_FRACTION, BROWSER_STL_SIZE_LIMIT_BYTES } from './browserTerrainLimits';
import type { DesktopWorkflowRecommendation } from './desktopWorkflow';

export const BROWSER_VIEWER_STL_LIMIT_ERROR_MESSAGE =
  'This STL is too large for the browser viewer. Download the desktop workflow and open it there instead.';

export type ViewerStlBrowserLimitResult = {
  blocked: boolean;
  nearLimit: boolean;
  limitBytes: number;
  warningThresholdBytes: number;
  recommendation: DesktopWorkflowRecommendation | null;
};

export function getViewerStlBrowserLimitResult(
  fileSizeBytes: number,
  limitBytes = BROWSER_STL_SIZE_LIMIT_BYTES,
  warningFraction = BROWSER_LIMIT_WARNING_FRACTION,
): ViewerStlBrowserLimitResult {
  const warningThresholdBytes = Math.floor(limitBytes * warningFraction);

  if (fileSizeBytes > limitBytes) {
    return {
      blocked: true,
      nearLimit: true,
      limitBytes,
      warningThresholdBytes,
      recommendation: {
        severity: 'blocked',
        headline: 'This STL is too large for the browser viewer.',
        summary:
          'Download the portable Windows desktop workflow and open the STL there instead. It is the safer path for large local inspection.',
      },
    };
  }

  if (fileSizeBytes >= warningThresholdBytes) {
    return {
      blocked: false,
      nearLimit: true,
      limitBytes,
      warningThresholdBytes,
      recommendation: {
        severity: 'warning',
        headline: 'This STL is near the browser viewer limit.',
        summary:
          'The browser may still stall or crash while reading or previewing this file. The portable Windows desktop workflow is recommended for large STL inspection.',
      },
    };
  }

  return {
    blocked: false,
    nearLimit: false,
    limitBytes,
    warningThresholdBytes,
    recommendation: null,
  };
}

export async function readViewerStlArrayBufferForBrowserViewer(
  file: Pick<File, 'size' | 'arrayBuffer'>,
): Promise<{
  bytes: ArrayBuffer;
  limitResult: ViewerStlBrowserLimitResult;
}> {
  const limitResult = getViewerStlBrowserLimitResult(file.size);
  if (limitResult.blocked) {
    throw new Error(BROWSER_VIEWER_STL_LIMIT_ERROR_MESSAGE);
  }

  return {
    bytes: await file.arrayBuffer(),
    limitResult,
  };
}
