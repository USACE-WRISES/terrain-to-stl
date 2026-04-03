import type { BrowserLimits } from './types';

export const DESKTOP_WORKFLOW_FILENAME = 'terrain-to-stl-desktop-windows-x64.zip';
export const DESKTOP_WORKFLOW_REPOSITORY = 'USACE-WRISES/terrain-to-stl';
export const DESKTOP_WORKFLOW_WARNING =
  'Portable Windows bundle. Works offline after download. Unsigned v1 build; Windows may still warn on some systems.';
export const DESKTOP_WORKFLOW_STEPS = [
  'Download the desktop workflow ZIP.',
  'Extract the ZIP to a normal folder on disk.',
  'Open the extracted folder and run Run Desktop GUI.cmd for the native desktop workflow.',
] as const;

export type DesktopWorkflowRecommendation = {
  severity: 'blocked' | 'warning';
  headline: string;
  summary: string;
};

type DesktopWorkflowUrlOptions = {
  override?: string;
  hostname?: string;
  baseUrl?: string;
};

function latestReleaseAssetUrl(repository: string): string {
  return `https://github.com/${repository}/releases/latest/download/${DESKTOP_WORKFLOW_FILENAME}`;
}

export function resolveDesktopWorkflowUrl({
  override,
  hostname,
  baseUrl = '/',
}: DesktopWorkflowUrlOptions = {}): string {
  const trimmedOverride = override?.trim();
  if (trimmedOverride) {
    return trimmedOverride;
  }

  if (!hostname || !hostname.endsWith('.github.io')) {
    return latestReleaseAssetUrl(DESKTOP_WORKFLOW_REPOSITORY);
  }

  const owner = hostname.slice(0, -'.github.io'.length);
  const repoMatch = /^\/([^/]+)/.exec(baseUrl);
  const repo = repoMatch?.[1];
  if (!owner || !repo) {
    return latestReleaseAssetUrl(DESKTOP_WORKFLOW_REPOSITORY);
  }

  return latestReleaseAssetUrl(`${owner}/${repo}`);
}

export function getDesktopWorkflowRecommendation(
  browserLimits: BrowserLimits,
): DesktopWorkflowRecommendation | null {
  if (browserLimits.blockingReasons.length > 0) {
    return {
      severity: 'blocked',
      headline: 'This terrain is too large for the browser workflow.',
      summary:
        'Use the portable Windows desktop workflow instead. It runs offline after download and is the safer path for large local conversions.',
    };
  }

  if (browserLimits.nearLimit) {
    return {
      severity: 'warning',
      headline: 'This terrain is near the browser limits.',
      summary:
        'The browser may still stall or crash before conversion finishes. The portable Windows desktop workflow is recommended for large local runs.',
    };
  }

  return null;
}
