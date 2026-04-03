import { describe, expect, it } from 'vitest';
import {
  DESKTOP_WORKFLOW_FILENAME,
  DESKTOP_WORKFLOW_REPOSITORY,
  getDesktopWorkflowRecommendation,
  resolveDesktopWorkflowUrl,
} from './desktopWorkflow';
import type { BrowserLimits } from './types';

function browserLimits(overrides: Partial<BrowserLimits> = {}): BrowserLimits {
  return {
    rasterWidth: 100,
    rasterHeight: 100,
    totalInputBytes: 1024,
    estimatedPeakWorkingSetBytes: 2048,
    peakWorkingSetLimitBytes: 4096,
    stlSizeLimitBytes: 8192,
    nearLimit: false,
    blockingReasons: [],
    ...overrides,
  };
}

describe('resolveDesktopWorkflowUrl', () => {
  it('prefers the explicit environment override', () => {
    expect(resolveDesktopWorkflowUrl({
      override: 'https://downloads.example.com/terrain.zip',
      hostname: 'owner.github.io',
      baseUrl: '/terrain-to-stl/',
    })).toBe('https://downloads.example.com/terrain.zip');
  });

  it('derives the latest release asset URL from GitHub Pages hosting', () => {
    expect(resolveDesktopWorkflowUrl({
      hostname: 'owner.github.io',
      baseUrl: '/terrain-to-stl/',
    })).toBe(`https://github.com/owner/terrain-to-stl/releases/latest/download/${DESKTOP_WORKFLOW_FILENAME}`);
  });

  it('falls back to the project release asset on localhost', () => {
    expect(resolveDesktopWorkflowUrl({
      hostname: 'localhost',
      baseUrl: '/',
    })).toBe(`https://github.com/${DESKTOP_WORKFLOW_REPOSITORY}/releases/latest/download/${DESKTOP_WORKFLOW_FILENAME}`);
  });

  it('falls back to the project release asset when host data is missing', () => {
    expect(resolveDesktopWorkflowUrl()).toBe(
      `https://github.com/${DESKTOP_WORKFLOW_REPOSITORY}/releases/latest/download/${DESKTOP_WORKFLOW_FILENAME}`,
    );
  });
});

describe('getDesktopWorkflowRecommendation', () => {
  it('marks blocked browser runs as desktop-required', () => {
    const recommendation = getDesktopWorkflowRecommendation(browserLimits({
      blockingReasons: ['Browser conversion is blocked for every available sample step.'],
    }));

    expect(recommendation?.severity).toBe('blocked');
    expect(recommendation?.headline).toContain('too large');
  });

  it('recommends the desktop workflow for near-limit terrains', () => {
    const recommendation = getDesktopWorkflowRecommendation(browserLimits({
      nearLimit: true,
    }));

    expect(recommendation?.severity).toBe('warning');
    expect(recommendation?.summary).toContain('recommended');
  });

  it('returns no recommendation for comfortably supported browser runs', () => {
    expect(getDesktopWorkflowRecommendation(browserLimits())).toBeNull();
  });
});
