import {
  GizmoHelper,
  GizmoViewport,
  Line,
  OrthographicCamera,
  PerspectiveCamera,
  TrackballControls,
} from '@react-three/drei';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  MOUSE,
  MathUtils,
  Mesh as MeshImpl,
  Plane,
  Raycaster,
  Vector2,
  Vector3,
  type OrthographicCamera as OrthographicCameraImpl,
  type PerspectiveCamera as PerspectiveCameraImpl,
} from 'three';
import type { TrackballControls as TrackballControlsImpl } from 'three-stdlib';
import { TERRAIN_LEGEND_STOPS, buildTerrainColorArray } from '../lib/terrainColors';
import type {
  ActiveView,
  ClipState,
  MeshBounds,
  ViewPreset,
  ViewRequest,
} from '../lib/types';

const PROFILE_LINE_COLOR = '#00e5ff';
const VIEW_TOLERANCE_ORTHO = 0.998;
const VIEW_TOLERANCE_ISO = 0.995;
const ISO_CAMERA_OFFSET = new Vector3(1, -1, 0.7).normalize();
const ISO_VIEW_DIRECTION = ISO_CAMERA_OFFSET.clone().multiplyScalar(-1).normalize();
const TOP_VIEW_DIRECTION = new Vector3(0, 0, -1);
const BOTTOM_VIEW_DIRECTION = new Vector3(0, 0, 1);
const WORLD_UP = new Vector3(0, 1, 0);
const SCALE_BAR_MAX_WIDTH = 148;
const DRAG_PLANE_NORMAL = new Vector3(0, 0, 1);
const HOVER_TERRAIN_EPSILON = 1e-4;
const PROFILE_HOVER_MARKER_COLOR = '#ff4d4d';
const MAP_HELP_TOOLTIP =
  'Left click and drag to pan. Right click and drag to rotate. Use Profile Tool to sample elevations. Hovered elevation is shown in the bottom status bar.';

type ScaleBarInfo = {
  label: string;
  widthPx: number;
};

type HoverElevationInfo = {
  elevationLabel: string;
};

type SceneCamera = PerspectiveCameraImpl | OrthographicCameraImpl;
type TrackballControlsInternal = TrackballControlsImpl & {
  STATE: { NONE: number };
  _state: number;
  _keyState: number;
  _movePrev: Vector2;
  _moveCurr: Vector2;
  _zoomStart: Vector2;
  _zoomEnd: Vector2;
  _panStart: Vector2;
  _panEnd: Vector2;
  _lastAngle: number;
  domElement?: HTMLElement;
  onPointerMove: (event: PointerEvent) => void;
  onPointerUp: (event: PointerEvent) => void;
};

type MeshSceneProps = {
  positions: Float32Array | null;
  bounds: MeshBounds | null;
  opacity: number;
  verticalExaggeration: number;
  wireframe: boolean;
  clip: ClipState;
  viewRequest: ViewRequest;
  activeView: ActiveView;
  showTopShell: boolean;
  profileEnabled: boolean;
  profileStart: [number, number] | null;
  profileEnd: [number, number] | null;
  hoveredProfileMapPoint: [number, number] | null;
  pickMode: 'start' | 'end' | null;
  onPickPoint: (mode: 'start' | 'end', point: [number, number]) => void;
  onRequestView: (preset: ViewPreset) => void;
  onActiveViewChange: (view: ActiveView) => void;
};

function formatNumber(value: number): string {
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  if (Math.abs(value) >= 100) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  if (Math.abs(value) >= 1) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function makeClipPlane(clip: ClipState): Plane | null {
  if (!clip.enabled) {
    return null;
  }

  if (clip.axis === 'x') {
    return clip.keep === 'greater'
      ? new Plane(new Vector3(1, 0, 0), -clip.value)
      : new Plane(new Vector3(-1, 0, 0), clip.value);
  }
  if (clip.axis === 'y') {
    return clip.keep === 'greater'
      ? new Plane(new Vector3(0, 1, 0), -clip.value)
      : new Plane(new Vector3(0, -1, 0), clip.value);
  }
  return clip.keep === 'greater'
    ? new Plane(new Vector3(0, 0, 1), -clip.value)
    : new Plane(new Vector3(0, 0, -1), clip.value);
}

function topPlaneTolerance(bounds: MeshBounds): number {
  return Math.max(1e-3, bounds.size[2] * 1e-6);
}

function filterTopCapPositions(positions: Float32Array, bounds: MeshBounds): Float32Array {
  const tolerance = topPlaneTolerance(bounds);
  const keptValues: number[] = [];

  for (let index = 0; index < positions.length; index += 9) {
    const z0 = positions[index + 2];
    const z1 = positions[index + 5];
    const z2 = positions[index + 8];
    const touchesTop =
      Math.abs(z0 - bounds.max[2]) <= tolerance &&
      Math.abs(z1 - bounds.max[2]) <= tolerance &&
      Math.abs(z2 - bounds.max[2]) <= tolerance;

    if (touchesTop) {
      continue;
    }

    keptValues.push(
      positions[index],
      positions[index + 1],
      z0,
      positions[index + 3],
      positions[index + 4],
      z1,
      positions[index + 6],
      positions[index + 7],
      z2,
    );
  }

  return new Float32Array(keptValues);
}

function localizeBounds(bounds: MeshBounds, verticalExaggeration: number): MeshBounds {
  const minZ = (bounds.min[2] - bounds.center[2]) * verticalExaggeration;
  const maxZ = (bounds.max[2] - bounds.center[2]) * verticalExaggeration;
  return {
    min: [
      bounds.min[0] - bounds.center[0],
      bounds.min[1] - bounds.center[1],
      minZ,
    ],
    max: [
      bounds.max[0] - bounds.center[0],
      bounds.max[1] - bounds.center[1],
      maxZ,
    ],
    center: [0, 0, 0],
    size: [bounds.size[0], bounds.size[1], maxZ - minZ],
  };
}

function localizePositions(
  positions: Float32Array,
  origin: MeshBounds['center'],
  verticalExaggeration: number,
): Float32Array {
  const localized = new Float32Array(positions.length);

  for (let index = 0; index < positions.length; index += 3) {
    localized[index] = positions[index] - origin[0];
    localized[index + 1] = positions[index + 1] - origin[1];
    localized[index + 2] = (positions[index + 2] - origin[2]) * verticalExaggeration;
  }

  return localized;
}

function localizeProfilePoint(
  point: [number, number],
  origin: MeshBounds['center'],
): [number, number] {
  return [point[0] - origin[0], point[1] - origin[1]];
}

function worldPointFromRenderPoint(
  point: [number, number],
  origin: MeshBounds['center'],
): [number, number] {
  return [point[0] + origin[0], point[1] + origin[1]];
}

function makeRenderClipPlane(
  clip: ClipState,
  origin: MeshBounds['center'],
  verticalExaggeration: number,
): Plane | null {
  if (!clip.enabled) {
    return null;
  }

  const localizedValue =
    clip.axis === 'x'
      ? clip.value - origin[0]
      : clip.axis === 'y'
        ? clip.value - origin[1]
        : (clip.value - origin[2]) * verticalExaggeration;

  return makeClipPlane({ ...clip, value: localizedValue });
}

function worldElevationFromRenderZ(
  renderZ: number,
  originZ: number,
  verticalExaggeration: number,
): number {
  return originZ + renderZ / Math.max(verticalExaggeration, 1e-6);
}

function fitPerspectiveDistance(
  bounds: MeshBounds,
  camera: PerspectiveCameraImpl,
  viewportWidth: number,
  viewportHeight: number,
): number {
  const radius = Math.max(
    Math.hypot(bounds.size[0], bounds.size[1], bounds.size[2]) * 0.5,
    1,
  );
  const verticalFov = MathUtils.degToRad(camera.fov);
  const aspect = viewportWidth / Math.max(viewportHeight, 1);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov * 0.5) * aspect);
  const verticalDistance = radius / Math.sin(verticalFov * 0.5);
  const horizontalDistance = radius / Math.sin(horizontalFov * 0.5);
  return Math.max(verticalDistance, horizontalDistance) * 1.12;
}

function fitOrthographicZoom(
  bounds: MeshBounds,
  viewportWidth: number,
  viewportHeight: number,
): number {
  const paddedWidth = Math.max(bounds.size[0], 1) * 1.12;
  const paddedHeight = Math.max(bounds.size[1], 1) * 1.12;
  return Math.max(
    0.01,
    Math.min(
      viewportWidth / Math.max(paddedWidth, 1e-6),
      viewportHeight / Math.max(paddedHeight, 1e-6),
    ),
  );
}

function perspectiveDistanceRange(
  bounds: MeshBounds,
  camera: PerspectiveCameraImpl,
  viewportWidth: number,
  viewportHeight: number,
): { min: number; max: number } {
  const fittedDistance = fitPerspectiveDistance(bounds, camera, viewportWidth, viewportHeight);
  const maxDimension = Math.max(bounds.size[0], bounds.size[1], bounds.size[2], 1);

  return {
    min: Math.max(maxDimension * 0.05, fittedDistance * 0.04, 0.5),
    max: Math.max(maxDimension * 320, fittedDistance * 48, 10),
  };
}

function orthographicZoomRange(
  bounds: MeshBounds,
  viewportWidth: number,
  viewportHeight: number,
): { min: number; max: number } {
  const fittedZoom = fitOrthographicZoom(bounds, viewportWidth, viewportHeight);

  return {
    min: Math.max(fittedZoom * 0.03, 0.001),
    max: Math.max(fittedZoom * 90, 0.01),
  };
}

function applyPresetView(
  preset: ViewPreset,
  bounds: MeshBounds,
  viewportWidth: number,
  viewportHeight: number,
  controls: TrackballControlsImpl | null,
  perspectiveCamera: PerspectiveCameraImpl | null,
  orthographicCamera: OrthographicCameraImpl | null,
): void {
  const center = new Vector3(...bounds.center);
  const target = controls?.target ?? center;
  target.set(...bounds.center);

  if (preset === 'isometric') {
    if (!perspectiveCamera) {
      return;
    }

    const distance = fitPerspectiveDistance(bounds, perspectiveCamera, viewportWidth, viewportHeight);
    perspectiveCamera.position.copy(center).addScaledVector(ISO_CAMERA_OFFSET, distance);
    perspectiveCamera.up.copy(WORLD_UP);
    perspectiveCamera.zoom = 1;
    perspectiveCamera.lookAt(center);
    perspectiveCamera.updateProjectionMatrix();
    controls?.update();
    return;
  }

  if (!orthographicCamera) {
    return;
  }

  const zoom = fitOrthographicZoom(bounds, viewportWidth, viewportHeight);
  const offset = Math.max(Math.max(bounds.size[0], bounds.size[1], bounds.size[2], 1) * 2.4, 10);
  orthographicCamera.position.set(
    bounds.center[0],
    bounds.center[1],
    preset === 'top' ? bounds.center[2] + offset : bounds.center[2] - offset,
  );
  orthographicCamera.up.copy(WORLD_UP);
  orthographicCamera.zoom = zoom;
  orthographicCamera.lookAt(center);
  orthographicCamera.updateProjectionMatrix();
  controls?.update();
}

function getSceneCamera(
  cameraKind: 'perspective' | 'orthographic',
  perspectiveCamera: PerspectiveCameraImpl | null,
  orthographicCamera: OrthographicCameraImpl | null,
): SceneCamera | null {
  return cameraKind === 'perspective' ? perspectiveCamera : orthographicCamera;
}

function classifyActiveView(camera: SceneCamera, direction: Vector3): ActiveView {
  if (camera.isOrthographicCamera) {
    if (direction.dot(TOP_VIEW_DIRECTION) >= VIEW_TOLERANCE_ORTHO) {
      return 'top';
    }
    if (direction.dot(BOTTOM_VIEW_DIRECTION) >= VIEW_TOLERANCE_ORTHO) {
      return 'bottom';
    }
    return 'custom';
  }

  return direction.dot(ISO_VIEW_DIRECTION) >= VIEW_TOLERANCE_ISO ? 'isometric' : 'custom';
}

function pickNiceScaleValue(maxUnits: number): number | null {
  if (!Number.isFinite(maxUnits) || maxUnits <= 0) {
    return null;
  }

  const exponent = Math.floor(Math.log10(maxUnits));
  const base = 10 ** exponent;
  if (5 * base <= maxUnits) {
    return 5 * base;
  }
  if (2 * base <= maxUnits) {
    return 2 * base;
  }
  return base;
}

function computeScaleBarInfo(
  camera: SceneCamera,
  target: Vector3,
  viewportWidth: number,
  viewportHeight: number,
): ScaleBarInfo | null {
  if (viewportWidth <= 0 || viewportHeight <= 0) {
    return null;
  }

  let unitsPerPixel: number;

  if (camera.isOrthographicCamera) {
    unitsPerPixel =
      (camera.right - camera.left) / Math.max(camera.zoom, 1e-6) / Math.max(viewportWidth, 1);
  } else {
    const distance = camera.position.distanceTo(target);
    const verticalFov = MathUtils.degToRad(camera.fov);
    const visibleHeight = 2 * distance * Math.tan(verticalFov * 0.5);
    const visibleWidth = visibleHeight * (viewportWidth / viewportHeight);
    unitsPerPixel = visibleWidth / viewportWidth;
  }

  const scaleValue = pickNiceScaleValue(unitsPerPixel * SCALE_BAR_MAX_WIDTH);
  if (!scaleValue) {
    return null;
  }

  return {
    label: `${formatNumber(scaleValue)} units`,
    widthPx: scaleValue / unitsPerPixel,
  };
}

function resetTrackballInteraction(controls: TrackballControlsImpl | null): void {
  if (!controls) {
    return;
  }

  const internal = controls as TrackballControlsInternal;
  internal._state = internal.STATE.NONE;
  internal._keyState = internal.STATE.NONE;
  internal._movePrev.copy(internal._moveCurr);
  internal._zoomStart.copy(internal._zoomEnd);
  internal._panStart.copy(internal._panEnd);
  internal._lastAngle = 0;
  internal.domElement?.ownerDocument.removeEventListener('pointermove', internal.onPointerMove);
  internal.domElement?.ownerDocument.removeEventListener('pointerup', internal.onPointerUp);
}

function stopScenePointer(event: ThreeEvent<PointerEvent>): void {
  event.stopPropagation();
  event.nativeEvent.preventDefault();
  event.nativeEvent.stopPropagation();
  event.nativeEvent.stopImmediatePropagation?.();
}

function SceneContent({
  positions,
  bounds,
  opacity,
  verticalExaggeration,
  wireframe,
  clip,
  viewRequest,
  showTopShell,
  profileEnabled,
  profileStart,
  profileEnd,
  hoveredProfileMapPoint,
  pickMode,
  onPickPoint,
  onActiveViewChange,
  onScaleBarChange,
  onHoverElevationChange,
}: Omit<MeshSceneProps, 'activeView' | 'onRequestView'> & {
  onScaleBarChange: (value: ScaleBarInfo | null) => void;
  onHoverElevationChange: (value: HoverElevationInfo | null) => void;
}) {
  const perspectiveCameraRef = useRef<PerspectiveCameraImpl | null>(null);
  const orthographicCameraRef = useRef<OrthographicCameraImpl | null>(null);
  const controlsRef = useRef<TrackballControlsImpl | null>(null);
  const terrainMeshRef = useRef<MeshImpl<BufferGeometry> | null>(null);
  const activeViewRef = useRef<ActiveView>('top');
  const appliedViewRequestIdRef = useRef<number | null>(null);
  const scaleBarRef = useRef<string | null>(null);
  const directionRef = useRef(new Vector3());
  const raycasterRef = useRef(new Raycaster());
  const pointerRef = useRef(new Vector2());
  const dragPointRef = useRef(new Vector3());
  const wheelEyeRef = useRef(new Vector3());
  const dragPlaneRef = useRef(new Plane());
  const [cameraKind, setCameraKind] = useState<'perspective' | 'orthographic'>(
    viewRequest.preset === 'isometric' ? 'perspective' : 'orthographic',
  );
  const [cameraVersion, setCameraVersion] = useState(0);
  const [dragMode, setDragMode] = useState<'start' | 'end' | null>(null);
  const size = useThree((state) => state.size);
  const setThree = useThree((state) => state.set);
  const gl = useThree((state) => state.gl);
  const handlePerspectiveCameraRef = useCallback((camera: PerspectiveCameraImpl | null) => {
    if (perspectiveCameraRef.current === camera) {
      return;
    }
    perspectiveCameraRef.current = camera;
    if (camera) {
      setCameraVersion((version) => version + 1);
    }
  }, []);
  const handleOrthographicCameraRef = useCallback((camera: OrthographicCameraImpl | null) => {
    if (orthographicCameraRef.current === camera) {
      return;
    }
    orthographicCameraRef.current = camera;
    if (camera) {
      setCameraVersion((version) => version + 1);
    }
  }, []);

  const displayPositions = useMemo(() => {
    if (!positions || !bounds || showTopShell) {
      return positions;
    }
    return filterTopCapPositions(positions, bounds);
  }, [bounds, positions, showTopShell]);
  const renderBounds = useMemo(
    () => (bounds ? localizeBounds(bounds, verticalExaggeration) : null),
    [bounds, verticalExaggeration],
  );
  const renderPositions = useMemo(() => {
    if (!displayPositions || !bounds) {
      return null;
    }
    return localizePositions(displayPositions, bounds.center, verticalExaggeration);
  }, [bounds, displayPositions, verticalExaggeration]);

  const geometry = useMemo(() => {
    if (!displayPositions || !renderPositions || !bounds) {
      return null;
    }

    const meshGeometry = new BufferGeometry();
    meshGeometry.setAttribute('position', new BufferAttribute(renderPositions, 3));
    meshGeometry.setAttribute(
      'color',
      new BufferAttribute(buildTerrainColorArray(displayPositions, bounds), 3),
    );
    meshGeometry.computeVertexNormals();
    return meshGeometry;
  }, [bounds, displayPositions, renderPositions]);

  const clipPlane = useMemo(
    () => (bounds ? makeRenderClipPlane(clip, bounds.center, verticalExaggeration) : null),
    [bounds, clip, verticalExaggeration],
  );
  const linePoints = useMemo(() => {
    if (!renderBounds || !bounds || !profileEnabled || !profileStart || !profileEnd) {
      return null;
    }
    const [startX, startY] = localizeProfilePoint(profileStart, bounds.center);
    const [endX, endY] = localizeProfilePoint(profileEnd, bounds.center);
    const z = renderBounds.max[2] + Math.max(renderBounds.size[2] * 0.015, 0.25);
    return [
      [startX, startY, z],
      [endX, endY, z],
    ] as [readonly [number, number, number], readonly [number, number, number]];
  }, [bounds, profileEnabled, profileEnd, profileStart, renderBounds]);
  const hoveredProfileLinePoint = useMemo(() => {
    if (!linePoints || !bounds || !hoveredProfileMapPoint) {
      return null;
    }

    const [x, y] = localizeProfilePoint(hoveredProfileMapPoint, bounds.center);
    return [x, y, linePoints[0][2]] as const;
  }, [bounds, hoveredProfileMapPoint, linePoints]);
  const pickingPlaneZ = renderBounds
    ? renderBounds.max[2] + Math.max(renderBounds.size[2] * 0.02, 0.5)
    : 0;
  const pickingPlaneSizeX = renderBounds ? Math.max(renderBounds.size[0] * 1.3, 1) : 1;
  const pickingPlaneSizeY = renderBounds ? Math.max(renderBounds.size[1] * 1.3, 1) : 1;
  const sphereRadius = renderBounds
    ? Math.max(Math.max(renderBounds.size[0], renderBounds.size[1]), 1) * 0.012
    : 1;
  const beginEndpointDrag = useCallback((mode: 'start' | 'end', event: ThreeEvent<PointerEvent>) => {
    stopScenePointer(event);
    if (controlsRef.current) {
      controlsRef.current.enabled = false;
    }
    resetTrackballInteraction(controlsRef.current);
    setDragMode(mode);
  }, []);

  useEffect(() => {
    controlsRef.current?.handleResize();
  }, [cameraKind, size.height, size.width]);

  useEffect(() => {
    const controls = controlsRef.current;
    const perspectiveCamera = perspectiveCameraRef.current;
    if (!controls || !renderBounds || !perspectiveCamera) {
      return;
    }

    const { min, max } = perspectiveDistanceRange(
      renderBounds,
      perspectiveCamera,
      size.width,
      size.height,
    );
    controls.minDistance = min;
    controls.maxDistance = max;
    const target = controls.target;
    const eye = wheelEyeRef.current.subVectors(perspectiveCamera.position, target);
    const clampedDistance = MathUtils.clamp(eye.length(), min, max);
    if (Math.abs(clampedDistance - eye.length()) > 1e-6) {
      eye.setLength(clampedDistance);
      perspectiveCamera.position.copy(target).add(eye);
      perspectiveCamera.lookAt(target);
      perspectiveCamera.updateProjectionMatrix();
      controls.update();
    }
  }, [cameraVersion, renderBounds, size.height, size.width]);

  useEffect(() => {
    const orthographicCamera = orthographicCameraRef.current;
    if (!orthographicCamera || !renderBounds) {
      return;
    }

    const { min, max } = orthographicZoomRange(renderBounds, size.width, size.height);
    const clampedZoom = MathUtils.clamp(orthographicCamera.zoom, min, max);
    if (Math.abs(clampedZoom - orthographicCamera.zoom) > 1e-6) {
      orthographicCamera.zoom = clampedZoom;
      orthographicCamera.updateProjectionMatrix();
      controlsRef.current?.update();
    }
  }, [cameraVersion, renderBounds, size.height, size.width]);

  useEffect(() => {
    if (!renderBounds) {
      return;
    }

    const domElement = gl.domElement;
    const handleWheel = (event: WheelEvent) => {
      const activeCamera = getSceneCamera(
        cameraKind,
        perspectiveCameraRef.current,
        orthographicCameraRef.current,
      );
      if (!activeCamera || dragMode) {
        return;
      }

      event.preventDefault();

      const zoomScale = Math.exp(event.deltaY * 0.0015);
      const controlsTarget = controlsRef.current?.target ?? new Vector3(...renderBounds.center);

      if (activeCamera.isPerspectiveCamera) {
        const eye = wheelEyeRef.current.subVectors(activeCamera.position, controlsTarget);
        const { min, max } = perspectiveDistanceRange(
          renderBounds,
          activeCamera,
          size.width,
          size.height,
        );
        eye.setLength(MathUtils.clamp(eye.length() * zoomScale, min, max));
        activeCamera.position.copy(controlsTarget).add(eye);
      } else {
        const { min, max } = orthographicZoomRange(renderBounds, size.width, size.height);
        activeCamera.zoom = MathUtils.clamp(activeCamera.zoom / zoomScale, min, max);
      }

      activeCamera.updateProjectionMatrix();
      activeCamera.lookAt(controlsTarget);
      controlsRef.current?.update();
    };
    const handleContextMenu = (event: MouseEvent) => {
      event.preventDefault();
    };

    domElement.addEventListener('wheel', handleWheel, { passive: false });
    domElement.addEventListener('contextmenu', handleContextMenu);
    return () => {
      domElement.removeEventListener('wheel', handleWheel);
      domElement.removeEventListener('contextmenu', handleContextMenu);
    };
  }, [cameraKind, dragMode, gl, renderBounds, size.height, size.width]);

  useEffect(() => {
    if (!dragMode || !bounds || !renderBounds) {
      return;
    }

    resetTrackballInteraction(controlsRef.current);
    const domElement = gl.domElement;
    const handlePointerMove = (event: PointerEvent) => {
      const activeCamera = getSceneCamera(
        cameraKind,
        perspectiveCameraRef.current,
        orthographicCameraRef.current,
      );
      if (!activeCamera) {
        return;
      }

      const rect = domElement.getBoundingClientRect();
      pointerRef.current.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycasterRef.current.setFromCamera(pointerRef.current, activeCamera);
      dragPlaneRef.current.set(DRAG_PLANE_NORMAL, -pickingPlaneZ);

      if (!raycasterRef.current.ray.intersectPlane(dragPlaneRef.current, dragPointRef.current)) {
        return;
      }

      onPickPoint(
        dragMode,
        worldPointFromRenderPoint([dragPointRef.current.x, dragPointRef.current.y], bounds.center),
      );
    };

    const handlePointerUp = () => {
      if (controlsRef.current) {
        controlsRef.current.enabled = false;
      }
      resetTrackballInteraction(controlsRef.current);
      setDragMode(null);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [bounds, cameraKind, dragMode, gl, onPickPoint, pickingPlaneZ, renderBounds]);

  useEffect(() => {
    const domElement = gl.domElement;

    const clearHover = () => {
      onHoverElevationChange(null);
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!bounds || !terrainMeshRef.current || pickMode || dragMode) {
        clearHover();
        return;
      }

      const activeCamera = getSceneCamera(
        cameraKind,
        perspectiveCameraRef.current,
        orthographicCameraRef.current,
      );
      if (!activeCamera) {
        clearHover();
        return;
      }

      const rect = domElement.getBoundingClientRect();
      pointerRef.current.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycasterRef.current.setFromCamera(pointerRef.current, activeCamera);

      const hit = raycasterRef.current.intersectObject(terrainMeshRef.current, false)[0];
      if (!hit) {
        clearHover();
        return;
      }

      if (clipPlane && clipPlane.distanceToPoint(hit.point) < -HOVER_TERRAIN_EPSILON) {
        clearHover();
        return;
      }

      onHoverElevationChange({
        elevationLabel: formatNumber(
          worldElevationFromRenderZ(hit.point.z, bounds.center[2], verticalExaggeration),
        ),
      });
    };

    domElement.addEventListener('pointermove', handlePointerMove);
    domElement.addEventListener('pointerleave', clearHover);

    return () => {
      domElement.removeEventListener('pointermove', handlePointerMove);
      domElement.removeEventListener('pointerleave', clearHover);
    };
  }, [
    bounds,
    cameraKind,
    clipPlane,
    dragMode,
    gl,
    onHoverElevationChange,
    pickMode,
    verticalExaggeration,
  ]);

  useEffect(() => {
    if (!bounds || pickMode || dragMode) {
      onHoverElevationChange(null);
    }
  }, [bounds, dragMode, onHoverElevationChange, pickMode]);

  useLayoutEffect(() => {
    const activeCamera = getSceneCamera(
      cameraKind,
      perspectiveCameraRef.current,
      orthographicCameraRef.current,
    );
    if (!activeCamera) {
      return;
    }

    setThree({ camera: activeCamera });
  }, [cameraKind, cameraVersion, setThree]);

  useLayoutEffect(() => {
    if (!renderBounds) {
      return;
    }

    const desiredCameraKind = viewRequest.preset === 'isometric' ? 'perspective' : 'orthographic';
    if (cameraKind !== desiredCameraKind) {
      setCameraKind(desiredCameraKind);
      return;
    }

    if (appliedViewRequestIdRef.current === viewRequest.id) {
      return;
    }

    applyPresetView(
      viewRequest.preset,
      renderBounds,
      size.width,
      size.height,
      controlsRef.current,
      perspectiveCameraRef.current,
      orthographicCameraRef.current,
    );
    appliedViewRequestIdRef.current = viewRequest.id;
  }, [cameraKind, cameraVersion, renderBounds, size.height, size.width, viewRequest.id, viewRequest.preset]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  useFrame(() => {
    const activeCamera = getSceneCamera(
      cameraKind,
      perspectiveCameraRef.current,
      orthographicCameraRef.current,
    );
    if (!renderBounds || !activeCamera) {
      return;
    }

    activeCamera.getWorldDirection(directionRef.current);
    const nextActiveView = classifyActiveView(activeCamera, directionRef.current);
    if (nextActiveView !== activeViewRef.current) {
      activeViewRef.current = nextActiveView;
      onActiveViewChange(nextActiveView);
    }

    const scaleBar = computeScaleBarInfo(
      activeCamera,
      controlsRef.current?.target ?? new Vector3(...renderBounds.center),
      size.width,
      size.height,
    );
    const scaleBarKey = scaleBar ? `${scaleBar.label}:${scaleBar.widthPx.toFixed(1)}` : null;
    if (scaleBarKey !== scaleBarRef.current) {
      scaleBarRef.current = scaleBarKey;
      onScaleBarChange(scaleBar);
    }
  });

  if (!bounds || !renderBounds) {
    return null;
  }

  const controlsEnabled = !pickMode && !dragMode;
  const activeCamera = getSceneCamera(
    cameraKind,
    perspectiveCameraRef.current,
    orthographicCameraRef.current,
  );

  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight position={[1.25, -1.4, 1.35]} intensity={1.05} color="#ffffff" />
      <directionalLight position={[-0.55, 0.75, 0.5]} intensity={0.35} color="#f4f7fa" />

      <PerspectiveCamera
        ref={handlePerspectiveCameraRef}
        fov={36}
        near={0.1}
        far={1_000_000}
      />
      <OrthographicCamera
        ref={handleOrthographicCameraRef}
        near={0.1}
        far={1_000_000}
      />

      {geometry ? (
        <mesh ref={terrainMeshRef} geometry={geometry}>
          <meshPhongMaterial
            vertexColors
            side={DoubleSide}
            transparent
            opacity={opacity}
            wireframe={wireframe}
            clippingPlanes={clipPlane ? [clipPlane] : []}
            flatShading
            shininess={8}
            specular="#f4f7fa"
            toneMapped={false}
          />
        </mesh>
      ) : null}

      {linePoints ? (
        <>
          <Line points={linePoints} color={PROFILE_LINE_COLOR} lineWidth={3.5} />
          <mesh
            position={linePoints[0]}
            onPointerDown={(event) => {
              beginEndpointDrag('start', event);
            }}
          >
            <sphereGeometry args={[sphereRadius, 18, 18]} />
            <meshBasicMaterial color={PROFILE_LINE_COLOR} toneMapped={false} />
          </mesh>
          <mesh
            position={linePoints[1]}
            onPointerDown={(event) => {
              beginEndpointDrag('end', event);
            }}
          >
            <sphereGeometry args={[sphereRadius, 18, 18]} />
            <meshBasicMaterial color={PROFILE_LINE_COLOR} toneMapped={false} />
          </mesh>
          {hoveredProfileLinePoint ? (
            <mesh position={hoveredProfileLinePoint}>
              <sphereGeometry args={[sphereRadius * 0.78, 18, 18]} />
              <meshBasicMaterial color={PROFILE_HOVER_MARKER_COLOR} toneMapped={false} />
            </mesh>
          ) : null}
        </>
      ) : null}

      {pickMode || dragMode ? (
        <mesh
          position={[renderBounds.center[0], renderBounds.center[1], pickingPlaneZ]}
          onPointerDown={(event) => {
            if (!pickMode) {
              return;
            }
            event.stopPropagation();
            onPickPoint(
              pickMode,
              worldPointFromRenderPoint([event.point.x, event.point.y], bounds.center),
            );
          }}
        >
          <planeGeometry args={[pickingPlaneSizeX, pickingPlaneSizeY]} />
          <meshBasicMaterial
            transparent
            opacity={0.08}
            color="#ffffff"
            side={DoubleSide}
            toneMapped={false}
          />
        </mesh>
      ) : null}

      <TrackballControls
        key={cameraKind}
        ref={controlsRef}
        camera={activeCamera ?? undefined}
        enabled={controlsEnabled}
        rotateSpeed={2.2}
        panSpeed={0.85}
        staticMoving
        noZoom
        noPan={false}
        noRotate={false}
        // TrackballControls maps these values to physical mouse buttons.
        mouseButtons={{
          LEFT: MOUSE.PAN,
          MIDDLE: -1 as MOUSE,
          RIGHT: MOUSE.ROTATE,
        }}
      />

      <GizmoHelper alignment="bottom-left" margin={[82, 78]} renderPriority={1}>
        <GizmoViewport
          axisColors={['#bb2d22', '#2ea34a', '#2b78c2']}
          labels={['X', 'Y', 'Z']}
          labelColor="#e6edf8"
          hideNegativeAxes
          disabled
        />
      </GizmoHelper>
    </>
  );
}

export function MeshScene({
  positions,
  bounds,
  opacity,
  verticalExaggeration,
  wireframe,
  clip,
  viewRequest,
  activeView,
  showTopShell,
  profileEnabled,
  profileStart,
  profileEnd,
  hoveredProfileMapPoint,
  pickMode,
  onPickPoint,
  onRequestView,
  onActiveViewChange,
}: MeshSceneProps) {
  const [scaleBar, setScaleBar] = useState<ScaleBarInfo | null>(null);
  const [hoverElevation, setHoverElevation] = useState<HoverElevationInfo | null>(null);
  const legendGradient = useMemo(
    () =>
      `linear-gradient(180deg, ${[...TERRAIN_LEGEND_STOPS]
        .reverse()
        .map((stop) => stop.hex)
        .join(', ')})`,
    [],
  );
  const activeViewLabel = activeView === 'custom' ? 'Custom' : `${activeView[0].toUpperCase()}${activeView.slice(1)}`;

  useEffect(() => {
    if (!bounds) {
      setHoverElevation(null);
    }
  }, [bounds]);

  return (
    <div className="viewer-canvas">
      <Canvas
        onCreated={({ gl }) => {
          gl.setClearColor('#000000', 0);
          gl.localClippingEnabled = true;
        }}
      >
        <SceneContent
          positions={positions}
          bounds={bounds}
          opacity={opacity}
          verticalExaggeration={verticalExaggeration}
          wireframe={wireframe}
          clip={clip}
          viewRequest={viewRequest}
          showTopShell={showTopShell}
          profileEnabled={profileEnabled}
          profileStart={profileStart}
          profileEnd={profileEnd}
          hoveredProfileMapPoint={hoveredProfileMapPoint}
          pickMode={pickMode}
          onPickPoint={onPickPoint}
          onActiveViewChange={onActiveViewChange}
          onScaleBarChange={setScaleBar}
          onHoverElevationChange={setHoverElevation}
        />
      </Canvas>

      <div className="viewer-map-toolbar">
        <div className="viewer-map-buttons">
          <button
            type="button"
            className={activeView === 'top' ? 'active' : ''}
            onClick={() => onRequestView('top')}
            disabled={!bounds}
          >
            Top
          </button>
          <button
            type="button"
            className={activeView === 'bottom' ? 'active' : ''}
            onClick={() => onRequestView('bottom')}
            disabled={!bounds}
          >
            Bottom
          </button>
          <button
            type="button"
            className={activeView === 'isometric' ? 'active' : ''}
            onClick={() => onRequestView('isometric')}
            disabled={!bounds}
          >
            Isometric
          </button>
          <button
            type="button"
            onClick={() =>
              onRequestView(activeView === 'custom' ? 'top' : (activeView as ViewPreset))
            }
            disabled={!bounds}
          >
            Reset
          </button>
          <span className="field-tooltip-anchor viewer-map-help-anchor">
            <button type="button" className="button-secondary viewer-map-help-button" disabled={!bounds}>
              Help
            </button>
            <span role="tooltip" className="field-tooltip viewer-map-help-tooltip">
              {MAP_HELP_TOOLTIP}
            </span>
          </span>
        </div>
        <div className={`viewer-view-indicator ${activeView === 'custom' ? 'custom' : ''}`}>
          {activeViewLabel}
        </div>
      </div>

      {bounds ? (
        <div className="viewer-legend">
          <div className="viewer-legend-title">ELEV</div>
          <div className="viewer-legend-body">
            <div className="viewer-legend-scale">
              <span>{formatNumber(bounds.max[2])}</span>
              <div className="viewer-legend-gradient" style={{ backgroundImage: legendGradient }} />
              <span>{formatNumber(bounds.min[2])}</span>
            </div>
          </div>
        </div>
      ) : null}

      {bounds ? (
        <div className={`viewer-hover-status ${hoverElevation ? 'active' : ''}`}>
          {hoverElevation ? `Elevation: ${hoverElevation.elevationLabel}` : 'Hover terrain to see elevation.'}
        </div>
      ) : null}

      {scaleBar ? (
        <div className="viewer-scale-bar">
          <div className="viewer-scale-bar-line" style={{ width: `${scaleBar.widthPx}px` }} />
          <div className="viewer-scale-bar-label">{scaleBar.label}</div>
        </div>
      ) : null}
    </div>
  );
}
