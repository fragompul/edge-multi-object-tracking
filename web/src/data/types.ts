export interface TrackerMeta {
  id: string;
  label: string;
  color: string;
  blurb: string;
  cmc: boolean;
  reid: boolean;
  embedder: string | null;
}

export interface SequenceMeta {
  id: string;
  camera: "static" | "moving";
  title: string;
  video: string;
  poster: string;
  tracks: string;
  width: number;
  height: number;
  fps: number;
  frames: number;
  sourceFps: number;
  startFrame: number;
  meanPeople: number;
  gtTracks: number;
  cameraMotionPx: number;
  sourceSize: [number, number];
}

export interface Manifest {
  backbone: string;
  clipSeconds: number;
  sequences: SequenceMeta[];
  trackers: TrackerMeta[];
  license: string;
}

/** `[id, x1, y1, x2, y2]` in the clip's pixel space. */
export type BoxRow = [number, number, number, number, number];
export type DetRow = [number, number, number, number, number];

export interface SwitchEvent {
  f: number;
  gt: number;
  from: number;
  to: number;
  box: [number, number, number, number] | null;
}

export interface ClipMetrics {
  hota: number;
  idf1: number;
  mota: number;
  assa: number;
  deta: number;
  num_switches: number;
}

export interface ClipTracks {
  trackers: Record<string, BoxRow[][]>;
  switches: Record<string, SwitchEvent[]>;
  metrics: Record<string, ClipMetrics>;
  detections: DetRow[][];
  gt: BoxRow[][];
}

export interface TrackingRow {
  sequence: string;
  backbone: string;
  tracker: string;
  imgsz: number;
  mota: number;
  motp: number;
  idf1: number;
  hota: number;
  deta: number;
  assa: number;
  loca: number;
  num_switches: number;
  num_false_positives: number;
  num_misses: number;
  mostly_tracked: number;
  mostly_lost: number;
  num_frames: number;
  tracker_ms_per_frame: number;
}

export interface DetectionRow {
  sequence: string;
  backbone: string;
  ap50: number;
  ap: number;
  recall50: number;
  num_gt: number;
  num_detections: number;
}

export interface SequenceRow {
  sequence: string;
  width: number;
  height: number;
  fps: number;
  frames: number;
  gt_tracks: number;
  gt_boxes: number;
  mean_people_per_frame: number;
  camera_motion_px: number;
  camera: "static" | "moving";
}

export interface EdgeRow {
  id: string;
  group: "main" | "threads" | "resolution";
  backbone: string;
  variant: string;
  size_mb: number;
  mean_ms: number;
  median_ms: number;
  p95_ms: number;
  fps: number;
  peak_rss_mb: number;
  ap50: number;
  ap: number;
  recall50: number;
  detections_per_frame: number;
  threads?: number;
  long_side?: number;
}

export interface EdgeDoc {
  machine: Record<string, unknown>;
  protocol: Record<string, unknown>;
  rows: EdgeRow[];
  profile?: Record<string, { op: string; ms: number; share: number }[]>;
}

export interface VariantInfo {
  label: string;
  family: string;
  cmc: boolean;
  reid: boolean;
  embedder: string | null;
  cascade: boolean;
  color: string;
}

export interface SweepRow {
  backbone: string;
  imgsz: number;
  tracker: string;
  hota: number;
  idf1: number;
  mota: number;
  assa: number;
  num_switches: number;
  ap50: number;
  ap: number;
}

export interface Results {
  tracking: TrackingRow[];
  resolutionSweep: SweepRow[];
  detection: DetectionRow[];
  sequences: SequenceRow[];
  edge: EdgeDoc;
  variants: Record<string, VariantInfo>;
  tuning: Record<string, unknown>;
}
