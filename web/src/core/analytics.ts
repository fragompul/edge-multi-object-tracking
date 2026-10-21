import type { FrameTracks, XYXY } from "./types";

/**
 * Scene analytics over tracks: a TypeScript port of `emot/analytics/{trajectories,events}.py`.
 * All geometry uses the **foot point** (bottom-centre of the box); speeds are in
 * body-heights/second so they are comparable across perspective depth.
 */
export type Point = [number, number];

export interface Zone {
  name: string;
  polygon: Point[];
}

export interface Line {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TrackHistory {
  frames: number[];
  centers: Point[];
  feet: Point[];
  heights: number[];
}

export interface AnalyticsEvent {
  kind: "crossing" | "loitering" | "crowding";
  frame: number;
  seconds: number;
  trackId?: number;
  zone?: string;
  details: Record<string, number | string>;
}

export const footPoint = (b: XYXY): Point => [(b[0] + b[2]) / 2, b[3]];

export function pointInPolygon([x, y]: Point, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    if (y1 > y !== y2 > y && x < ((x2 - x1) * (y - y1)) / (y2 - y1 + 1e-12) + x1) inside = !inside;
  }
  return inside;
}

export function buildTrajectories(frames: FrameTracks): Map<number, TrackHistory> {
  const out = new Map<number, TrackHistory>();
  for (const f of [...frames.keys()].sort((a, b) => a - b)) {
    for (const t of frames.get(f)!) {
      let h = out.get(t.id);
      if (!h) out.set(t.id, (h = { frames: [], centers: [], feet: [], heights: [] }));
      h.frames.push(f);
      h.centers.push([(t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2]);
      h.feet.push(footPoint(t.box));
      h.heights.push(t.box[3] - t.box[1]);
    }
  }
  return out;
}

const side = (l: Line, p: Point): number => (l.x2 - l.x1) * (p[1] - l.y1) - (l.y2 - l.y1) * (p[0] - l.x1);

/** Side A: cross((p2-p1), (p-p1)) < 0; side B: > 0 (screen coordinates, y down). */
export function crossingEvents(trajs: Map<number, TrackHistory>, line: Line, fps: number): AnalyticsEvent[] {
  const events: AnalyticsEvent[] = [];
  for (const [id, h] of trajs) {
    for (let i = 1; i < h.feet.length; i++) {
      const a = side(line, h.feet[i - 1]);
      const b = side(line, h.feet[i]);
      if (a === 0 || b === 0 || a > 0 === b > 0) continue;
      const frame = h.frames[i];
      events.push({
        kind: "crossing",
        frame,
        seconds: frame / fps,
        trackId: id,
        zone: "line",
        details: { direction: a < 0 ? "a_to_b" : "b_to_a" },
      });
    }
  }
  return events.sort((x, y) => x.frame - y.frame);
}

export interface DwellStat {
  seconds: number;
  entries: number;
  firstFrame: number;
}

export function dwellStats(trajs: Map<number, TrackHistory>, zone: Zone, fps: number): Map<number, DwellStat> {
  const out = new Map<number, DwellStat>();
  for (const [id, h] of trajs) {
    const inside = h.feet.map((p) => pointInPolygon(p, zone.polygon));
    const n = inside.filter(Boolean).length;
    if (!n) continue;
    let entries = inside[0] ? 1 : 0;
    for (let i = 1; i < inside.length; i++) if (inside[i] && !inside[i - 1]) entries++;
    out.set(id, { seconds: n / fps, entries, firstFrame: h.frames[inside.indexOf(true)] });
  }
  return out;
}

export function occupancySeries(frames: FrameTracks, zone: Zone | null): Map<number, number> {
  const out = new Map<number, number>();
  for (const f of [...frames.keys()].sort((a, b) => a - b)) {
    out.set(
      f,
      frames.get(f)!.filter((t) => !zone || pointInPolygon(footPoint(t.box), zone.polygon)).length,
    );
  }
  return out;
}

export function detectCrowding(
  frames: FrameTracks,
  zone: Zone | null,
  fps: number,
  threshold: number,
  minSeconds = 1,
): AnalyticsEvent[] {
  const series = occupancySeries(frames, zone);
  const need = Math.max(1, Math.round(minSeconds * fps));
  const events: AnalyticsEvent[] = [];
  let run: [number, number][] = [];
  const close = () => {
    if (run.length >= need)
      events.push({
        kind: "crowding",
        frame: run[0][0],
        seconds: run[0][0] / fps,
        zone: zone?.name,
        details: { end_frame: run[run.length - 1][0], peak: Math.max(...run.map((r) => r[1])), threshold },
      });
    run = [];
  };
  let prev: number | null = null;
  for (const [f, n] of series) {
    if (n >= threshold && (prev === null || run.length === 0 || f === prev + 1)) run.push([f, n]);
    else {
      close();
      if (n >= threshold) run.push([f, n]);
    }
    prev = f;
  }
  close();
  return events;
}

export function detectLoitering(trajs: Map<number, TrackHistory>, zone: Zone, fps: number, minSeconds: number): AnalyticsEvent[] {
  const need = Math.ceil(minSeconds * fps);
  const events: AnalyticsEvent[] = [];
  for (const [id, h] of trajs) {
    let count = 0;
    for (let i = 0; i < h.feet.length; i++) {
      if (pointInPolygon(h.feet[i], zone.polygon) && ++count === need) {
        events.push({ kind: "loitering", frame: h.frames[i], seconds: h.frames[i] / fps, trackId: id, zone: zone.name, details: { min_seconds: minSeconds } });
        break;
      }
    }
  }
  return events.sort((a, b) => a.frame - b.frame);
}

export const median = (v: number[]): number => {
  if (!v.length) return 0;
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export const mean = (v: number[]): number => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0);

export function motionStats(h: TrackHistory, fps: number, minFrames = 10) {
  if (h.feet.length < minFrames) return null;
  const speeds: number[] = [];
  let path = 0;
  for (let i = 1; i < h.feet.length; i++) {
    const dt = (h.frames[i] - h.frames[i - 1]) / fps;
    if (dt <= 0) continue;
    const step = Math.hypot(h.feet[i][0] - h.feet[i - 1][0], h.feet[i][1] - h.feet[i - 1][1]);
    path += step;
    speeds.push(step / Math.max(h.heights[i], 1) / dt);
  }
  return {
    medianSpeedBhPerS: median(speeds),
    pathPx: path,
    durationS: (h.frames[h.frames.length - 1] - h.frames[0]) / fps,
  };
}

export function sceneSummary(frames: FrameTracks, trajs: Map<number, TrackHistory>, fps: number) {
  const series = occupancySeries(frames, null);
  let peakFrame = 0;
  let peak = -1;
  for (const [f, n] of series) if (n > peak) [peak, peakFrame] = [n, f];
  const durations = [...trajs.values()].map((h) => (h.frames[h.frames.length - 1] - h.frames[0] + 1) / fps);
  return {
    frames: frames.size,
    duration_s: frames.size / fps,
    unique_tracks: trajs.size,
    mean_people_per_frame: mean([...series.values()]),
    peak_people: Math.max(peak, 0),
    peak_frame: peakFrame,
    peak_time_s: peakFrame / fps,
    median_track_duration_s: median(durations),
  };
}

/** Normalised occupancy of foot points over a `bins x bins` grid, scaled to [0, 1]. */
export function occupancyHeatmap(trajs: Map<number, TrackHistory>, w: number, h: number, bins = 48): number[][] {
  const grid = Array.from({ length: bins }, () => new Array<number>(bins).fill(0));
  let max = 0;
  for (const t of trajs.values())
    for (const [x, y] of t.feet) {
      const gx = Math.min(bins - 1, Math.max(0, Math.floor((x / w) * bins)));
      const gy = Math.min(bins - 1, Math.max(0, Math.floor((y / h) * bins)));
      max = Math.max(max, ++grid[gy][gx]);
    }
  return max ? grid.map((r) => r.map((v) => v / max)) : grid;
}
