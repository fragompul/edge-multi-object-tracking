import {
  type AnalyticsEvent,
  type Line,
  type Point,
  type TrackHistory,
  type Zone,
  buildTrajectories,
  crossingEvents,
  detectCrowding,
  dwellStats,
  footPoint,
  mean,
  motionStats,
  occupancySeries,
  pointInPolygon,
  sceneSummary,
} from "./analytics";
import type { FrameTracks } from "./types";

/** Normalised (0-1) region presets, identical to `emot/agent/tools.py`. */
export const REGION_PRESETS: Record<string, Point[]> = {
  full: [[0, 0], [1, 0], [1, 1], [0, 1]],
  left: [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],
  right: [[0.5, 0], [1, 0], [1, 1], [0.5, 1]],
  top: [[0, 0], [1, 0], [1, 0.5], [0, 0.5]],
  bottom: [[0, 0.5], [1, 0.5], [1, 1], [0, 1]],
  center: [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
};

export type Region = string | Point[] | null | undefined;
type Json = Record<string, unknown>;

const round = (x: number, d = 2) => Math.round(x * 10 ** d) / 10 ** d;

/** The same deterministic tools the Python agent calls (`TrackStore`), in the browser. */
export class TrackStore {
  readonly trajectories: Map<number, TrackHistory>;
  constructor(
    readonly frames: FrameTracks,
    readonly fps: number,
    readonly width: number,
    readonly height: number,
  ) {
    this.trajectories = buildTrajectories(frames);
  }

  get durationS(): number {
    return this.frames.size ? Math.max(...this.frames.keys()) / this.fps : 0;
  }

  zone(region: Region): Zone | null {
    if (region === null || region === undefined) return null;
    const pts = typeof region === "string" ? REGION_PRESETS[region] : region;
    if (!pts) throw new Error(`unknown region '${region}'; use one of ${Object.keys(REGION_PRESETS).join(", ")}`);
    if (pts.length < 3) throw new Error("a polygon needs at least 3 points");
    return {
      name: typeof region === "string" ? region : "custom",
      polygon: pts.map(([x, y]) => [x * this.width, y * this.height] as Point),
    };
  }

  private window(startS?: number | null, endS?: number | null): FrameTracks {
    const lo = startS == null ? 1 : Math.floor(startS * this.fps) + 1;
    const hi = endS == null ? Math.max(0, ...this.frames.keys()) : Math.floor(endS * this.fps);
    return new Map([...this.frames].filter(([f]) => f >= lo && f <= hi));
  }

  sceneSummary(): Json {
    const s = sceneSummary(this.frames, this.trajectories, this.fps);
    return {
      ...s,
      duration_s: round(s.duration_s),
      mean_people_per_frame: round(s.mean_people_per_frame),
      peak_time_s: round(s.peak_time_s),
      median_track_duration_s: round(s.median_track_duration_s),
    };
  }

  countPeople(a: { start_s?: number | null; end_s?: number | null; region?: Region } = {}): Json {
    const win = this.window(a.start_s, a.end_s);
    const zone = this.zone(a.region);
    const ids = new Set<number>();
    for (const tracks of win.values())
      for (const t of tracks) if (!zone || pointInPolygon(footPoint(t.box), zone.polygon)) ids.add(t.id);
    const series = [...occupancySeries(win, zone).values()];
    return {
      unique_people: ids.size,
      max_simultaneous: series.length ? Math.max(...series) : 0,
      mean_simultaneous: round(mean(series)),
      window_s: [a.start_s ?? 0, a.end_s ?? round(this.durationS)],
      region: a.region ?? "full frame",
    };
  }

  lineCrossings(a: { x1: number; y1: number; x2: number; y2: number; start_s?: number | null; end_s?: number | null }): Json {
    const line: Line = { x1: a.x1 * this.width, y1: a.y1 * this.height, x2: a.x2 * this.width, y2: a.y2 * this.height };
    const lo = a.start_s ?? 0;
    const hi = a.end_s ?? Infinity;
    const events = crossingEvents(this.trajectories, line, this.fps).filter((e) => e.seconds >= lo && e.seconds <= hi);
    const aToB = events.filter((e) => e.details.direction === "a_to_b").length;
    return {
      a_to_b: aToB,
      b_to_a: events.length - aToB,
      total: events.length,
      note: "A is the side where cross(p2-p1, p-p1) < 0 (screen coordinates, y down).",
      first_events: events.slice(0, 5).map((e) => ({ t_s: round(e.seconds), track_id: e.trackId, direction: e.details.direction })),
    };
  }

  dwellTime(a: { region: Region; top_k?: number }): Json {
    const zone = this.zone(a.region);
    if (!zone) throw new Error("dwell_time needs a region");
    const stats = dwellStats(this.trajectories, zone, this.fps);
    const ranked = [...stats].sort((x, y) => y[1].seconds - x[1].seconds);
    const secs = [...stats.values()].map((s) => s.seconds);
    return {
      people_who_entered: stats.size,
      mean_dwell_s: round(mean(secs)),
      max_dwell_s: round(secs.length ? Math.max(...secs) : 0),
      longest: ranked.slice(0, a.top_k ?? 5).map(([id, s]) => ({ track_id: id, seconds: round(s.seconds), entries: s.entries })),
    };
  }

  crowding(a: { threshold: number; region?: Region; min_seconds?: number }): Json {
    const events: AnalyticsEvent[] = detectCrowding(this.frames, this.zone(a.region), this.fps, a.threshold, a.min_seconds ?? 1);
    return {
      episodes: events.map((e) => ({
        start_s: round(e.seconds),
        end_s: round(Number(e.details.end_frame) / this.fps),
        peak: e.details.peak,
      })),
      count: events.length,
    };
  }

  busiestMoments(a: { top_k?: number; region?: Region } = {}): Json {
    const series = occupancySeries(this.frames, this.zone(a.region));
    const perSecond = new Map<number, number[]>();
    for (const [f, n] of series) {
      const s = Math.floor((f - 1) / this.fps);
      perSecond.set(s, [...(perSecond.get(s) ?? []), n]);
    }
    const ranked = [...perSecond].sort((x, y) => mean(y[1]) - mean(x[1])).slice(0, a.top_k ?? 3);
    return { moments: ranked.map(([second, v]) => ({ second, mean_people: round(mean(v), 1), peak: Math.max(...v) })) };
  }

  trackInfo(a: { track_id: number }): Json {
    const h = this.trajectories.get(Math.trunc(a.track_id));
    if (!h) return { error: `no track with id ${a.track_id}` };
    const m = motionStats(h, this.fps, 2);
    const first = h.feet[0];
    const last = h.feet[h.feet.length - 1];
    return {
      track_id: Math.trunc(a.track_id),
      first_seen_s: round((h.frames[0] - 1) / this.fps),
      last_seen_s: round((h.frames[h.frames.length - 1] - 1) / this.fps),
      frames_visible: h.frames.length,
      start_xy_norm: [round(first[0] / this.width, 3), round(first[1] / this.height, 3)],
      end_xy_norm: [round(last[0] / this.width, 3), round(last[1] / this.height, 3)],
      ...(m ? { median_speed_bh_per_s: round(m.medianSpeedBhPerS), path_px: round(m.pathPx), duration_s: round(m.durationS) } : {}),
    };
  }

  call(name: string, args: Record<string, unknown>): Json {
    try {
      switch (name) {
        case "scene_summary": return this.sceneSummary();
        case "count_people": return this.countPeople(args as never);
        case "line_crossings": return this.lineCrossings(args as never);
        case "dwell_time": return this.dwellTime(args as never);
        case "crowding": return this.crowding(args as never);
        case "busiest_moments": return this.busiestMoments(args as never);
        case "track_info": return this.trackInfo(args as never);
        default: return { error: `unknown tool '${name}'` };
      }
    } catch (e) {
      return { error: (e as Error).message };
    }
  }
}
