import { linearSumAssignment } from "./hungarian";
import { iouMatrix } from "./iou";
import { KalmanBox } from "./kalman";
import type { Detection, TrackedBox, XYXY } from "./types";

/** Port of `emot/tracking/bytetrack.py` (Zhang et al., ECCV 2022). Defaults = Python defaults. */
export interface ByteTrackConfig {
  highConfThreshold: number;
  lowConfThreshold: number;
  matchIouThreshold: number; // max IoU *distance* in stage 1 (1 - IoU)
  lowMatchIouThreshold: number;
  minHits: number;
  maxAge: number;
}

export const DEFAULT_BYTETRACK: ByteTrackConfig = {
  highConfThreshold: 0.6,
  lowConfThreshold: 0.1,
  matchIouThreshold: 0.8,
  lowMatchIouThreshold: 0.5,
  minHits: 3,
  maxAge: 30,
};

type State = "tentative" | "confirmed" | "lost";

class Track {
  readonly kalman: KalmanBox;
  state: State = "tentative";
  hits = 1;
  timeSinceUpdate = 0;
  confidence: number;
  constructor(readonly id: number, box: XYXY, conf: number) {
    this.kalman = new KalmanBox(box);
    this.confidence = conf;
  }
  get xyxy(): XYXY {
    return this.kalman.xyxy;
  }
  predict(): void {
    this.kalman.predict();
    this.timeSinceUpdate += 1;
  }
  update(box: XYXY, conf: number): void {
    this.kalman.update(box);
    this.confidence = conf;
    this.hits += 1;
    this.timeSinceUpdate = 0;
  }
  markMissed(): void {
    if (this.state === "tentative") this.state = "lost";
    else if (this.timeSinceUpdate > 0) this.state = "lost";
  }
  confirmIfReady(minHits: number): void {
    const fromTentative = this.state === "tentative" && this.hits >= minHits;
    const fromLost = this.state === "lost" && this.timeSinceUpdate === 0;
    if (fromTentative || fromLost) this.state = "confirmed";
  }
}

function match(
  tracks: Track[],
  dets: Detection[],
  maxDistance: number,
): { matches: [number, number][]; unmatchedTracks: number[]; unmatchedDets: number[] } {
  if (!tracks.length || !dets.length)
    return {
      matches: [],
      unmatchedTracks: tracks.map((_, i) => i),
      unmatchedDets: dets.map((_, i) => i),
    };
  const ious = iouMatrix(tracks.map((t) => t.xyxy), dets.map((d) => d.box));
  const cost = ious.map((row) => row.map((v) => 1 - v));
  const matches: [number, number][] = [];
  const usedT = new Set<number>();
  const usedD = new Set<number>();
  for (const [r, c] of linearSumAssignment(cost)) {
    if (ious[r][c] >= 1 - maxDistance) {
      matches.push([r, c]);
      usedT.add(r);
      usedD.add(c);
    }
  }
  return {
    matches,
    unmatchedTracks: tracks.map((_, i) => i).filter((i) => !usedT.has(i)),
    unmatchedDets: dets.map((_, i) => i).filter((i) => !usedD.has(i)),
  };
}

export class ByteTracker {
  private tracks: Track[] = [];
  private nextId = 1;
  constructor(readonly config: ByteTrackConfig = DEFAULT_BYTETRACK) {}

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
  }

  update(detections: Detection[]): TrackedBox[] {
    const cfg = this.config;
    for (const t of this.tracks) t.predict();

    const high = detections.filter((d) => d.conf >= cfg.highConfThreshold);
    const low = detections.filter(
      (d) => d.conf >= cfg.lowConfThreshold && d.conf < cfg.highConfThreshold,
    );

    const s1 = match(this.tracks, high, cfg.matchIouThreshold);
    for (const [ti, di] of s1.matches) this.tracks[ti].update(high[di].box, high[di].conf);

    const remaining = s1.unmatchedTracks.map((i) => this.tracks[i]);
    const s2 = match(remaining, low, cfg.lowMatchIouThreshold);
    for (const [ti, di] of s2.matches) remaining[ti].update(low[di].box, low[di].conf);
    for (const i of s2.unmatchedTracks) remaining[i].markMissed();

    for (const di of s1.unmatchedDets)
      this.tracks.push(new Track(this.nextId++, high[di].box, high[di].conf));

    for (const t of this.tracks) t.confirmIfReady(cfg.minHits);
    this.tracks = this.tracks.filter((t) => t.timeSinceUpdate <= cfg.maxAge);

    return this.tracks
      .filter((t) => t.state === "confirmed")
      .map((t) => ({ id: t.id, box: t.xyxy, conf: t.confidence }));
  }
}
