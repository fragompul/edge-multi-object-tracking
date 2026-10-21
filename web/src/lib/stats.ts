import type { EdgeRow, Results, TrackingRow } from "../data/types";
import { groupBy, mean } from "./format";

export type Camera = "all" | "static" | "moving";
export const METRICS = ["hota", "idf1", "mota", "assa", "deta", "num_switches"] as const;
export type Metric = (typeof METRICS)[number];

export function cameraOf(results: Results): Map<string, "static" | "moving"> {
  return new Map(results.sequences.map((s) => [s.sequence, s.camera]));
}

export function filterRows(
  results: Results,
  o: { camera?: Camera; backbone?: string; sequences?: string[] },
): TrackingRow[] {
  const cams = cameraOf(results);
  return results.tracking.filter(
    (r) =>
      (!o.camera || o.camera === "all" || cams.get(r.sequence) === o.camera) &&
      (!o.backbone || o.backbone === "all" || r.backbone === o.backbone) &&
      (!o.sequences || o.sequences.includes(r.sequence)),
  );
}

export interface TrackerMean {
  tracker: string;
  n: number;
  hota: number;
  idf1: number;
  mota: number;
  assa: number;
  deta: number;
  num_switches: number;
  tracker_ms_per_frame: number;
}

export function meansByTracker(rows: TrackingRow[]): TrackerMean[] {
  return [...groupBy(rows, (r) => r.tracker)].map(([tracker, rs]) => ({
    tracker,
    n: rs.length,
    hota: mean(rs.map((r) => r.hota)),
    idf1: mean(rs.map((r) => r.idf1)),
    mota: mean(rs.map((r) => r.mota)),
    assa: mean(rs.map((r) => r.assa)),
    deta: mean(rs.map((r) => r.deta)),
    num_switches: mean(rs.map((r) => r.num_switches)),
    tracker_ms_per_frame: mean(rs.map((r) => r.tracker_ms_per_frame)),
  }));
}

/** Mean per sequence (over backbones) of one metric for one tracker. */
export function perSequence(rows: TrackingRow[], tracker: string, metric: Metric): Map<string, number> {
  const rs = rows.filter((r) => r.tracker === tracker);
  return new Map([...groupBy(rs, (r) => r.sequence)].map(([s, v]) => [s, mean(v.map((r) => r[metric]))]));
}

export const edgeRows = (results: Results, group: EdgeRow["group"]): EdgeRow[] => results.edge.rows.filter((r) => r.group === group);

export function edgeFind(rows: EdgeRow[], backbone: string, variant: string): EdgeRow | undefined {
  return rows.find((r) => r.backbone === backbone && r.variant === variant);
}
