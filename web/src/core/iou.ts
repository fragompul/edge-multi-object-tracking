import type { XYXY } from "./types";

export function iou(a: XYXY, b: XYXY): number {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 1e-9 ? inter / union : 0;
}

export function iouMatrix(a: XYXY[], b: XYXY[]): number[][] {
  return a.map((x) => b.map((y) => iou(x, y)));
}
