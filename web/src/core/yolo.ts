import type { Detection, XYXY } from "./types";

/**
 * Pre/post-processing for Ultralytics YOLO ONNX exports (`(1, 4 + classes, anchors)` output),
 * mirroring `emot/detection/onnx_detector.py` -- which was itself verified to reproduce the
 * PyTorch detector's boxes to IoU 0.99999.
 */
export interface Letterbox {
  scale: number;
  padX: number;
  padY: number;
  newW: number;
  newH: number;
}

/** Python's `round()` (half to even), so letterbox geometry matches the Python reference. */
export function pyRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function letterboxGeometry(w: number, h: number, targetH: number, targetW: number): Letterbox {
  const scale = Math.min(targetH / h, targetW / w);
  const newW = pyRound(w * scale);
  const newH = pyRound(h * scale);
  const padX = pyRound((targetW - newW) / 2 - 0.1);
  const padY = pyRound((targetH - newH) / 2 - 0.1);
  return { scale, padX, padY, newW, newH };
}

export function decodePersonBoxes(
  raw: Float32Array,
  anchors: number,
  confThreshold: number,
  personClass = 0,
): { boxes: XYXY[]; scores: number[] } {
  const boxes: XYXY[] = [];
  const scores: number[] = [];
  const clsRow = (4 + personClass) * anchors;
  for (let i = 0; i < anchors; i++) {
    const s = raw[clsRow + i];
    if (s < confThreshold) continue;
    const cx = raw[i];
    const cy = raw[anchors + i];
    const w = raw[2 * anchors + i];
    const h = raw[3 * anchors + i];
    boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
    scores.push(s);
  }
  return { boxes, scores };
}

function iouXyxy(a: XYXY, b: XYXY): number {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** Greedy NMS; returns kept indices by descending score. */
export function nms(boxes: XYXY[], scores: number[], iouThreshold: number, maxDet = 300): number[] {
  const order = scores.map((_, i) => i).sort((i, j) => scores[j] - scores[i]);
  const keep: number[] = [];
  for (const i of order) {
    if (keep.length >= maxDet) break;
    if (keep.every((k) => iouXyxy(boxes[i], boxes[k]) <= iouThreshold)) keep.push(i);
  }
  return keep;
}

/** Full post-process: decode -> NMS -> undo the letterbox -> clip to the frame. */
export function postprocess(
  raw: Float32Array,
  anchors: number,
  lb: Letterbox,
  frameW: number,
  frameH: number,
  conf = 0.1,
  iouThr = 0.7,
): Detection[] {
  const { boxes, scores } = decodePersonBoxes(raw, anchors, conf);
  return nms(boxes, scores, iouThr).map((i) => {
    const [x1, y1, x2, y2] = boxes[i];
    const clip = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
    return {
      box: [
        clip((x1 - lb.padX) / lb.scale, frameW),
        clip((y1 - lb.padY) / lb.scale, frameH),
        clip((x2 - lb.padX) / lb.scale, frameW),
        clip((y2 - lb.padY) / lb.scale, frameH),
      ] as XYXY,
      conf: scores[i],
    };
  });
}
