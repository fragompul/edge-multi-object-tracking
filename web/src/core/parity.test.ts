import { describe, expect, it } from "vitest";
import analytics from "./__fixtures__/analytics.json";
import bytetrack from "./__fixtures__/bytetrack.json";
import yolo from "./__fixtures__/yolo.json";
import { ByteTracker, type ByteTrackConfig } from "./bytetrack";
import { linearSumAssignment } from "./hungarian";
import { askOffline } from "./planner";
import { TrackStore } from "./store";
import type { FrameTracks, XYXY } from "./types";
import { decodePersonBoxes, letterboxGeometry, nms } from "./yolo";

/** These tests run the TypeScript ports on inputs recorded from the Python implementation. */
describe("ByteTrack parity with Python", () => {
  const c = bytetrack.config;
  const cfg: ByteTrackConfig = {
    highConfThreshold: c.high_conf_threshold,
    lowConfThreshold: c.low_conf_threshold,
    matchIouThreshold: c.match_iou_threshold,
    lowMatchIouThreshold: c.low_match_iou_threshold,
    minHits: c.min_hits,
    maxAge: c.max_age,
  };

  it("emits the same tracks, ids and boxes on 150 real frames", () => {
    const tracker = new ByteTracker(cfg);
    let compared = 0;
    bytetrack.detections.forEach((frame, i) => {
      const out = tracker.update(frame.map((d) => ({ box: d.slice(0, 4) as XYXY, conf: d[4] })));
      const expected = bytetrack.tracks[i];
      expect(out.length).toBe(expected.length);
      const byId = new Map(out.map((t) => [t.id, t]));
      for (const e of expected) {
        const got = byId.get(e[0]);
        expect(got, `frame ${i + 1} track ${e[0]}`).toBeDefined();
        e.slice(1, 5).forEach((v, k) => expect(got!.box[k]).toBeCloseTo(v, 3));
        compared++;
      }
    });
    expect(compared).toBeGreaterThan(500);
  });
});

describe("Hungarian assignment", () => {
  it("finds the optimal rectangular assignment", () => {
    const cost = [
      [4, 1, 3],
      [2, 0, 5],
      [3, 2, 2],
      [9, 9, 1],
    ];
    const pairs = linearSumAssignment(cost);
    const total = pairs.reduce((s, [r, c]) => s + cost[r][c], 0);
    expect(pairs.length).toBe(3);
    expect(total).toBe(Math.min(...[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]].flatMap((rows) =>
      permutations([0, 1, 2]).map((p) => rows.reduce((s, r, k) => s + cost[r][p[k]], 0)))));
  });
});

function permutations(a: number[]): number[][] {
  if (a.length <= 1) return [a];
  return a.flatMap((x, i) => permutations([...a.slice(0, i), ...a.slice(i + 1)]).map((p) => [x, ...p]));
}

describe("YOLO post-processing parity with Python", () => {
  it("decodes and NMS-filters the same anchors", () => {
    const raw = Float32Array.from(yolo.raw);
    const { boxes, scores } = decodePersonBoxes(raw, yolo.anchors, 0.1);
    expect(boxes.length).toBe(yolo.n_decoded);
    expect(nms(boxes, scores, 0.7).sort((a, b) => a - b)).toEqual([...yolo.kept].sort((a, b) => a - b));
  });

  it("computes the same letterbox geometry", () => {
    for (const g of yolo.geometries) {
      const lb = letterboxGeometry(g.w, g.h, g.th, g.tw);
      expect(lb.scale).toBeCloseTo(g.scale, 9);
      expect(lb.padX).toBeCloseTo(g.pad_x, 9);
      expect(lb.padY).toBeCloseTo(g.pad_y, 9);
    }
  });
});

describe("Analytics tools parity with Python", () => {
  const frames: FrameTracks = new Map(
    bytetrack.tracks.map((f, i) => [
      i + 1,
      f.map((t) => ({ id: t[0], box: t.slice(1, 5) as XYXY, conf: t[5] })),
    ]),
  );
  const store = new TrackStore(frames, 30, 1920, 1080);

  const close = (a: unknown, b: unknown, path: string): void => {
    if (typeof b === "number") expect(a as number, path).toBeCloseTo(b, 1);
    else if (Array.isArray(b)) {
      expect((a as unknown[]).length, path).toBe(b.length);
      b.forEach((v, i) => close((a as unknown[])[i], v, `${path}[${i}]`));
    } else if (b && typeof b === "object")
      for (const [k, v] of Object.entries(b)) close((a as Record<string, unknown>)[k], v, `${path}.${k}`);
    else expect(a, path).toEqual(b);
  };

  for (const call of analytics.calls) {
    it(`tool ${call.name}(${JSON.stringify(call.args)})`, () => {
      close(store.call(call.name, call.args as Record<string, unknown>), call.result, call.name);
    });
  }

  for (const q of analytics.questions) {
    it(`offline planner: "${q.q}"`, () => {
      const answer = askOffline(store, q.q);
      expect(answer.calls.map((c) => c.name)).toEqual(q.tools);
      expect(answer.text.replace(/\d+\.\d+/g, "N")).toBe(q.text.replace(/\d+\.\d+/g, "N"));
    });
  }
});
