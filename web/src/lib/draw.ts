import type { BoxRow, DetRow, SwitchEvent } from "../data/types";

/** Stable, well-separated colour per identity (golden-ratio hue walk), as in the Python renderer. */
export function idColor(id: number, s = 65, l = 62): string {
  return `hsl(${Math.round(((id * 0.61803398875) % 1) * 360)} ${s}% ${l}%)`;
}

export interface DrawOptions {
  trails: boolean;
  ids: boolean;
  scale: number; // canvas pixels per clip pixel
  accent?: string;
}

function brackets(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, lw: number) {
  const arm = Math.max(6, Math.min(x2 - x1, y2 - y1) * 0.22);
  ctx.lineWidth = lw;
  ctx.beginPath();
  for (const [cx, cy, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]] as const) {
    ctx.moveTo(cx + dx * arm, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * arm);
  }
  ctx.stroke();
}

export function drawTracks(
  ctx: CanvasRenderingContext2D,
  frames: BoxRow[][],
  index: number,
  o: DrawOptions,
): void {
  const s = o.scale;
  const lw = Math.max(1.5, 2 * s);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  if (o.trails) {
    const history = new Map<number, [number, number][]>();
    for (let f = Math.max(0, index - 45); f <= index; f++)
      for (const [id, x1, , x2, y2] of frames[f] ?? []) {
        const arr = history.get(id) ?? [];
        arr.push([((x1 + x2) / 2) * s, y2 * s]);
        history.set(id, arr);
      }
    for (const [id, pts] of history) {
      if (pts.length < 2 || !(frames[index] ?? []).some((b) => b[0] === id)) continue;
      ctx.strokeStyle = idColor(id);
      for (let i = 1; i < pts.length; i++) {
        ctx.globalAlpha = 0.15 + 0.85 * (i / pts.length);
        ctx.lineWidth = Math.max(1, lw * (i / pts.length) * 1.6);
        ctx.beginPath();
        ctx.moveTo(pts[i - 1][0], pts[i - 1][1]);
        ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  for (const [id, x1, y1, x2, y2] of frames[index] ?? []) {
    const c = idColor(id);
    const [a, b, w, h] = [x1 * s, y1 * s, x2 * s, y2 * s];
    ctx.fillStyle = c;
    ctx.globalAlpha = 0.1;
    ctx.fillRect(a, b, w - a, h - b);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = c;
    brackets(ctx, a, b, w, h, lw);
    if (o.ids) {
      const fs = Math.max(10, 11 * Math.min(1.6, s * 1.1));
      ctx.font = `700 ${fs}px "Inter Variable", system-ui, sans-serif`;
      const label = String(id);
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = c;
      ctx.globalAlpha = 0.92;
      ctx.fillRect(a, b - fs - 5, tw, fs + 5);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0b1020";
      ctx.fillText(label, a + 4, b - 5);
    }
  }
}

export function drawDetections(ctx: CanvasRenderingContext2D, dets: DetRow[], s: number): void {
  ctx.strokeStyle = "rgba(230,234,247,0.75)";
  ctx.lineWidth = Math.max(1, s);
  ctx.setLineDash([4, 3]);
  for (const [x1, y1, x2, y2] of dets) ctx.strokeRect(x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s);
  ctx.setLineDash([]);
}

export function drawGroundTruth(ctx: CanvasRenderingContext2D, gt: BoxRow[], s: number): void {
  ctx.strokeStyle = "rgba(163,230,53,0.9)";
  ctx.lineWidth = Math.max(1, 1.4 * s);
  for (const [, x1, y1, x2, y2] of gt) ctx.strokeRect(x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s);
}

/** Pulsing red ring on the ground-truth person whose identity was just broken. */
export function drawSwitches(
  ctx: CanvasRenderingContext2D,
  events: SwitchEvent[],
  index: number,
  s: number,
  now: number,
): void {
  for (const e of events) {
    const age = index - e.f;
    if (age < 0 || age > 14 || !e.box) continue;
    const [x1, y1, x2, y2] = e.box;
    const pulse = 1 + 0.08 * Math.sin(now / 90);
    const cx = ((x1 + x2) / 2) * s;
    const cy = ((y1 + y2) / 2) * s;
    const rw = ((x2 - x1) / 2 + 6) * s * pulse;
    const rh = ((y2 - y1) / 2 + 6) * s * pulse;
    ctx.globalAlpha = 1 - age / 15;
    ctx.strokeStyle = "#f87171";
    ctx.lineWidth = Math.max(2, 3 * s);
    ctx.beginPath();
    ctx.ellipse(cx, cy, rw, rh, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "#f87171";
    ctx.font = `800 ${Math.max(11, 12 * s)}px "Inter Variable", system-ui, sans-serif`;
    ctx.fillText(`ID ${e.from}→${e.to}`, x1 * s, Math.max(14, y1 * s - 10));
  }
  ctx.globalAlpha = 1;
}
