import type { XYXY } from "./types";
import { type Mat, add, diag, eye, inverse, matMul, matVec, sub, transpose } from "./matrix";

/**
 * Constant-velocity Kalman filter over `[cx, cy, aspect, height]` -- a line-for-line port of
 * `emot/tracking/kalman.py`, so the browser tracker reproduces the Python one (verified by
 * `bytetrack.parity.test.ts` against fixtures generated from Python).
 */
const STD_POS = 1 / 20;
const STD_VEL = 1 / 160;

export function xyxyToState([l, t, r, b]: XYXY): number[] {
  const w = r - l;
  const h = b - t;
  return [l + w / 2, t + h / 2, h > 1e-6 ? w / h : 1e-6, h];
}

export function stateToXyxy(s: number[]): XYXY {
  const [cx, cy, a, h] = s;
  const w = a * h;
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

const F: Mat = (() => {
  const f = eye(8);
  for (let i = 0; i < 4; i++) f[i][i + 4] = 1;
  return f;
})();
const FT = transpose(F);

export class KalmanBox {
  mean: number[];
  cov: Mat;

  constructor(xyxy: XYXY) {
    this.mean = [...xyxyToState(xyxy), 0, 0, 0, 0];
    const h = this.mean[3];
    const std = [
      2 * STD_POS * h, 2 * STD_POS * h, 1e-2, 2 * STD_POS * h,
      10 * STD_VEL * h, 10 * STD_VEL * h, 1e-5, 10 * STD_VEL * h,
    ];
    this.cov = diag(std.map((s) => s * s));
  }

  predict(): void {
    const h = this.mean[3];
    const std = [
      STD_POS * h, STD_POS * h, 1e-2, STD_POS * h,
      STD_VEL * h, STD_VEL * h, 1e-5, STD_VEL * h,
    ];
    const q = diag(std.map((s) => s * s));
    this.mean = matVec(F, this.mean);
    this.cov = add(matMul(matMul(F, this.cov), FT), q);
  }

  update(xyxy: XYXY): void {
    const z = xyxyToState(xyxy);
    const h = this.mean[3];
    const obs = [STD_POS * h, STD_POS * h, 1e-1, STD_POS * h];
    const r = diag(obs.map((s) => s * s));
    // H = [I4 | 0]: projection is the top-left 4x4 block / the first four state entries.
    const s = add(this.cov.slice(0, 4).map((row) => row.slice(0, 4)), r);
    const pht = this.cov.map((row) => row.slice(0, 4)); // P H^T  (8x4)
    const k = matMul(pht, inverse(s)); // 8x4
    const y = z.map((v, i) => v - this.mean[i]);
    const ky = matVec(k, y);
    this.mean = this.mean.map((v, i) => v + ky[i]);
    const hp = this.cov.slice(0, 4); // H P (4x8)
    this.cov = sub(this.cov, matMul(k, hp));
  }

  get xyxy(): XYXY {
    return stateToXyxy(this.mean);
  }

  /** Re-expresses the track after a camera move (2x3 similarity warp), as in BoT-SORT. */
  applyAffine(w: [number, number, number, number, number, number]): void {
    const [a, b, tx, c, d, ty] = w;
    const scale = Math.sqrt(Math.abs(a * d - b * c));
    const [cx, cy, , h] = this.mean;
    this.mean[0] = a * cx + b * cy + tx;
    this.mean[1] = c * cx + d * cy + ty;
    const [vx, vy] = [this.mean[4], this.mean[5]];
    this.mean[4] = a * vx + b * vy;
    this.mean[5] = c * vx + d * vy;
    this.mean[3] = h * scale;
    this.mean[7] *= scale;
    const blk = eye(8);
    blk[0][0] = a; blk[0][1] = b; blk[1][0] = c; blk[1][1] = d;
    blk[4][4] = a; blk[4][5] = b; blk[5][4] = c; blk[5][5] = d;
    this.cov = matMul(matMul(blk, this.cov), transpose(blk));
  }
}

