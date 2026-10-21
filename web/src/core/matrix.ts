/** Tiny dense-matrix helpers for the 8-state Kalman filter (row-major `number[][]`). */
export type Mat = number[][];

export const zeros = (r: number, c: number): Mat =>
  Array.from({ length: r }, () => new Array<number>(c).fill(0));

export const eye = (n: number): Mat => {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m[i][i] = 1;
  return m;
};

export const diag = (v: number[]): Mat => {
  const m = zeros(v.length, v.length);
  v.forEach((x, i) => (m[i][i] = x));
  return m;
};

export const transpose = (a: Mat): Mat => a[0].map((_, j) => a.map((row) => row[j]));

export function matMul(a: Mat, b: Mat): Mat {
  const out = zeros(a.length, b[0].length);
  for (let i = 0; i < a.length; i++)
    for (let k = 0; k < b.length; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < b[0].length; j++) out[i][j] += aik * b[k][j];
    }
  return out;
}

export const matVec = (a: Mat, v: number[]): number[] =>
  a.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));

export const add = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((x, j) => x + b[i][j]));
export const sub = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((x, j) => x - b[i][j]));

/** Gauss-Jordan inverse with partial pivoting (small SPD matrices only). */
export function inverse(a: Mat): Mat {
  const n = a.length;
  const m = a.map((row, i) => [...row, ...eye(n)[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    if (Math.abs(m[pivot][col]) < 1e-14) throw new Error("singular matrix");
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const p = m[col][col];
    for (let j = 0; j < 2 * n; j++) m[col][j] /= p;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[col][j];
    }
  }
  return m.map((row) => row.slice(n));
}
