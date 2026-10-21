/**
 * Minimum-cost assignment on a rectangular cost matrix (Jonker-Volgenant-style shortest
 * augmenting paths with potentials, O(n^2 m)). Returns `[row, col]` pairs, one per
 * min(rows, cols) -- the same contract as `scipy.optimize.linear_sum_assignment`.
 */
export function linearSumAssignment(cost: number[][]): [number, number][] {
  const nRows = cost.length;
  if (nRows === 0) return [];
  const nCols = cost[0].length;
  if (nCols === 0) return [];
  const transposed = nRows > nCols;
  const n = transposed ? nCols : nRows;
  const m = transposed ? nRows : nCols;
  const c = (i: number, j: number) => (transposed ? cost[j][i] : cost[i][j]);

  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1); // p[j] = row assigned to column j (1-based, 0 = none)
  const way = new Int32Array(m + 1);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(Infinity);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = c(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const pairs: [number, number][] = [];
  for (let j = 1; j <= m; j++) {
    if (p[j] === 0) continue;
    const i = p[j] - 1;
    pairs.push(transposed ? [j - 1, i] : [i, j - 1]);
  }
  return pairs.sort((a, b) => a[0] - b[0]);
}
