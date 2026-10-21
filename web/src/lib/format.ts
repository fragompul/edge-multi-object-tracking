export const f3 = (x: number): string => x.toFixed(3);
export const f1 = (x: number): string => x.toFixed(1);
export const pct = (x: number, d = 1): string => `${(x * 100).toFixed(d)}%`;
export const mean = (v: number[]): number => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0);
export const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
export const mmss = (s: number): string => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function groupBy<T, K extends string | number>(rows: T[], key: (r: T) => K): Map<K, T[]> {
  const m = new Map<K, T[]>();
  for (const r of rows) {
    const k = key(r);
    const arr = m.get(k);
    if (arr) arr.push(r);
    else m.set(k, [r]);
  }
  return m;
}
