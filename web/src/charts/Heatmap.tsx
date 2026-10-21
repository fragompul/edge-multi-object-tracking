import { Tip, useTip } from "./Tip";
import { useWidth } from "./useWidth";

interface Props {
  rows: string[];
  cols: string[];
  colSub?: string[];
  values: (number | null)[][];
  format?: (v: number) => string;
  /** true: higher is better (bright), false: lower is better */
  higherBetter?: boolean;
  cell?: number;
}

const stops = ["#1b1b4b", "#4b2a8c", "#a3358d", "#f0586a", "#fdb74e", "#fcf1a4"];
function colour(t: number): string {
  const k = Math.min(stops.length - 1.0001, Math.max(0, t) * (stops.length - 1));
  const i = Math.floor(k);
  const f = k - i;
  const a = stops[i], b = stops[i + 1];
  const mix = (s: number) => Math.round(parseInt(a.slice(s, s + 2), 16) * (1 - f) + parseInt(b.slice(s, s + 2), 16) * f);
  return `rgb(${mix(1)},${mix(3)},${mix(5)})`;
}

export function Heatmap({ rows, cols, colSub, values, format = (v) => v.toFixed(3), higherBetter = true, cell = 44 }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const { tip, show, hide } = useTip();
  const labelW = 168;
  const cw = Math.max(46, (width - labelW - 8) / cols.length);
  const height = rows.length * cell + 54;
  const flat = values.flat().filter((v): v is number => v !== null);
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const norm = (v: number) => ((v - lo) / (hi - lo || 1)) * (higherBetter ? 1 : -1) + (higherBetter ? 0 : 1);

  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="heatmap">
        <g transform={`translate(${labelW},34)`}>
          {cols.map((c, j) => (
            <g key={c} transform={`translate(${j * cw + cw / 2},-16)`}>
              <text textAnchor="middle" style={{ fill: "var(--ink-dim)", fontWeight: 650 }}>{c}</text>
              {colSub && <text y={12} textAnchor="middle" style={{ fontSize: 9.5 }}>{colSub[j]}</text>}
            </g>
          ))}
          {rows.map((r, i) => (
            <g key={r} transform={`translate(0,${i * cell})`}>
              <text x={-12} y={cell / 2 + 4} textAnchor="end" style={{ fill: "var(--ink-dim)", fontSize: 12 }}>{r}</text>
              {cols.map((c, j) => {
                const v = values[i][j];
                return (
                  <g key={c} transform={`translate(${j * cw},0)`}
                     onMouseMove={(e) => v !== null && show(e, ref.current, <><b>{r}</b> · {c}<br />{format(v)}</>)} onMouseLeave={hide}>
                    <rect x={2} y={2} width={cw - 4} height={cell - 4} rx={7} fill={v === null ? "var(--surface)" : colour(norm(v))} />
                    {v !== null && (
                      <text x={cw / 2} y={cell / 2 + 4} textAnchor="middle" style={{ fill: norm(v) > 0.55 ? "#0b1020" : "#fff", fontWeight: 650, fontSize: 11.5 }}>{format(v)}</text>
                    )}
                  </g>
                );
              })}
            </g>
          ))}
        </g>
      </svg>
      <Tip tip={tip} />
    </div>
  );
}
