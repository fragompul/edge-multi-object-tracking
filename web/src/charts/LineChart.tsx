import { scaleLinear } from "d3-scale";
import { Tip, useTip } from "./Tip";
import { useWidth } from "./useWidth";

export interface Series {
  name: string;
  color: string;
  points: [number, number][];
}

interface Props {
  series: Series[];
  xLabel: string;
  yLabel: string;
  height?: number;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  xTicks?: number[];
  fill?: boolean;
  yMin?: number;
  /** vertical marker (e.g. playhead) in x units */
  marker?: number;
  onSeek?: (x: number) => void;
}

export function LineChart({ series, xLabel, yLabel, height = 260, xFormat = (v) => v.toFixed(0), yFormat = (v) => v.toFixed(1), xTicks, fill, yMin, marker, onSeek }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const { tip, show, hide } = useTip();
  const m = { top: 12, right: 20, bottom: 40, left: 52 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const x = scaleLinear().domain([Math.min(...xs), Math.max(...xs)]).range([0, iw]);
  const y = scaleLinear().domain([yMin ?? Math.min(0, ...ys), Math.max(...ys) * 1.1 || 1]).range([ih, 0]).nice();
  const seek = (e: React.MouseEvent<SVGRectElement>) => {
    if (!onSeek) return;
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(x.invert(((e.clientX - r.left) / r.width) * iw));
  };

  return (
    <div className="chart" ref={ref}>
      {series.length > 1 && (
        <div className="legend" style={{ marginBottom: 6 }}>
          {series.map((s) => <span key={s.name}><i className="dot" style={{ background: s.color }} />{s.name}</span>)}
        </div>
      )}
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${yLabel} versus ${xLabel}`}>
        <g transform={`translate(${m.left},${m.top})`}>
          {y.ticks(4).map((t) => (
            <g key={t} transform={`translate(0,${y(t)})`}>
              <line className="grid-line" x1={0} x2={iw} />
              <text x={-8} y={4} textAnchor="end">{yFormat(t)}</text>
            </g>
          ))}
          {(xTicks ?? x.ticks(6)).map((t) => (
            <text key={t} x={x(t)} y={ih + 18} textAnchor="middle">{xFormat(t)}</text>
          ))}
          <text x={iw / 2} y={ih + 36} textAnchor="middle">{xLabel}</text>
          <text transform={`translate(${-38},${ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
          {series.map((s) => {
            const d = s.points.map(([a, b]) => `${x(a)},${y(b)}`).join(" ");
            return (
              <g key={s.name}>
                {fill && <polygon points={`${x(s.points[0][0])},${ih} ${d} ${x(s.points[s.points.length - 1][0])},${ih}`} fill={s.color} opacity={0.12} />}
                <polyline fill="none" stroke={s.color} strokeWidth={2.2} strokeLinejoin="round" points={d} />
                {s.points.length <= 12 && s.points.map(([a, b]) => (
                  <circle key={a} cx={x(a)} cy={y(b)} r={4.5} fill={s.color} stroke="var(--bg)" strokeWidth={2}
                          onMouseMove={(e) => show(e, ref.current, <><b>{s.name}</b><br />{xLabel}: {xFormat(a)}<br />{yLabel}: {yFormat(b)}</>)} onMouseLeave={hide} />
                ))}
              </g>
            );
          })}
          {marker !== undefined && <line x1={x(marker)} x2={x(marker)} y1={0} y2={ih} stroke="var(--ink)" strokeWidth={1.5} opacity={0.8} />}
          <line className="axis-line" x1={0} x2={iw} y1={ih} y2={ih} />
          {onSeek && <rect x={0} y={0} width={iw} height={ih} fill="transparent" style={{ cursor: "pointer" }} onClick={seek} />}
        </g>
      </svg>
      <Tip tip={tip} />
    </div>
  );
}
