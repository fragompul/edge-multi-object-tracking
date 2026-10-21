import { scaleBand, scaleLinear } from "d3-scale";
import { Tip, useTip } from "./Tip";
import { useWidth } from "./useWidth";

export interface Bar {
  series: string;
  value: number;
  color: string;
}
export interface BarGroup {
  label: string;
  sub?: string;
  bars: Bar[];
}

interface Props {
  groups: BarGroup[];
  height?: number;
  yMax?: number;
  format?: (v: number) => string;
  yLabel?: string;
  labels?: boolean;
  legend?: { name: string; color: string }[];
  title?: string;
}

/** Grouped vertical bar chart (SVG) with hover tooltips. */
export function BarChart({ groups, height = 300, yMax, format = (v) => v.toFixed(3), yLabel, labels = true, legend, title }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const { tip, show, hide } = useTip();
  const m = { top: title ? 30 : 14, right: 12, bottom: groups.some((g) => g.sub) ? 52 : 38, left: 52 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const max = yMax ?? Math.max(...groups.flatMap((g) => g.bars.map((b) => b.value)), 1e-9) * 1.12;
  const x0 = scaleBand<string>().domain(groups.map((g) => g.label)).range([0, iw]).paddingInner(0.22).paddingOuter(0.08);
  const seriesNames = [...new Set(groups.flatMap((g) => g.bars.map((b) => b.series)))];
  const x1 = scaleBand<string>().domain(seriesNames).range([0, x0.bandwidth()]).padding(0.12);
  const y = scaleLinear().domain([0, max]).range([ih, 0]).nice();
  const ticks = y.ticks(5);
  const showLabels = labels && x1.bandwidth() > 26;

  return (
    <div className="chart" ref={ref}>
      {legend && (
        <div className="legend" style={{ marginBottom: 8 }}>
          {legend.map((l) => (
            <span key={l.name}><i className="dot" style={{ background: l.color }} />{l.name}</span>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title ?? "bar chart"}>
        {title && <text className="title" x={m.left} y={14}>{title}</text>}
        <g transform={`translate(${m.left},${m.top})`}>
          {ticks.map((t) => (
            <g key={t} transform={`translate(0,${y(t)})`}>
              <line className="grid-line" x1={0} x2={iw} />
              <text x={-8} y={4} textAnchor="end">{format(t)}</text>
            </g>
          ))}
          {yLabel && <text transform={`translate(${-40},${ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>}
          {groups.map((g) => (
            <g key={g.label} transform={`translate(${x0(g.label)},0)`}>
              {g.bars.map((b) => {
                const bx = x1(b.series) ?? 0;
                const h = Math.max(0, ih - y(b.value));
                return (
                  <g key={b.series}>
                    <rect
                      x={bx} y={y(b.value)} width={x1.bandwidth()} height={h} rx={3} fill={b.color}
                      opacity={0.92}
                      onMouseMove={(e) => show(e, ref.current, <><b>{g.label.replace("\n", " ")}</b><br />{b.series}: {format(b.value)}</>)}
                      onMouseLeave={hide}
                    />
                    {showLabels && (
                      <text x={bx + x1.bandwidth() / 2} y={y(b.value) - 4} textAnchor="middle" style={{ fontSize: 9.5, fill: "var(--ink-dim)" }}>
                        {format(b.value)}
                      </text>
                    )}
                  </g>
                );
              })}
              {g.label.split("\n").map((line, i) => (
                <text key={i} x={x0.bandwidth() / 2} y={ih + 18 + i * 13} textAnchor="middle" style={{ fill: i ? "var(--muted)" : "var(--ink-dim)" }}>{line}</text>
              ))}
              {g.sub && <text x={x0.bandwidth() / 2} y={ih + 44} textAnchor="middle" style={{ fontSize: 10 }}>{g.sub}</text>}
            </g>
          ))}
          <line className="axis-line" x1={0} x2={iw} y1={ih} y2={ih} />
        </g>
      </svg>
      <Tip tip={tip} />
    </div>
  );
}
