import { scaleLinear, scaleLog } from "d3-scale";
import type { ReactNode } from "react";
import { Tip, useTip } from "./Tip";
import { useWidth } from "./useWidth";

export interface Point {
  x: number;
  y: number;
  label: string;
  color: string;
  tip?: ReactNode;
  r?: number;
}

interface Props {
  points: Point[];
  xLabel: string;
  yLabel: string;
  height?: number;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  /** Draw the Pareto frontier (lower x better, higher y better). */
  pareto?: boolean;
  logX?: boolean;
  lines?: { name: string; color: string; pts: [number, number][] }[];
}

export function Scatter({ points, xLabel, yLabel, height = 340, xFormat = (v) => v.toFixed(0), yFormat = (v) => v.toFixed(2), pareto, lines, logX }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const { tip, show, hide } = useTip();
  const m = { top: 16, right: 24, bottom: 44, left: 56 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const all = [...points, ...(lines?.flatMap((l) => l.pts.map(([x, y]) => ({ x, y }))) ?? [])];
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const px = (Math.max(...xs) - Math.min(...xs) || 1) * 0.08;
  const py = (Math.max(...ys) - Math.min(...ys) || 1) * 0.12;
  const x = logX
    ? scaleLog().domain([Math.min(...xs) * 0.8, Math.max(...xs) * 1.25]).range([0, iw])
    : scaleLinear().domain([Math.max(0, Math.min(...xs) - px), Math.max(...xs) + px]).range([0, iw]).nice();
  const xTicks = logX ? [2, 5, 10, 20, 50, 100, 200, 500].filter((t) => t >= x.domain()[0] && t <= x.domain()[1]) : x.ticks(6);
  const y = scaleLinear().domain([Math.min(...ys) - py, Math.max(...ys) + py]).range([ih, 0]).nice();

  let frontier: Point[] = [];
  if (pareto) {
    const sorted = [...points].sort((a, b) => a.x - b.x);
    let best = -Infinity;
    for (const p of sorted) if (p.y > best) { best = p.y; frontier.push(p); }
  }
  frontier = frontier.length > 1 ? frontier : [];

  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${yLabel} versus ${xLabel}`}>
        <g transform={`translate(${m.left},${m.top})`}>
          {y.ticks(5).map((t) => (
            <g key={`y${t}`} transform={`translate(0,${y(t)})`}>
              <line className="grid-line" x1={0} x2={iw} />
              <text x={-8} y={4} textAnchor="end">{yFormat(t)}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={`x${t}`} transform={`translate(${x(t)},${ih})`}>
              <line className="grid-line" y1={0} y2={-ih} />
              <text y={18} textAnchor="middle">{xFormat(t)}</text>
            </g>
          ))}
          <text x={iw / 2} y={ih + 38} textAnchor="middle">{xLabel}</text>
          <text transform={`translate(${-42},${ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
          {frontier.length > 1 && (
            <polyline
              fill="none" stroke="var(--lime)" strokeWidth={1.6} strokeDasharray="5 4" opacity={0.8}
              points={frontier.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")}
            />
          )}
          {lines?.map((l) => (
            <g key={l.name}>
              <polyline fill="none" stroke={l.color} strokeWidth={2.2} opacity={0.85} points={l.pts.map(([a, b]) => `${x(a)},${y(b)}`).join(" ")} />
              <text x={x(l.pts[l.pts.length - 1][0]) + 6} y={y(l.pts[l.pts.length - 1][1]) + 4} style={{ fill: l.color, fontWeight: 650 }}>{l.name}</text>
            </g>
          ))}
          {points.map((p) => (
            <g key={p.label + p.x + p.y} transform={`translate(${x(p.x)},${y(p.y)})`}
               onMouseMove={(e) => show(e, ref.current, p.tip ?? <b>{p.label}</b>)} onMouseLeave={hide}>
              <circle r={(p.r ?? 7) + 5} fill={p.color} opacity={0.14} />
              <circle r={p.r ?? 7} fill={p.color} stroke="var(--bg)" strokeWidth={2} />
              <text x={(p.r ?? 7) + 7} y={4} style={{ fill: "var(--ink-dim)", fontSize: 10.5 }}>{p.label}</text>
            </g>
          ))}
          <line className="axis-line" x1={0} x2={iw} y1={ih} y2={ih} />
          <line className="axis-line" x1={0} x2={0} y1={0} y2={ih} />
        </g>
      </svg>
      <Tip tip={tip} />
    </div>
  );
}
