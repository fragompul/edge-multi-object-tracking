import { useCallback, useMemo, useRef, useState } from "react";
import { LineChart } from "../charts/LineChart";
import { ClipVideo, type Layers, Panel, Transport, useClipPlayer } from "../components/ClipPlayer";
import { ErrorBox, Loading, PageHead, Seg, Toggle } from "../components/ui";
import { useClipTracks, useManifest } from "../data/load";
import type { Manifest, SequenceMeta } from "../data/types";
import { askOffline, type Answer } from "../core/planner";
import { type Point, detectLoitering, occupancyHeatmap, occupancySeries } from "../core/analytics";
import { TrackStore } from "../core/store";
import type { FrameTracks, XYXY } from "../core/types";

type Tool = "view" | "line" | "zone";
const SUGGESTIONS = [
  "How many people are there?",
  "How many people crossed the vertical line?",
  "When was it busiest?",
  "How long did people stay in the center?",
  "Summarize this video",
];
const LAYERS: Layers = { trails: true, ids: false, detections: false, gt: false, switches: false };

function StudioStage({ seq, manifest }: { seq: SequenceMeta; manifest: Manifest }) {
  const player = useClipPlayer(seq);
  const [tracker, setTracker] = useState("botsort");
  const clip = useClipTracks(seq.tracks);
  const [tool, setTool] = useState<Tool>("view");
  const [line, setLine] = useState<[Point, Point] | null>([[0.5, 0.0], [0.5, 1.0]]);
  const [draft, setDraft] = useState<Point[]>([]);
  const [zone, setZone] = useState<Point[] | null>([[0.3, 0.45], [0.7, 0.45], [0.7, 0.95], [0.3, 0.95]]);
  const [hover, setHover] = useState<Point | null>(null);
  const [heat, setHeat] = useState(false);
  const [loiter, setLoiter] = useState(4);
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState<{ q: string; a: Answer }[]>([]);
  const chatEnd = useRef<HTMLDivElement>(null);

  const store = useMemo(() => {
    const rows = clip.data?.trackers[tracker];
    if (!rows) return null;
    const frames: FrameTracks = new Map(rows.map((f, i) => [i + 1, f.map((r) => ({ id: r[0], box: [r[1], r[2], r[3], r[4]] as XYXY, conf: 1 }))]));
    return new TrackStore(frames, seq.fps, seq.width, seq.height);
  }, [clip.data, tracker, seq]);

  const norm = (p: Point): Point => [p[0] / seq.width, p[1] / seq.height];
  const toPx = (p: Point): Point => [p[0] * seq.width, p[1] * seq.height];

  const crossings = useMemo(() => (store && line ? store.lineCrossings({ x1: line[0][0], y1: line[0][1], x2: line[1][0], y2: line[1][1] }) : null), [store, line]);
  const dwell = useMemo(() => (store && zone ? store.dwellTime({ region: zone, top_k: 3 }) : null), [store, zone]);
  const zoneSeries = useMemo(() => {
    if (!store) return null;
    const z = zone ? store.zone(zone) : null;
    return [...occupancySeries(store.frames, z)].map(([f, n]) => [(f - 1) / seq.fps, n] as [number, number]);
  }, [store, zone, seq.fps]);
  const loiterers = useMemo(() => {
    if (!store || !zone) return 0;
    const z = store.zone(zone);
    return z ? detectLoitering(store.trajectories, z, seq.fps, loiter).length : 0;
  }, [store, zone, seq.fps, loiter]);
  const heatmap = useMemo(() => (store ? occupancyHeatmap(store.trajectories, seq.width, seq.height, 40) : null), [store, seq]);

  const onPointer = useCallback((x: number, y: number, kind: "click" | "move" | "dblclick") => {
    const p = norm([x, y]);
    if (kind === "move") { setHover(p); return; }
    if (tool === "line" && kind === "click") {
      setDraft((d) => {
        if (d.length === 1) { setLine([d[0], p]); setTool("view"); return []; }
        return [p];
      });
    } else if (tool === "zone") {
      if (kind === "dblclick") {
        setDraft((d) => { if (d.length >= 3) { setZone(d); setTool("view"); return []; } return d; });
      } else setDraft((d) => [...d, p]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool, seq]);

  const drawExtra = useCallback((ctx: CanvasRenderingContext2D) => {
    if (heat && heatmap) {
      const bins = heatmap.length;
      const cw = seq.width / bins, ch = seq.height / bins;
      for (let gy = 0; gy < bins; gy++) for (let gx = 0; gx < bins; gx++) {
        const v = heatmap[gy][gx];
        if (v <= 0.02) continue;
        ctx.fillStyle = `rgba(244,114,182,${Math.min(0.75, v * 0.8)})`;
        ctx.fillRect(gx * cw, gy * ch, cw + 0.5, ch + 0.5);
      }
    }
    const poly = tool === "zone" && draft.length ? draft : zone;
    if (poly && poly.length > 1) {
      ctx.beginPath();
      poly.map(toPx).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (!(tool === "zone" && draft.length)) ctx.closePath();
      else if (hover) { const [hx, hy] = toPx(hover); ctx.lineTo(hx, hy); }
      ctx.fillStyle = "rgba(34,211,238,0.14)"; ctx.fill();
      ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 2.5; ctx.setLineDash(tool === "zone" && draft.length ? [6, 4] : []); ctx.stroke(); ctx.setLineDash([]);
    }
    const lp: [Point, Point] | null = tool === "line" && draft.length === 1 && hover ? [draft[0], hover] : line;
    if (lp) {
      const [[x1, y1], [x2, y2]] = lp.map(toPx) as [Point, Point];
      ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 3.5; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      // arrow toward side B (cross product > 0 side)
      const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len, ny = dx / len; // screen-space normal; side B is where cross(d, p-p1) > 0 -> (-dy, dx) direction
      ctx.fillStyle = "#fbbf24"; ctx.font = "700 13px Inter Variable, sans-serif";
      ctx.beginPath(); ctx.moveTo(mx + nx * 34, my + ny * 34); ctx.lineTo(mx + nx * 18 - ny * 7, my + ny * 18 + nx * 7); ctx.lineTo(mx + nx * 18 + ny * 7, my + ny * 18 - nx * 7); ctx.closePath(); ctx.fill();
      ctx.fillText("B", mx + nx * 52 - 4, my + ny * 52 + 5);
      ctx.fillText("A", mx - nx * 52 - 4, my - ny * 52 + 5);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [heat, heatmap, zone, line, draft, hover, tool, seq]);

  const ask = (q: string) => {
    if (!store || !q.trim()) return;
    setChat((c) => [...c, { q, a: askOffline(store, q) }]);
    setQuestion("");
    setTimeout(() => chatEnd.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 50);
  };

  if (clip.error) return <ErrorBox message={clip.error} />;
  const t = manifest.trackers.find((x) => x.id === tracker)!;

  return (
    <div className="grid split-studio">
      <div className="stack">
        <ClipVideo player={player} />
        <div className="row spread">
          <Seg<Tool> value={tool} onChange={(v) => { setTool(v); setDraft([]); }} label="Tool" options={[
            { value: "view", label: "👁 View" }, { value: "line", label: "╱ Draw line" }, { value: "zone", label: "⬠ Draw zone" }]} />
          <Seg value={tracker} onChange={setTracker} options={manifest.trackers.filter((x) => ["bytetrack", "botsort", "deepsort_osnet_cascade"].includes(x.id)).map((x) => ({ value: x.id, label: x.label, dot: x.color }))} />
        </div>
        {tool !== "view" && <div className="callout">{tool === "line" ? "Click two points on the video to place a counting line." : `Click to add zone corners (${draft.length} so far), double-click to close the polygon.`}</div>}
        <Panel player={player} tracks={clip.data?.trackers[tracker] ?? null} layers={LAYERS} drawExtra={drawExtra} interactive={tool !== "view"} onPointer={onPointer}
          badge={<><i className="dot" style={{ background: t.color }} />{t.label}</>} />
        <Transport player={player} />
        <div className="row" style={{ gap: 20 }}>
          <Toggle checked={heat} onChange={setHeat}>Occupancy heatmap</Toggle>
          <button className="btn sm" onClick={() => { setLine(null); setZone(null); setDraft([]); }}>Clear drawings</button>
        </div>
        {zoneSeries && (
          <div className="card stack">
            <div><h3>People {zone ? "in the zone" : "in frame"} over time</h3><p className="sub">Click the chart to seek.</p></div>
            <LineChart series={[{ name: "people", color: "#22D3EE", points: zoneSeries }]} xLabel="seconds" yLabel="people" fill marker={player.index / seq.fps} onSeek={(x) => player.seek(Math.round(x * seq.fps))} yFormat={(v) => v.toFixed(0)} height={200} />
          </div>
        )}
      </div>

      <div className="stack">
        <div className="card stack">
          <h3>Live analytics</h3>
          <div className="grid cols-2" style={{ gap: 10 }}>
            <div className="stat"><span className="v" style={{ color: "var(--amber)" }}>{crossings ? String(crossings.total) : "—"}</span><span className="l">line crossings{crossings && ` (A→B ${crossings.a_to_b} · B→A ${crossings.b_to_a})`}</span></div>
            <div className="stat"><span className="v" style={{ color: "var(--cyan)" }}>{dwell ? String(dwell.people_who_entered) : "—"}</span><span className="l">people entered the zone</span></div>
            <div className="stat"><span className="v">{dwell ? `${dwell.mean_dwell_s}s` : "—"}</span><span className="l">mean dwell time</span></div>
            <div className="stat"><span className="v">{dwell ? `${dwell.max_dwell_s}s` : "—"}</span><span className="l">longest stay</span></div>
          </div>
          <label className="field">Loitering threshold: {loiter}s
            <input type="range" min={1} max={9} value={loiter} onChange={(e) => setLoiter(+e.target.value)} />
          </label>
          <div className="pill" style={{ alignSelf: "flex-start" }}><i className="dot" style={{ background: "var(--red)" }} />{loiterers} {loiterers === 1 ? "person" : "people"} stayed ≥ {loiter}s in the zone</div>
          <p className="sub" style={{ fontSize: "0.78rem" }}>Geometry uses each person's <b>foot point</b>. Track IDs are anonymous, session-local labels; nothing here identifies anyone.</p>
        </div>

        <div className="card stack">
          <div><h3>Ask EMOT</h3><p className="sub" style={{ fontSize: "0.82rem" }}>Questions are routed to the same deterministic tools the Claude agent calls in the CLI (<code>emot ask</code>) — the model never does the counting.</p></div>
          <div className="chat">
            {chat.length === 0 && <div className="row" style={{ gap: 6 }}>{SUGGESTIONS.map((s) => <button key={s} className="btn sm" onClick={() => ask(s)}>{s}</button>)}</div>}
            {chat.map((c, i) => (
              <div key={i} className="stack" style={{ gap: 6 }}>
                <div className="bubble q">{c.q}</div>
                <div className="bubble a">{c.a.text}
                  {c.a.calls.map((k, j) => <div key={j} className="trace">⤷ {k.name}({JSON.stringify(k.args)})</div>)}
                </div>
              </div>
            ))}
            <div ref={chatEnd} />
          </div>
          <form className="row" style={{ flexWrap: "nowrap" }} onSubmit={(e) => { e.preventDefault(); ask(question); }}>
            <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. how many people in the last 3 seconds?" style={{ flex: 1, minWidth: 0 }} aria-label="Question" />
            <button className="btn primary sm" type="submit">Ask</button>
          </form>
        </div>
      </div>
    </div>
  );
}

export function Studio() {
  const manifest = useManifest();
  const [seqId, setSeqId] = useState("MOT17-09");
  const seq = manifest.data?.sequences.find((s) => s.id === seqId);
  if (manifest.error) return <ErrorBox message={manifest.error} />;
  if (!manifest.data || !seq) return <Loading what="clips" />;
  return (
    <div className="stack" style={{ gap: 22 }}>
      <PageHead eyebrow="Analytics Studio" title="From tracks to answers">
        Draw a counting line or a zone on tracked video and get crossings, dwell time, loitering and occupancy instantly — computed in your browser by a TypeScript port of the Python analytics, verified against it by cross-language parity tests.
      </PageHead>
      <Seg value={seqId} onChange={setSeqId} options={manifest.data.sequences.map((s) => ({ value: s.id, label: s.id.replace("MOT17-", "Scene ") }))} />
      <StudioStage key={seq.id} seq={seq} manifest={manifest.data} />
    </div>
  );
}
