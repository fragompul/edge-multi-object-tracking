import { type ReactNode, type RefObject, useCallback, useEffect, useRef } from "react";
import { dataUrl } from "../data/load";
import type { BoxRow, DetRow, SequenceMeta, SwitchEvent } from "../data/types";
import { drawDetections, drawGroundTruth, drawSwitches, drawTracks } from "../lib/draw";
import { mmss } from "../lib/format";
import { useVideoClock } from "../lib/useVideoClock";

type Listener = (index: number, now: number) => void;

export interface ClipPlayer {
  videoRef: RefObject<HTMLVideoElement>;
  seq: SequenceMeta;
  index: number;
  playing: boolean;
  seek: (i: number) => void;
  toggle: () => void;
  subscribe: (fn: Listener) => () => void;
}

/** One decoder per clip: every panel draws *this* video's frames, so panels can never drift apart. */
export function useClipPlayer(seq: SequenceMeta): ClipPlayer {
  const videoRef = useRef<HTMLVideoElement>(null);
  const listeners = useRef(new Set<Listener>());
  const clock = useVideoClock(videoRef, seq.fps, seq.frames, (i, now) => {
    listeners.current.forEach((l) => l(i, now));
  });
  const subscribe = useCallback((fn: Listener) => {
    listeners.current.add(fn);
    return () => {
      listeners.current.delete(fn);
    };
  }, []);
  return { videoRef, seq, subscribe, ...clock };
}

/** The (visually hidden) decoder. Rendered once per player. */
export function ClipVideo({ player, autoPlay = false }: { player: ClipPlayer; autoPlay?: boolean }) {
  return (
    <video
      ref={player.videoRef}
      key={player.seq.id}
      src={dataUrl(player.seq.video)}
      poster={dataUrl(player.seq.poster)}
      muted
      loop
      playsInline
      autoPlay={autoPlay}
      preload="auto"
      crossOrigin="anonymous"
      style={{ position: "absolute", width: 2, height: 2, opacity: 0, pointerEvents: "none" }}
    />
  );
}

export interface Layers {
  trails: boolean;
  ids: boolean;
  detections: boolean;
  gt: boolean;
  switches: boolean;
}

interface PanelProps {
  player: ClipPlayer;
  tracks: BoxRow[][] | null;
  detections?: DetRow[][];
  gt?: BoxRow[][];
  switches?: SwitchEvent[];
  layers: Layers;
  badge?: ReactNode;
  /** extra drawing in clip-pixel space (zones, counting lines...) */
  drawExtra?: (ctx: CanvasRenderingContext2D, index: number) => void;
  onPointer?: (x: number, y: number, kind: "click" | "move" | "dblclick") => void;
  interactive?: boolean;
}

export function Panel({ player, tracks, detections, gt, switches, layers, badge, drawExtra, onPointer, interactive }: PanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const latest = useRef({ tracks, detections, gt, switches, layers, drawExtra });
  latest.current = { tracks, detections, gt, switches, layers, drawExtra };
  const { seq } = player;

  const draw = useCallback(
    (index: number, now: number) => {
      const c = canvasRef.current;
      const v = player.videoRef.current;
      if (!c || !v) return;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      const L = latest.current;
      if (v.readyState >= 2) ctx.drawImage(v, 0, 0, c.width, c.height);
      else ctx.clearRect(0, 0, c.width, c.height);
      if (L.layers.gt && L.gt) drawGroundTruth(ctx, L.gt[index] ?? [], 1);
      if (L.layers.detections && L.detections) drawDetections(ctx, L.detections[index] ?? [], 1);
      if (L.tracks) drawTracks(ctx, L.tracks, index, { trails: L.layers.trails, ids: L.layers.ids, scale: 1 });
      if (L.layers.switches && L.switches) drawSwitches(ctx, L.switches, index, 1, now);
      L.drawExtra?.(ctx, index);
    },
    [player],
  );

  const point = (e: React.MouseEvent<HTMLCanvasElement>, kind: "click" | "move" | "dblclick") => {
    if (!onPointer || !interactive) return;
    const r = e.currentTarget.getBoundingClientRect();
    onPointer(((e.clientX - r.left) / r.width) * seq.width, ((e.clientY - r.top) / r.height) * seq.height, kind);
  };

  useEffect(() => player.subscribe(draw), [player, draw]);
  // redraw when paused and a layer/tracker/option changes
  useEffect(() => {
    draw(player.index, performance.now());
  }, [draw, player.index, tracks, layers, detections, gt, switches, drawExtra]);
  // keep the switch ring pulsing while paused on one
  useEffect(() => {
    if (player.playing || !layers.switches || !switches?.length) return;
    let raf = 0;
    const tick = () => {
      draw(player.index, performance.now());
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [player.playing, player.index, layers.switches, switches, draw]);

  return (
    <div className="stage">
      <canvas
        ref={canvasRef}
        className={`under${interactive ? " interactive" : ""}`}
        width={seq.width}
        height={seq.height}
        style={{ cursor: interactive ? "crosshair" : undefined }}
        onClick={(e) => point(e, "click")}
        onDoubleClick={(e) => point(e, "dblclick")}
        onMouseMove={(e) => interactive && point(e, "move")}
      />
      {badge && <div className="badge">{badge}</div>}
    </div>
  );
}

export interface Lane {
  label: string;
  color: string;
  ticks: { f: number }[];
}

/** One play button, one scrubber; each lane draws its identity-switch ticks on its own row. */
export function Transport({ player, lanes = [] }: { player: ClipPlayer; lanes?: Lane[] }) {
  const { seq, index } = player;
  const trackRef = useRef<HTMLDivElement>(null);
  const seekFromEvent = (e: React.MouseEvent) => {
    const r = trackRef.current?.getBoundingClientRect();
    if (!r) return;
    player.seek(Math.round(((e.clientX - r.left) / r.width) * (seq.frames - 1)));
  };
  const laneH = 20;
  const height = Math.max(34, lanes.length * laneH + 10);
  return (
    <div className="row" style={{ gap: 12, flexWrap: "nowrap", alignItems: "stretch" }}>
      <button className="btn sm" onClick={player.toggle} aria-label={player.playing ? "Pause" : "Play"} style={{ minWidth: 84, justifyContent: "center", alignSelf: "center" }}>
        {player.playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      {lanes.length > 1 && (
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-around", fontSize: "0.74rem", color: "var(--muted)", minWidth: 96 }}>
          {lanes.map((l) => (
            <span key={l.label} className="row" style={{ gap: 6, flexWrap: "nowrap", height: laneH }}><i className="dot" style={{ background: l.color }} />{l.label.replace(/\(.*\)/, "").trim()}</span>
          ))}
        </div>
      )}
      <div className="timeline" ref={trackRef} onClick={seekFromEvent} style={{ flex: 1, height, borderRadius: 14 }} role="slider" aria-label="Seek" aria-valuemin={0} aria-valuemax={seq.frames - 1} aria-valuenow={index}>
        {lanes.map((l, li) =>
          l.ticks.map((t, i) => (
            <span key={`${li}-${i}`} className="tick" style={{ left: `${(t.f / (seq.frames - 1)) * 100}%`, top: 5 + li * laneH, height: laneH - 6, bottom: "auto", background: "var(--red)" }} />
          )),
        )}
        <span className="head" style={{ left: `${(index / (seq.frames - 1)) * 100}%` }} />
      </div>
      <span className="mono" style={{ color: "var(--muted)", minWidth: 84, textAlign: "right", alignSelf: "center" }}>
        {mmss(index / seq.fps)} / {mmss(seq.frames / seq.fps)}
      </span>
    </div>
  );
}
