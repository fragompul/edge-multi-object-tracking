import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ClipVideo, type Layers, Panel, useClipPlayer } from "../components/ClipPlayer";
import { Loading, Stat } from "../components/ui";
import { useClipTracks, useManifest, useResults } from "../data/load";
import type { Manifest, Results, SequenceMeta } from "../data/types";
import { f3, pct } from "../lib/format";
import { edgeFind, edgeRows, filterRows, meansByTracker } from "../lib/stats";

const HERO_LAYERS: Layers = { trails: true, ids: true, detections: false, gt: false, switches: false };

function HeroPlayer({ seq, manifest }: { seq: SequenceMeta; manifest: Manifest }) {
  const player = useClipPlayer(seq);
  const clip = useClipTracks(seq.tracks);
  const [hover, setHover] = useState(false);
  const tracker = manifest.trackers.find((t) => t.id === "botsort")!;
  return (
    <div onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} style={{ position: "relative" }}>
      <ClipVideo player={player} autoPlay />
      <Panel
        player={player}
        tracks={clip.data?.trackers.botsort ?? null}
        layers={HERO_LAYERS}
        badge={<><i className="dot" style={{ background: tracker.color }} />{tracker.label} · live overlay</>}
      />
      <button className="btn sm" style={{ position: "absolute", right: 12, bottom: 12, opacity: hover || !player.playing ? 1 : 0, transition: "opacity .2s" }} onClick={player.toggle}>
        {player.playing ? "❚❚" : "▶"}
      </button>
    </div>
  );
}

function Findings({ results }: { results: Results }) {
  const f = useMemo(() => {
    const moving = meansByTracker(filterRows(results, { camera: "moving" }));
    const all = meansByTracker(filterRows(results, {}));
    const get = (rows: ReturnType<typeof meansByTracker>, t: string) => rows.find((r) => r.tracker === t)!;
    const edge = edgeRows(results, "main");
    const squareVsRect = edgeFind(edge, "yolo11s", "onnx_fp32");
    const int8 = edgeFind(edge, "yolo11s", "onnx_int8_static");
    return {
      cmcSw: 1 - get(moving, "botsort_cmc").num_switches / get(moving, "bytetrack").num_switches,
      reidHota: get(all, "deepsort_osnet").hota - get(all, "deepsort").hota,
      cascadeSw: 1 - get(all, "deepsort_osnet_cascade").num_switches / get(all, "deepsort_osnet").num_switches,
      sizeRatio: squareVsRect && int8 ? squareVsRect.size_mb / int8.size_mb : null,
      apDrop: squareVsRect && int8 ? squareVsRect.ap50 - int8.ap50 : null,
    };
  }, [results]);
  return (
    <div className="grid cols-4">
      <div className="card"><Stat value={`−${pct(f.cmcSw, 0)}`} label="identity switches from camera-motion compensation on moving-camera scenes" color="var(--violet)" /></div>
      <div className="card"><Stat value={`+${f3(f.reidHota)}`} label="HOTA when DeepSORT's generic ResNet18 is swapped for a real ReID net (OSNet)" color="var(--lime)" /></div>
      <div className="card"><Stat value={`−${pct(f.cascadeSw, 0)}`} label="identity switches from the paper's matching cascade (vs single-stage matching)" color="var(--cyan)" /></div>
      <div className="card"><Stat value={f.sizeRatio ? `${f.sizeRatio.toFixed(1)}× smaller` : "—"} label={`INT8 model at only ${f.apDrop !== null ? (f.apDrop * 100).toFixed(1) : "—"} pt AP50 cost — but not faster on CPU`} color="var(--amber)" /></div>
    </div>
  );
}

export function Home() {
  const manifest = useManifest();
  const results = useResults();
  const hero = manifest.data?.sequences.find((s) => s.id === "MOT17-04") ?? manifest.data?.sequences[0];

  const totals = useMemo(() => {
    if (!results.data) return null;
    const r = results.data;
    return {
      runs: r.tracking.length,
      seqs: new Set(r.tracking.map((t) => t.sequence)).size,
      backbones: new Set(r.tracking.map((t) => t.backbone)).size,
      variants: new Set(r.tracking.map((t) => t.tracker)).size,
    };
  }, [results.data]);

  return (
    <div className="stack" style={{ gap: 44 }}>
      <section className="hero fade-in">
        <div>
          <div className="eyebrow">Edge multi-object tracking</div>
          <h1>See every identity.<br /><span className="grad-text">On the edge.</span></h1>
          <p className="lead">
            Trackers built from scratch (ByteTrack, BoT-SORT, DeepSORT), benchmarked on real surveillance video, squeezed through ONNX and INT8 for CPU deployment — and running <b>live in your browser</b>, no upload, no server.
          </p>
          <div className="row cta">
            <Link className="btn primary" to="/arena">Open the Tracker Arena →</Link>
            <Link className="btn" to="/live">Run it on my webcam</Link>
          </div>
          {totals && (
            <div className="kpis">
              <Stat value={totals.runs} label="tracking runs benchmarked" />
              <Stat value={totals.seqs} label="MOT17 sequences" />
              <Stat value={totals.backbones} label="YOLO backbones" />
              <Stat value={totals.variants} label="tracker variants" />
            </div>
          )}
        </div>
        <div>{hero && manifest.data ? <HeroPlayer seq={hero} manifest={manifest.data} /> : <Loading what="demo clip" />}</div>
      </section>

      <section className="stack">
        <div><div className="eyebrow">Headline findings</div><h2>What the benchmark actually shows</h2></div>
        {results.data ? <Findings results={results.data} /> : <Loading what="results" />}
        <p style={{ fontSize: "0.86rem", color: "var(--muted)" }}>
          Means over 7 MOT17 sequences × 6 detectors. Every number is computed in your browser from the committed result files — open <Link to="/benchmarks">Benchmarks</Link> to slice them yourself.
        </p>
      </section>

      <section className="stack">
        <div><div className="eyebrow">Explore</div><h2>Six ways in</h2></div>
        <div className="grid cols-3">
          {[
            ["/arena", "◎", "Tracker Arena", "Play real MOT17 clips with any tracker, or put two side by side and watch where identities break."],
            ["/benchmarks", "▦", "Benchmarks", "HOTA, IDF1, MOTA and identity switches across 7 sequences, 6 detectors and 7 tracker variants, with ablations."],
            ["/edge", "⚡", "Edge Lab", "What ONNX, INT8, CoreML, threads and input size really buy on CPU — including what did not work."],
            ["/studio", "◈", "Analytics Studio", "Draw lines and zones on tracked video: counts, dwell time, crowding, heatmaps — then just ask in English."],
            ["/live", "●", "Live in browser", "YOLO11 + ByteTrack running locally via ONNX Runtime Web. Webcam or your own video. Nothing leaves your device."],
            ["/about", "ⓘ", "How it's built", "Architecture, engineering decisions, cross-language parity tests, privacy by design, honest limitations."],
          ].map(([to, ic, title, text]) => (
            <Link key={to} to={to} className="card feature" style={{ color: "inherit", textDecoration: "none" }}>
              <div className="ic">{ic}</div>
              <h3>{title}</h3>
              <p>{text}</p>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
