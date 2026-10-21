import { useMemo, useState } from "react";
import { ClipVideo, type Layers, Panel, Transport, useClipPlayer } from "../components/ClipPlayer";
import { ErrorBox, Loading, PageHead, Seg, Toggle } from "../components/ui";
import { dataUrl, useClipTracks, useManifest } from "../data/load";
import type { ClipMetrics, Manifest, SequenceMeta } from "../data/types";
import { f3 } from "../lib/format";

type Mode = "single" | "compare";

function Metrics({ manifest, metrics, selected }: { manifest: Manifest; metrics: Record<string, ClipMetrics>; selected: string[] }) {
  const rows = manifest.trackers
    .map((t) => ({ t, m: metrics[t.id] }))
    .filter((r) => r.m)
    .sort((a, b) => b.m.hota - a.m.hota);
  const maxHota = Math.max(...rows.map((r) => r.m.hota));
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr><th>Tracker</th><th>HOTA</th><th>IDF1</th><th>MOTA</th><th>AssA</th><th>ID switches</th></tr>
        </thead>
        <tbody>
          {rows.map(({ t, m }) => (
            <tr key={t.id} style={selected.includes(t.id) ? { background: "rgba(34,211,238,0.07)" } : undefined}>
              <td><span className="row" style={{ gap: 8 }}><i className="dot" style={{ background: t.color }} />{t.label}</span></td>
              <td>{f3(m.hota)}<span className="bar" style={{ width: `${(m.hota / maxHota) * 100}%`, background: t.color }} /></td>
              <td>{f3(m.idf1)}</td>
              <td>{f3(m.mota)}</td>
              <td>{f3(m.assa)}</td>
              <td style={{ color: m.num_switches === Math.min(...rows.map((r) => r.m.num_switches)) ? "var(--lime)" : undefined }}>{m.num_switches}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ArenaStage({ seq, manifest }: { seq: SequenceMeta; manifest: Manifest }) {
  const player = useClipPlayer(seq);
  const clip = useClipTracks(seq.tracks);
  const [mode, setMode] = useState<Mode>("compare");
  const [a, setA] = useState("bytetrack");
  const [b, setB] = useState("botsort");
  const [layers, setLayers] = useState<Layers>({ trails: true, ids: true, detections: false, gt: false, switches: true });
  const set = (k: keyof Layers) => (v: boolean) => setLayers((l) => ({ ...l, [k]: v }));
  const meta = (id: string) => manifest.trackers.find((t) => t.id === id)!;
  const shown = mode === "single" ? [a] : [a, b];

  if (clip.error) return <ErrorBox message={clip.error} />;

  const panel = (id: string) => {
    const t = meta(id);
    const m = clip.data?.metrics[id];
    return (
      <Panel
        key={id}
        player={player}
        tracks={clip.data?.trackers[id] ?? null}
        detections={clip.data?.detections}
        gt={clip.data?.gt}
        switches={clip.data?.switches[id]}
        layers={layers}
        badge={<><i className="dot" style={{ background: t.color }} />{t.label}{m && <span style={{ color: "var(--red)" }}>· {m.num_switches} switches</span>}</>}
      />
    );
  };

  return (
    <div className="stack" style={{ gap: 16 }}>
      <ClipVideo player={player} />
      <div className="row spread">
        <div className="row">
          <Seg<Mode> value={mode} onChange={setMode} label="Mode" options={[{ value: "single", label: "Single" }, { value: "compare", label: "Compare" }]} />
          <Seg value={a} onChange={setA} label="Tracker A" options={manifest.trackers.map((t) => ({ value: t.id, label: mode === "compare" ? `A · ${t.label}` : t.label, dot: t.color }))} />
        </div>
      </div>
      {mode === "compare" && (
        <Seg value={b} onChange={setB} label="Tracker B" options={manifest.trackers.map((t) => ({ value: t.id, label: `B · ${t.label}`, dot: t.color }))} />
      )}

      <div className={`grid ${mode === "compare" ? "cols-2" : ""}`}>{shown.map(panel)}</div>

      <div className="stack" style={{ gap: 8 }}>
        <Transport player={player} lanes={shown.map((id) => ({ label: meta(id).label, color: meta(id).color, ticks: clip.data?.switches[id] ?? [] }))} />
        <p style={{ fontSize: "0.82rem", color: "var(--muted)" }}>Red ticks on the timeline are <b>identity switches</b>: moments where a ground-truth person was handed a different track ID.</p>
      </div>

      <div className="row" style={{ gap: 20 }}>
        <Toggle checked={layers.trails} onChange={set("trails")}>Trails</Toggle>
        <Toggle checked={layers.ids} onChange={set("ids")}>Track IDs</Toggle>
        <Toggle checked={layers.switches} onChange={set("switches")}>ID-switch markers</Toggle>
        <Toggle checked={layers.gt} onChange={set("gt")}><span style={{ color: "var(--lime)" }}>Ground truth</span></Toggle>
        <Toggle checked={layers.detections} onChange={set("detections")}>Raw detections (≥0.3)</Toggle>
      </div>

      {clip.data ? (
        <div className="stack">
          <div><h3>Metrics on this clip</h3><p className="sub" style={{ fontSize: "0.86rem" }}>Scored against MOT17 ground truth over the clip's full-rate frames. Higher HOTA/IDF1/MOTA is better; fewer switches is better.</p></div>
          <Metrics manifest={manifest} metrics={clip.data.metrics} selected={shown} />
        </div>
      ) : <Loading what="tracks" />}
    </div>
  );
}

export function Arena() {
  const manifest = useManifest();
  const [seqId, setSeqId] = useState("MOT17-04");
  const seq = useMemo(() => manifest.data?.sequences.find((s) => s.id === seqId), [manifest.data, seqId]);

  if (manifest.error) return <ErrorBox message={manifest.error} />;
  if (!manifest.data || !seq) return <Loading what="clips" />;

  return (
    <div className="stack" style={{ gap: 22 }}>
      <PageHead eyebrow="Tracker Arena" title="Same video. Same detector. Different trackers.">
        Every tracker below consumes the <b>same</b> YOLO11s detections, so any difference you see is the association logic alone. Clips are the 10-second window of each sequence where ByteTrack makes the most identity switches — the hard part, not a cherry-picked easy one.
      </PageHead>

      <div className="grid cols-4" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(150px,1fr))" }}>
        {manifest.data.sequences.map((s) => (
          <button key={s.id} onClick={() => setSeqId(s.id)} className="card" aria-pressed={s.id === seqId}
            style={{ padding: 0, overflow: "hidden", cursor: "pointer", textAlign: "left", color: "inherit", font: "inherit", borderColor: s.id === seqId ? "var(--cyan)" : undefined, boxShadow: s.id === seqId ? "0 0 0 1px var(--cyan)" : undefined }}>
            <img src={dataUrl(s.poster)} alt="" style={{ width: "100%", display: "block", aspectRatio: "16/9", objectFit: "cover" }} loading="lazy" />
            <div style={{ padding: "8px 12px" }}>
              <div style={{ fontWeight: 700, fontSize: "0.88rem" }}>{s.id.replace("MOT17-", "Scene ")}</div>
              <div style={{ fontSize: "0.74rem", color: "var(--muted)" }}>{s.camera} camera · {s.meanPeople} people/frame</div>
            </div>
          </button>
        ))}
      </div>

      <ArenaStage key={seq.id} seq={seq} manifest={manifest.data} />
      <div className="callout">
        {seq.camera === "moving"
          ? <>This is a <b>moving-camera</b> scene (≈{seq.cameraMotionPx.toFixed(1)} px of global motion per frame, measured by the CMC estimator). Compare <b>ByteTrack</b> with <b>BoT-SORT (CMC)</b>: the only difference is warping every track by the estimated camera motion before matching.</>
          : <>This is a <b>static-camera</b> scene (≈{seq.cameraMotionPx.toFixed(1)} px of global motion per frame). Camera-motion compensation has nothing to compensate here — the differences come from appearance (ReID) and matching strategy.</>}
      </div>
    </div>
  );
}
