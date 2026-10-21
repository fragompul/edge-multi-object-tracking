import { useMemo, useState } from "react";
import { BarChart } from "../charts/BarChart";
import { Heatmap } from "../charts/Heatmap";
import { Scatter } from "../charts/Scatter";
import { ErrorBox, Loading, PageHead, Seg } from "../components/ui";
import { useResults } from "../data/load";
import type { Results } from "../data/types";
import { f1, f3, groupBy, mean } from "../lib/format";
import { type Camera, type Metric, edgeFind, edgeRows, filterRows, meansByTracker } from "../lib/stats";

const ORDER = ["bytetrack", "botsort_cmc", "botsort_reid", "botsort", "deepsort", "deepsort_osnet", "deepsort_osnet_cascade"];
const SHORT: Record<string, string> = {
  bytetrack: "ByteTrack", botsort_cmc: "BoT-SORT\nCMC", botsort_reid: "BoT-SORT\nReID", botsort: "BoT-SORT\nCMC+ReID",
  deepsort: "DeepSORT\nResNet18", deepsort_osnet: "DeepSORT\nOSNet", deepsort_osnet_cascade: "DeepSORT\nOSNet+cascade",
};
type SortKey = "hota" | "deta" | "assa" | "idf1" | "mota" | "num_switches" | "tracker_ms_per_frame";
const COLS: { key: SortKey; label: string; fmt: (v: number) => string; lower?: boolean }[] = [
  { key: "hota", label: "HOTA", fmt: f3 }, { key: "deta", label: "DetA", fmt: f3 }, { key: "assa", label: "AssA", fmt: f3 },
  { key: "idf1", label: "IDF1", fmt: f3 }, { key: "mota", label: "MOTA", fmt: f3 },
  { key: "num_switches", label: "ID sw.", fmt: (v) => v.toFixed(0), lower: true },
  { key: "tracker_ms_per_frame", label: "ms/frame*", fmt: (v) => v.toFixed(2), lower: true },
];

function Leaderboard({ results, camera, backbone }: { results: Results; camera: Camera; backbone: string }) {
  const [sort, setSort] = useState<SortKey>("hota");
  const rows = useMemo(() => meansByTracker(filterRows(results, { camera, backbone })), [results, camera, backbone]);
  const col = COLS.find((c) => c.key === sort)!;
  const sorted = [...rows].sort((a, b) => (col.lower ? a[sort] - b[sort] : b[sort] - a[sort]));
  const best = (c: (typeof COLS)[number]) => (c.lower ? Math.min : Math.max)(...rows.map((r) => r[c.key]));
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Tracker</th>
            {COLS.map((c) => <th key={c.key} className={c.key === sort ? "sorted" : ""} onClick={() => setSort(c.key)}>{c.label}{c.key === sort ? (c.lower ? " ▲" : " ▼") : ""}</th>)}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.tracker}>
              <td><span className="row" style={{ gap: 8 }}><i className="dot" style={{ background: results.variants[r.tracker].color }} />{results.variants[r.tracker].label}</span></td>
              {COLS.map((c) => (
                <td key={c.key} style={r[c.key] === best(c) ? { color: "var(--lime)", fontWeight: 700 } : undefined}>
                  {c.fmt(r[c.key])}
                  {c.key === "hota" && <span className="bar" style={{ width: `${(r.hota / Math.max(...rows.map((x) => x.hota))) * 100}%`, background: results.variants[r.tracker].color }} />}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Benchmarks() {
  const { data, error } = useResults();
  const [camera, setCamera] = useState<Camera>("all");
  const [backbone, setBackbone] = useState("all");
  const [metric, setMetric] = useState<Metric>("hota");

  const backbones = useMemo(() => (data ? [...new Set(data.tracking.map((r) => r.backbone))].sort() : []), [data]);
  const heat = useMemo(() => {
    if (!data) return null;
    const rows = filterRows(data, { backbone });
    const seqs = data.sequences.map((s) => s.sequence);
    return {
      seqs,
      values: ORDER.map((t) => seqs.map((s) => {
        const v = rows.filter((r) => r.tracker === t && r.sequence === s).map((r) => r[metric]);
        return v.length ? mean(v) : null;
      })),
    };
  }, [data, backbone, metric]);

  if (error) return <ErrorBox message={error} />;
  if (!data || !heat) return <Loading what="benchmark results" />;

  const static_ = meansByTracker(filterRows(data, { camera: "static", backbone }));
  const moving = meansByTracker(filterRows(data, { camera: "moving", backbone }));
  const pick = (rows: ReturnType<typeof meansByTracker>, t: string, k: SortKey) => rows.find((r) => r.tracker === t)?.[k] ?? 0;
  const nStatic = data.sequences.filter((s) => s.camera === "static").length;
  const nMoving = data.sequences.length - nStatic;

  const ablationSteps = ["bytetrack", "botsort_cmc", "botsort_reid", "botsort"];
  const ablation = (["static", "moving"] as const).map((cam) => {
    const rows = cam === "static" ? static_ : moving;
    return {
      label: cam === "static" ? "Static camera" : "Moving camera",
      sub: `${cam === "static" ? nStatic : nMoving} sequences`,
      bars: ablationSteps.map((t, i) => ({ series: SHORT[t].replace("\n", " "), value: pick(rows, t, "hota"), color: ["#22D3EE", "#A78BFA", "#FB923C", "#F472B6"][i] })),
    };
  });
  const switchesAbl = (["static", "moving"] as const).map((cam) => {
    const rows = cam === "static" ? static_ : moving;
    return {
      label: cam === "static" ? "Static camera" : "Moving camera",
      bars: ablationSteps.map((t, i) => ({ series: SHORT[t].replace("\n", " "), value: pick(rows, t, "num_switches"), color: ["#22D3EE", "#A78BFA", "#FB923C", "#F472B6"][i] })),
    };
  });
  const reidSteps = ["deepsort", "deepsort_osnet", "deepsort_osnet_cascade"];
  const all = meansByTracker(filterRows(data, { camera, backbone }));
  const reid = ["hota", "assa"].map((k) => ({
    label: k === "hota" ? "HOTA" : "AssA (association)",
    bars: reidSteps.map((t, i) => ({ series: SHORT[t].replace("\n", " "), value: pick(all, t, k as SortKey), color: ["#FBBF24", "#A3E635", "#34D399"][i] })),
  }));

  const edgeMain = edgeRows(data, "main");
  const det = [...groupBy(data.detection, (d) => d.backbone)].map(([bb, rs]) => ({
    bb, ap50: mean(rs.map((r) => r.ap50)), ap: mean(rs.map((r) => r.ap)),
    hota: mean(filterRows(data, { backbone: bb }).filter((r) => r.tracker === "bytetrack").map((r) => r.hota)),
    lat: edgeFind(edgeMain, bb, "onnx_fp32")?.mean_ms,
  })).sort((a, b) => b.ap50 - a.ap50);

  return (
    <div className="stack" style={{ gap: 28 }}>
      <PageHead eyebrow="Benchmarks" title={<>{data.tracking.length} tracking runs, one protocol</>}>
        {data.sequences.length} MOT17 sequences × {backbones.length} YOLO backbones × {new Set(data.tracking.map((r) => r.tracker)).size} tracker variants, scored with HOTA, IDF1, MOTA and identity switches. All trackers share cached detections, so the comparison isolates association.
      </PageHead>

      <div className="row spread">
        <div className="row">
          <Seg<Camera> value={camera} onChange={setCamera} label="Camera" options={[{ value: "all", label: "All scenes" }, { value: "static", label: `Static (${nStatic})` }, { value: "moving", label: `Moving (${nMoving})` }]} />
          <select value={backbone} onChange={(e) => setBackbone(e.target.value)} aria-label="Backbone">
            <option value="all">All backbones (mean)</option>
            {backbones.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </div>
        <span className="pill">click a column to sort</span>
      </div>

      <section className="stack">
        <h3>Leaderboard</h3>
        <Leaderboard results={data} camera={camera} backbone={backbone} />
        <p className="sub" style={{ fontSize: "0.8rem" }}>*Tracker-only cost per frame on cached inputs (association logic); ReID embedding and optical-flow costs are separate — see Edge Lab.</p>
      </section>

      <section className="card stack">
        <div className="row spread">
          <div><h3>Tracker × scene</h3><p className="sub">Mean over {backbone === "all" ? "all backbones" : backbone}. Column label: camera type.</p></div>
          <Seg<Metric> value={metric} onChange={setMetric} options={[{ value: "hota", label: "HOTA" }, { value: "idf1", label: "IDF1" }, { value: "mota", label: "MOTA" }, { value: "assa", label: "AssA" }, { value: "num_switches", label: "ID switches" }]} />
        </div>
        <Heatmap
          rows={ORDER.map((t) => data.variants[t].label)}
          cols={heat.seqs.map((s) => s.replace("MOT17-", ""))}
          colSub={data.sequences.map((s) => s.camera)}
          values={heat.values}
          higherBetter={metric !== "num_switches"}
          format={metric === "num_switches" ? (v) => v.toFixed(0) : f3}
        />
      </section>

      <section className="grid cols-2">
        <div className="card stack">
          <div><h3>Ablation · what camera-motion compensation and ReID each buy</h3><p className="sub">HOTA. CMC only matters when the camera moves; ReID helps most when identities are ambiguous.</p></div>
          <BarChart groups={ablation} height={290} yMax={0.6} legend={ablationSteps.map((t, i) => ({ name: SHORT[t].replace("\n", " "), color: ["#22D3EE", "#A78BFA", "#FB923C", "#F472B6"][i] }))} />
        </div>
        <div className="card stack">
          <div><h3>…and the identity switches behind it</h3><p className="sub">Mean per sequence. Lower is better.</p></div>
          <BarChart groups={switchesAbl} height={290} format={(v) => v.toFixed(0)} labels />
        </div>
      </section>

      <section className="grid cols-2">
        <div className="card stack">
          <div><h3>Closing DeepSORT's ReID gap</h3><p className="sub">Same tracker, three upgrades: a generic ImageNet ResNet18 → a real person-ReID network (OSNet, trained on MSMT17) → the paper's matching cascade.</p></div>
          <BarChart groups={reid} height={290} yMax={0.5} legend={reidSteps.map((t, i) => ({ name: SHORT[t].replace("\n", " "), color: ["#FBBF24", "#A3E635", "#34D399"][i] }))} />
        </div>
        <div className="card stack">
          <div><h3>Detector backbones</h3><p className="sub">Detection AP50 vs measured CPU latency (ONNX FP32, 384×640). Dashed line = Pareto frontier.</p></div>
          <Scatter
            xLabel="CPU latency per frame (ms)" yLabel="detection AP50" xFormat={(v) => v.toFixed(0)} pareto
            points={det.filter((d) => d.lat).map((d) => ({
              x: d.lat!, y: d.ap50, label: d.bb, color: d.bb.startsWith("yolo11") ? "#22D3EE" : "#F472B6",
              tip: <><b>{d.bb}</b><br />AP50 {f3(d.ap50)} · AP {f3(d.ap)}<br />ByteTrack HOTA {f3(d.hota)}<br />{f1(d.lat!)} ms</>,
            }))}
          />
        </div>
      </section>

      <div className="callout warn">
        <b>Read this before quoting a number.</b> Off-the-shelf COCO YOLO (not fine-tuned) at a 640 px input on 1080p footage, no distractor-class filtering, trackers tuned on one sequence ({String(data.tuning.tuned_on_sequence)}, last 25% of frames, objective {String(data.tuning.objective).split(" on ")[0]}). These are comparable <i>to each other</i>, not to MOT17 leaderboard scores.
      </div>
    </div>
  );
}
