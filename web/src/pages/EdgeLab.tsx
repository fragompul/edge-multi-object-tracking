import { useMemo, useState } from "react";
import { BarChart } from "../charts/BarChart";
import { LineChart } from "../charts/LineChart";
import { Scatter } from "../charts/Scatter";
import { ErrorBox, Loading, PageHead, Seg } from "../components/ui";
import { useResults } from "../data/load";
import type { EdgeRow } from "../data/types";
import { f1, f3, mean } from "../lib/format";
import { edgeFind, edgeRows } from "../lib/stats";

const VARIANTS: { id: string; label: string; color: string; note: string }[] = [
  { id: "torch_cpu", label: "PyTorch · CPU", color: "#8B93B0", note: "Ultralytics, FP32, rectangular input" },
  { id: "torch_mps", label: "PyTorch · Apple GPU", color: "#FBBF24", note: "MPS backend" },
  { id: "onnx_fp32", label: "ONNX Runtime · CPU", color: "#22D3EE", note: "FP32, 384×640 static shape" },
  { id: "onnx_int8_dynamic", label: "ONNX · INT8 dynamic", color: "#A78BFA", note: "weights INT8, activations quantised on the fly" },
  { id: "onnx_int8_static", label: "ONNX · INT8 static", color: "#F472B6", note: "QDQ, calibrated on 66 frames" },
  { id: "onnx_int8_static_fp32head", label: "ONNX · INT8 static, FP32 head", color: "#FB923C", note: "detection head kept in float" },
  { id: "onnx_coreml", label: "ONNX · CoreML (ANE/GPU)", color: "#A3E635", note: "CoreML execution provider" },
];
const vInfo = (id: string) => VARIANTS.find((v) => v.id === id)!;

export function EdgeLab() {
  const { data, error } = useResults();
  const [backbone, setBackbone] = useState("yolo11s");
  const [resMetric, setResMetric] = useState<"ap50" | "hota">("hota");
  const main = useMemo(() => (data ? edgeRows(data, "main") : []), [data]);
  const threads = useMemo(() => (data ? edgeRows(data, "threads") : []), [data]);
  const res = useMemo(() => (data ? edgeRows(data, "resolution") : []), [data]);

  if (error) return <ErrorBox message={error} />;
  if (!data) return <Loading what="edge benchmark" />;
  if (!main.length) return <div className="callout warn">The edge benchmark has not been run yet (<code>make edge</code>).</div>;

  const machine = data.edge.machine as Record<string, string | number>;
  const backbones = [...new Set(main.map((r) => r.backbone))];
  const rowsFor = (bb: string) => VARIANTS.map((v) => edgeFind(main, bb, v.id)).filter((r): r is EdgeRow => !!r);
  const table = rowsFor(backbone);
  const ref = edgeFind(main, backbone, "torch_cpu");
  const fp32 = edgeFind(main, backbone, "onnx_fp32");

  const latencyGroups = backbones.map((bb) => ({
    label: bb,
    bars: VARIANTS.filter((v) => !["onnx_int8_static_fp32head"].includes(v.id)).map((v) => ({ series: v.label, value: edgeFind(main, bb, v.id)?.mean_ms ?? 0, color: v.color })),
  }));

  const threadSeries = (["yolo11n", "yolo11s"] as const).flatMap((bb) =>
    (["onnx_fp32", "onnx_int8_static"] as const).map((v) => ({
      name: `${bb} · ${v === "onnx_fp32" ? "FP32" : "INT8"}`,
      color: v === "onnx_fp32" ? (bb === "yolo11n" ? "#22D3EE" : "#A78BFA") : (bb === "yolo11n" ? "#F472B6" : "#FB923C"),
      points: threads.filter((r) => r.backbone === bb && r.variant === v).sort((a, b) => a.threads! - b.threads!).map((r) => [r.threads!, r.mean_ms] as [number, number]),
    })).filter((s) => s.points.length),
  );
  const sweep = data.resolutionSweep ?? [];
  const resSeries = ["yolo11n", "yolo11s", "yolo11m"].map((bb, i) => ({
    name: bb, color: ["#22D3EE", "#A78BFA", "#F472B6"][i],
    rows: res.filter((r) => r.backbone === bb).sort((a, b) => a.long_side! - b.long_side!).map((r) => ({
      ...r,
      hota: sweep.find((w) => w.backbone === bb && w.imgsz === r.long_side && w.tracker === "botsort_cmc")?.hota,
    })),
  })).filter((s) => s.rows.length);
  const hasHota = resSeries.some((s) => s.rows.some((r) => r.hota !== undefined));
  const metricOf = (r: { ap50: number; hota?: number }) => (resMetric === "hota" && hasHota ? r.hota : r.ap50);
  const resLabel = resMetric === "hota" && hasHota ? "HOTA (BoT-SORT + CMC)" : "detection AP50";

  const speed = (v: string) => mean(backbones.map((bb) => { const a = edgeFind(main, bb, "torch_cpu"), b = edgeFind(main, bb, v); return a && b ? a.mean_ms / b.mean_ms : 0; }).filter(Boolean));
  const i8Size = mean(backbones.map((bb) => (edgeFind(main, bb, "onnx_fp32")?.size_mb ?? 0) / (edgeFind(main, bb, "onnx_int8_static")?.size_mb ?? 1)));
  const i8Ap = mean(backbones.map((bb) => ((edgeFind(main, bb, "onnx_int8_static")?.ap50 ?? 0) - (edgeFind(main, bb, "onnx_fp32")?.ap50 ?? 0)) * 100));
  const i8Lat = mean(backbones.map((bb) => (edgeFind(main, bb, "onnx_fp32")?.mean_ms ?? 0) / (edgeFind(main, bb, "onnx_int8_static")?.mean_ms ?? 1)));
  const t1 = (bb: string, v: string) => threads.find((r) => r.backbone === bb && r.variant === v && r.threads === 1)?.mean_ms;
  const oneThread = t1("yolo11s", "onnx_int8_static") && t1("yolo11s", "onnx_fp32") ? t1("yolo11s", "onnx_int8_static")! / t1("yolo11s", "onnx_fp32")! : null;

  return (
    <div className="stack" style={{ gap: 28 }}>
      <PageHead eyebrow="Edge Lab" title="What running on the edge actually costs">
        Every row is a separate process measured on the same 150 MOT17 frames, scored with COCO-style AP against ground truth — latency, memory <b>and</b> accuracy, not a proxy. Measured on <b>{String(machine.cpu)}</b> ({String(machine.cores)} cores), ONNX Runtime {String(machine.onnxruntime)}.
      </PageHead>

      <div className="callout">
        <b>A correction worth showing.</b> The first version of this project concluded that ONNX was <i>slower</i> than PyTorch and INT8 gave no speed-up. Re-measuring found part of that was a methodology error: the PyTorch path ran a rectangular 384×640 input while the ONNX export used a padded 640×640 square — <b>1.67× more pixels</b> for the same image. With matched shapes, ONNX Runtime on CPU is {speed("onnx_fp32").toFixed(2)}× faster than PyTorch on average (all six backbones).
      </div>

      <section className="card stack">
        <div><h3>Latency by runtime</h3><p className="sub">Mean milliseconds per frame (image in → boxes out, including pre/post-processing). Lower is better.</p></div>
        <BarChart groups={latencyGroups} height={320} format={(v) => v.toFixed(1)} labels={false} legend={VARIANTS.filter((v) => v.id !== "onnx_int8_static_fp32head").map((v) => ({ name: v.label, color: v.color }))} yLabel="ms / frame" />
      </section>

      <section className="stack">
        <div className="row spread">
          <h3>The full picture for one backbone</h3>
          <Seg value={backbone} onChange={setBackbone} options={backbones.map((b) => ({ value: b, label: b }))} />
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Runtime</th><th>Size</th><th>Mean ms</th><th>p95 ms</th><th>FPS</th><th>Speed-up</th><th>Peak RAM</th><th>AP50</th><th>Δ AP50</th></tr></thead>
            <tbody>
              {table.map((r) => (
                <tr key={r.variant}>
                  <td><span className="row" style={{ gap: 8 }}><i className="dot" style={{ background: vInfo(r.variant).color }} /><span>{vInfo(r.variant).label}<br /><span className="sub" style={{ fontSize: "0.72rem", color: "var(--muted)" }}>{vInfo(r.variant).note}</span></span></span></td>
                  <td>{f1(r.size_mb)} MB</td><td>{f1(r.mean_ms)}</td><td>{f1(r.p95_ms)}</td><td>{r.fps.toFixed(0)}</td>
                  <td style={{ color: ref && r.mean_ms < ref.mean_ms ? "var(--lime)" : "var(--red)" }}>{ref ? `${(ref.mean_ms / r.mean_ms).toFixed(2)}×` : "—"}</td>
                  <td>{r.peak_rss_mb.toFixed(0)} MB</td><td>{f3(r.ap50)}</td>
                  <td style={{ color: fp32 && Math.abs(r.ap50 - fp32.ap50) > 0.01 ? "var(--amber)" : "var(--muted)" }}>{fp32 ? (r.ap50 - fp32.ap50 >= 0 ? "+" : "") + ((r.ap50 - fp32.ap50) * 100).toFixed(1) + " pt" : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="sub" style={{ fontSize: "0.82rem" }}>Speed-up is relative to PyTorch on CPU. Δ AP50 is relative to ONNX FP32 on the same 150 frames. Peak RAM is the worker process's maximum resident set (it includes the Python runtime and image buffers).</p>
      </section>

      <section className="grid cols-2">
        <div className="card stack">
          <div><h3>Threads: a constrained-CPU proxy</h3><p className="sub">Single-thread latency approximates a small embedded CPU far better than "all cores" does. Not a substitute for measuring the real device.</p></div>
          {threadSeries.length ? <LineChart series={threadSeries} xLabel="intra-op threads" yLabel="ms / frame" xTicks={[...new Set(threads.map((t) => t.threads!))].sort((a, b) => a - b)} yFormat={(v) => v.toFixed(0)} /> : <p className="sub">not measured</p>}
        </div>
        <div className="card stack">
          <div className="row spread"><div><h3>Input size is the biggest lever</h3><p className="sub">{resLabel} vs latency across input long side 320 → 1280 (CPU, ONNX FP32). Labels = pixels on the long side. Dashed line = Pareto frontier.</p></div>
            {hasHota && <Seg value={resMetric} onChange={setResMetric} options={[{ value: "hota", label: "HOTA" }, { value: "ap50", label: "AP50" }]} />}</div>
          {resSeries.length ? (
            <Scatter
              key={resMetric}
              xLabel="CPU latency per frame (ms, log scale)" yLabel={resLabel} xFormat={(v) => v.toFixed(0)} yFormat={(v) => v.toFixed(2)} pareto logX
              lines={resSeries.map((s) => ({ name: s.name, color: s.color, pts: s.rows.filter((r) => metricOf(r) !== undefined).map((r) => [r.mean_ms, metricOf(r)!] as [number, number]) }))}
              points={resSeries.flatMap((s) => s.rows.filter((r) => metricOf(r) !== undefined).map((r) => ({ x: r.mean_ms, y: metricOf(r)!, label: String(r.long_side), color: s.color, r: 5, tip: <><b>{s.name} @ {r.long_side}px</b><br />{resLabel} {f3(metricOf(r)!)} · {f1(r.mean_ms)} ms</> })))}
            />
          ) : <p className="sub">not measured</p>}
        </div>
      </section>

      <section className="card stack">
        <div><h3>Lessons, honestly</h3></div>
        <ul className="stack" style={{ paddingLeft: 18, margin: 0, color: "var(--ink-dim)" }}>
          <li><b>Match the export's shape to your camera.</b> Padding 16:9 video into a square wastes ~40% of the compute on grey bars.</li>
          <li><b>Use the accelerator you have.</b> The CoreML provider (ANE/GPU) is {speed("onnx_coreml").toFixed(1)}× faster than PyTorch on CPU on average, at the same accuracy -- the largest single win on this machine.</li>
          <li><b>INT8 is mainly a size and memory win here.</b> Static quantisation makes weights {i8Size.toFixed(1)}× smaller for a {i8Ap >= 0 ? "+" : ""}{i8Ap.toFixed(1)} pt mean AP50 change, but latency is only {i8Lat.toFixed(2)}× of FP32 with all cores{oneThread ? ` -- and ${oneThread.toFixed(1)}× slower on a single thread (yolo11s)` : ""}. Dynamic INT8 is slower than FP32. Measure on your target: INT8 behaves very differently across CPUs.</li>
          <li><b>Quantisation needs care in the head.</b> The first static INT8 export scored AP 0.0: the decode graph mixes pixel coordinates (up to ~640) and probabilities (0-1) in one tensor, so a single 8-bit scale flattened every score. Keeping the decode nodes in float (and MinMax calibration) fixed it.</li>
          <li><b>Pixels beat parameters.</b> MOT17's pedestrians are small: a small model at a higher input size reaches the quality of a bigger model at the default size, at a fraction of the latency (see the HOTA view).</li>
        </ul>
      </section>
    </div>
  );
}
