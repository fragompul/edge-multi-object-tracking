import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ByteTracker, type ByteTrackConfig, DEFAULT_BYTETRACK } from "../core/bytetrack";
import { letterboxGeometry, postprocess } from "../core/yolo";
import { PageHead, Seg } from "../components/ui";
import { assetUrl, dataUrl, useResults } from "../data/load";
import { idColor } from "../lib/draw";

type Source = "sample" | "webcam" | "file";
type Provider = "wasm" | "webgpu";
type Ort = typeof import("onnxruntime-web");
type Session = import("onnxruntime-web").InferenceSession;

const MODELS = {
  fp32: { label: "YOLO11n · FP32 (10.7 MB)", file: "models/yolo11n-384x640-fp32.onnx" },
  int8: { label: "YOLO11n · INT8 (3.2 MB)", file: "models/yolo11n-384x640-int8.onnx" },
} as const;
const IN_H = 384;
const IN_W = 640;
const base = import.meta.env.BASE_URL;

interface Stats {
  fps: number;
  detectMs: number;
  preMs: number;
  trackMs: number;
  people: number;
  uniqueIds: number;
  frames: number;
}

function tunedConfig(t: Record<string, number> | undefined): ByteTrackConfig {
  if (!t) return DEFAULT_BYTETRACK;
  return {
    highConfThreshold: t.high_conf_threshold,
    lowConfThreshold: t.low_conf_threshold,
    matchIouThreshold: t.match_iou_threshold,
    lowMatchIouThreshold: t.low_match_iou_threshold,
    minHits: t.min_hits,
    maxAge: t.max_age,
  };
}

export function Live() {
  const results = useResults();
  const cfg = useRef<ByteTrackConfig>(DEFAULT_BYTETRACK);
  cfg.current = tunedConfig(results.data?.tuning.bytetrack as Record<string, number> | undefined);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const work = useRef<HTMLCanvasElement | null>(null);
  const session = useRef<Session | null>(null);
  const ortRef = useRef<Ort | null>(null);
  const running = useRef(false);
  const tracker = useRef(new ByteTracker(DEFAULT_BYTETRACK));
  const seen = useRef(new Set<number>());
  const trails = useRef(new Map<number, [number, number][]>());

  const [source, setSource] = useState<Source>("sample");
  const [provider, setProvider] = useState<Provider>("wasm");
  const [model, setModel] = useState<keyof typeof MODELS>("fp32");
  const [status, setStatus] = useState<"idle" | "loading" | "running" | "error">("idle");
  const [message, setMessage] = useState("");
  const [stats, setStats] = useState<Stats | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const hasWebGpu = typeof navigator !== "undefined" && "gpu" in navigator;

  const stop = useCallback(() => {
    running.current = false;
    const v = videoRef.current;
    if (v) {
      v.pause();
      const s = v.srcObject as MediaStream | null;
      s?.getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }
    setStatus("idle");
  }, []);
  useEffect(() => stop, [stop]);

  const loadSession = useCallback(async () => {
    const ort = ortRef.current ?? (await import("onnxruntime-web/webgpu"));
    ortRef.current = ort;
    ort.env.wasm.wasmPaths = `${base}ort/`;
    // Multi-threaded WASM needs cross-origin isolation, which static hosts rarely provide.
    ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    session.current = await ort.InferenceSession.create(assetUrl(MODELS[model].file), {
      executionProviders: [provider],
      graphOptimizationLevel: "all",
    });
  }, [model, provider]);

  const start = useCallback(async () => {
    try {
      setStatus("loading");
      setMessage("Loading the model…");
      session.current = null;
      await loadSession();
      const v = videoRef.current!;
      if (source === "webcam") {
        v.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      } else {
        v.srcObject = null;
        v.src = source === "file" && fileUrl ? fileUrl : dataUrl("clips/MOT17-09.mp4");
        v.loop = true;
      }
      await v.play();
      tracker.current = new ByteTracker(cfg.current);
      seen.current = new Set();
      trails.current = new Map();
      setStatus("running");
      setMessage("");
      running.current = true;
      void loop();
    } catch (e) {
      setStatus("error");
      setMessage((e as Error).message);
      running.current = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, fileUrl, loadSession]);

  const loop = useCallback(async () => {
    const v = videoRef.current!;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const ort = ortRef.current!;
    if (!work.current) {
      work.current = document.createElement("canvas");
      work.current.width = IN_W;
      work.current.height = IN_H;
    }
    const wctx = work.current.getContext("2d", { willReadFrequently: true })!;
    const input = new Float32Array(3 * IN_H * IN_W);
    let ema = { fps: 0, det: 0, pre: 0, trk: 0 };
    let last = performance.now();
    let lastTime = -1;
    let frames = 0;

    while (running.current) {
      if (v.readyState < 2 || v.videoWidth === 0) { await new Promise((r) => requestAnimationFrame(r)); continue; }
      const t0 = performance.now();
      const w = v.videoWidth, h = v.videoHeight;
      if (canvas.width !== w) { canvas.width = w; canvas.height = h; }
      if (source !== "webcam" && v.currentTime < lastTime) { tracker.current = new ByteTracker(cfg.current); trails.current = new Map(); }
      lastTime = v.currentTime;

      // preprocess: letterbox into 640x384, RGB, CHW, /255
      const lb = letterboxGeometry(w, h, IN_H, IN_W);
      wctx.fillStyle = "rgb(114,114,114)";
      wctx.fillRect(0, 0, IN_W, IN_H);
      wctx.drawImage(v, lb.padX, lb.padY, lb.newW, lb.newH);
      const px = wctx.getImageData(0, 0, IN_W, IN_H).data;
      const plane = IN_H * IN_W;
      for (let i = 0, p = 0; i < plane; i++, p += 4) {
        input[i] = px[p] / 255;
        input[plane + i] = px[p + 1] / 255;
        input[2 * plane + i] = px[p + 2] / 255;
      }
      const t1 = performance.now();
      const sess = session.current!;
      const out = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, IN_H, IN_W]) });
      const o = out[sess.outputNames[0]];
      const t2 = performance.now();
      const dets = postprocess(o.data as Float32Array, o.dims[2], lb, w, h, 0.1, 0.7);
      const tracks = tracker.current.update(dets);
      const t3 = performance.now();

      // draw
      ctx.drawImage(v, 0, 0, w, h);
      const lw = Math.max(2, w / 640);
      ctx.lineWidth = lw;
      for (const t of tracks) {
        seen.current.add(t.id);
        const pts = trails.current.get(t.id) ?? [];
        pts.push([(t.box[0] + t.box[2]) / 2, t.box[3]]);
        if (pts.length > 40) pts.shift();
        trails.current.set(t.id, pts);
      }
      const alive = new Set(tracks.map((t) => t.id));
      for (const [id, pts] of trails.current) {
        if (!alive.has(id)) { trails.current.delete(id); continue; }
        ctx.strokeStyle = idColor(id);
        for (let i = 1; i < pts.length; i++) {
          ctx.globalAlpha = 0.2 + 0.8 * (i / pts.length);
          ctx.beginPath(); ctx.moveTo(pts[i - 1][0], pts[i - 1][1]); ctx.lineTo(pts[i][0], pts[i][1]); ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      for (const t of tracks) {
        const [x1, y1, x2, y2] = t.box;
        const c = idColor(t.id);
        ctx.strokeStyle = c; ctx.fillStyle = c;
        ctx.globalAlpha = 0.1; ctx.fillRect(x1, y1, x2 - x1, y2 - y1); ctx.globalAlpha = 1;
        const arm = Math.max(8, Math.min(x2 - x1, y2 - y1) * 0.22);
        ctx.beginPath();
        for (const [cx, cy, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]] as const) {
          ctx.moveTo(cx + dx * arm, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * arm);
        }
        ctx.stroke();
        const fs = Math.max(11, w / 90);
        ctx.font = `700 ${fs}px "Inter Variable", system-ui, sans-serif`;
        const tw = ctx.measureText(String(t.id)).width + 8;
        ctx.fillRect(x1, y1 - fs - 5, tw, fs + 5);
        ctx.fillStyle = "#0b1020"; ctx.fillText(String(t.id), x1 + 4, y1 - 5);
      }

      frames++;
      const now = performance.now();
      const a = frames === 1 ? 1 : 0.08;
      ema = {
        fps: ema.fps + a * (1000 / Math.max(1, now - last) - ema.fps),
        det: ema.det + a * ((t2 - t1) - ema.det),
        pre: ema.pre + a * ((t1 - t0) - ema.pre),
        trk: ema.trk + a * ((t3 - t2) - ema.trk),
      };
      last = now;
      if (frames % 5 === 0)
        setStats({ fps: ema.fps, detectMs: ema.det, preMs: ema.pre, trackMs: ema.trk, people: tracks.length, uniqueIds: seen.current.size, frames });
      await new Promise((r) => (typeof v.requestVideoFrameCallback === "function" && source !== "webcam" ? setTimeout(r, 0) : requestAnimationFrame(r)));
    }
  }, [source]);

  const onFile = (f: File | undefined) => {
    if (!f) return;
    setFileUrl(URL.createObjectURL(f));
    setSource("file");
  };

  return (
    <div className="stack" style={{ gap: 22 }}>
      <PageHead eyebrow="Live in browser" title="The edge is your browser tab">
        YOLO11n and a TypeScript port of ByteTrack run entirely on your device through ONNX Runtime Web — WebGPU where available, WebAssembly otherwise. No frame is uploaded anywhere. Both the pre/post-processing and the tracker are verified against the Python reference by parity tests.
      </PageHead>

      <div className="grid split-live">
        <div className="stack">
          <div className="stage" style={{ minHeight: 240 }}>
            <canvas ref={canvasRef} className="under" width={960} height={540} />
            <video ref={videoRef} muted playsInline style={{ position: "absolute", width: 2, height: 2, opacity: 0 }} />
            <div className="badge"><i className="dot" style={{ background: status === "running" ? "var(--lime)" : "var(--muted)" }} />{status === "running" ? `${provider.toUpperCase()} · ${stats ? stats.fps.toFixed(1) : "…"} FPS` : status === "loading" ? "loading…" : "ready"}</div>
            {status !== "running" && (
              <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", background: "rgba(11,16,32,0.55)", lineHeight: 1.4 }}>
                {status === "loading" ? <span className="row"><span className="spinner" /> {message}</span> : <button className="btn primary" onClick={() => void start()}>▶ Start tracking</button>}
              </div>
            )}
          </div>
          {status === "error" && <div className="callout warn">{message}</div>}
          <div className="row spread">
            <div className="row">
              <Seg<Source> value={source} onChange={(v) => { stop(); setSource(v); }} label="Source" options={[{ value: "sample", label: "Sample clip" }, { value: "webcam", label: "Webcam" }, { value: "file", label: "My video" }]} />
              {source === "file" && <input type="file" accept="video/*" onChange={(e) => onFile(e.target.files?.[0])} aria-label="Choose a video" />}
            </div>
            {status === "running" && <button className="btn" onClick={stop}>■ Stop</button>}
          </div>
          <div className="row">
            <Seg<Provider> value={provider} onChange={(v) => { stop(); setProvider(v); }} label="Backend" options={[{ value: "wasm", label: "WebAssembly (CPU)" }, { value: "webgpu", label: hasWebGpu ? "WebGPU" : "WebGPU (unavailable)" }]} />
            <Seg<keyof typeof MODELS> value={model} onChange={(v) => { stop(); setModel(v); }} label="Model" options={(Object.keys(MODELS) as (keyof typeof MODELS)[]).map((k) => ({ value: k, label: MODELS[k].label }))} />
          </div>
          {provider === "webgpu" && !hasWebGpu && <div className="callout warn">This browser does not expose WebGPU; use the WebAssembly backend (try a recent Chrome or Edge for WebGPU).</div>}
        </div>

        <div className="stack">
          <div className="card stack">
            <h3>Live telemetry</h3>
            <div className="grid cols-2" style={{ gap: 10 }}>
              <div className="stat"><span className="v" style={{ color: "var(--cyan)" }}>{stats ? stats.fps.toFixed(1) : "—"}</span><span className="l">frames / second</span></div>
              <div className="stat"><span className="v">{stats ? stats.people : "—"}</span><span className="l">people tracked now</span></div>
              <div className="stat"><span className="v">{stats ? `${stats.detectMs.toFixed(0)}ms` : "—"}</span><span className="l">YOLO inference</span></div>
              <div className="stat"><span className="v">{stats ? `${(stats.preMs + stats.trackMs).toFixed(0)}ms` : "—"}</span><span className="l">pre/post + tracking</span></div>
            </div>
            <div className="pill" style={{ alignSelf: "flex-start" }}>{stats ? `${stats.uniqueIds} distinct IDs · ${stats.frames} frames` : "waiting for first frame"}</div>
          </div>
          <div className="card stack">
            <h3>Why this matters</h3>
            <p style={{ fontSize: "0.9rem" }}><b>Privacy by construction:</b> video never leaves the device, there is no server to breach, and tracker IDs are anonymous and ephemeral. <b>Zero-ops:</b> this page is static files on GitHub Pages.</p>
            <p style={{ fontSize: "0.9rem" }}>On an Apple M5 laptop this runs at roughly 10 FPS on single-threaded WebAssembly (static hosting cannot enable the cross-origin isolation multi-threading needs) and about 55-75 FPS on WebGPU. Your numbers depend on your hardware; the same model needs ~11 ms natively on CPU — see <Link to="/edge">Edge Lab</Link>.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
