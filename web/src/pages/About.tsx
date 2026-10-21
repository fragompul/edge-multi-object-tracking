import { Link } from "react-router-dom";
import { REPO } from "../components/Layout";
import { PageHead } from "../components/ui";

const STAGES = [
  ["MOT17 video", "7 sequences · static + moving cameras", "#8B93B0"],
  ["YOLO detector", "6 backbones · v8 & 11 · cached", "#22D3EE"],
  ["Tracker", "ByteTrack · BoT-SORT · DeepSORT", "#A78BFA"],
  ["Evaluation", "HOTA · IDF1 · MOTA · AP", "#F472B6"],
  ["Edge export", "ONNX · INT8 · CoreML", "#FBBF24"],
  ["Apps", "CLI · dashboard · this site", "#A3E635"],
];

export function About() {
  return (
    <div className="stack" style={{ gap: 28 }}>
      <PageHead eyebrow="How it's built" title="From pixels to identities — and what it takes to ship them">
        EMOT is a research-grade tracking stack with product-grade edges: a from-scratch implementation, a reproducible benchmark, a deployment study and three front-ends.
      </PageHead>

      <section className="card">
        <div className="row" style={{ gap: 0, flexWrap: "wrap", justifyContent: "space-between" }}>
          {STAGES.map(([t, s, c], i) => (
            <div key={t} className="row" style={{ gap: 10, flexWrap: "nowrap" }}>
              <div style={{ padding: "10px 14px", borderRadius: 12, border: `1px solid ${c}55`, background: `${c}14` }}>
                <div style={{ fontWeight: 700, color: c }}>{t}</div>
                <div style={{ fontSize: "0.76rem", color: "var(--muted)" }}>{s}</div>
              </div>
              {i < STAGES.length - 1 && <span style={{ color: "var(--muted)" }}>→</span>}
            </div>
          ))}
        </div>
      </section>

      <section className="grid cols-2">
        <div className="card stack">
          <h3>Built from the primitives</h3>
          <p>Kalman filter, IoU/Mahalanobis association, Hungarian assignment, matching cascade, camera-motion compensation (Lucas-Kanade + RANSAC), OSNet (written from the paper, loading the authors' MSMT17 weights with <code>strict=True</code> and verified against an independent ONNX export to cosine 1.0) and HOTA (validated against the reference TrackEval to 10 decimals). No tracking library is wrapped.</p>
        </div>
        <div className="card stack">
          <h3>Verified, not assumed</h3>
          <p>The browser runs TypeScript ports of the tracker, analytics and YOLO post-processing. Fixtures recorded from the Python implementation feed <b>cross-language parity tests</b> in CI — and writing them caught a real NMS bug in the Python reference (OpenCV's <code>top_k</code> truncates candidates <i>before</i> suppression).</p>
        </div>
        <div className="card stack">
          <h3>Honest by default</h3>
          <p>Each result states its protocol and limits: off-the-shelf detector, single-sequence tuning, distractors not filtered, a measured edge benchmark on one machine. A methodology error in the first edge benchmark was found, fixed and documented rather than buried — see Edge Lab.</p>
        </div>
        <div className="card stack">
          <h3>Privacy by design</h3>
          <p>Tracking is within a session: IDs are anonymous and ephemeral, appearance embeddings live in RAM for association only, and the browser demo never uploads a frame. Person re-identification across sessions or cameras is a different (regulated) problem and is out of scope by design.</p>
        </div>
      </section>

      <section className="card stack">
        <h3>Where to go next</h3>
        <div className="row">
          <a className="btn primary" href={REPO} target="_blank" rel="noreferrer">Source on GitHub ↗</a>
          <a className="btn" href="./docs/" target="_blank" rel="noreferrer">Documentation ↗</a>
          <Link className="btn" to="/benchmarks">Raw benchmark numbers</Link>
        </div>
        <pre style={{ margin: 0, padding: 16, background: "rgba(0,0,0,0.35)", borderRadius: 12, overflowX: "auto", fontSize: "0.82rem", border: "1px solid var(--border)" }}>
{`pip install -e ".[dev,dashboard]"
emot track my_video.mp4 --tracker botsort --save out.mp4 --jsonl tracks.jsonl
emot ask tracks.jsonl "How many people crossed the middle line?"
emot results            # the full benchmark in your terminal`}
        </pre>
        <p className="sub" style={{ fontSize: "0.82rem" }}>Video: MOT17 © MOTChallenge (Milan et al., arXiv:1603.00831), CC BY-NC-SA 3.0. Models: Ultralytics YOLO (AGPL-3.0), OSNet weights from the torchreid authors (MIT).</p>
      </section>
    </div>
  );
}
