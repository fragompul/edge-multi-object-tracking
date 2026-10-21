// Prepares generated public assets (not committed): shared brand files and the ONNX Runtime Web
// WebGPU/WASM runtime that the "Live in browser" page loads on demand.
import { cpSync, existsSync, mkdirSync } from "node:fs";

mkdirSync("public/brand", { recursive: true });
cpSync("../docs/assets/brand", "public/brand", { recursive: true });

mkdirSync("public/ort", { recursive: true });
for (const f of ["ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]) {
  const src = `node_modules/onnxruntime-web/dist/${f}`;
  if (existsSync(src)) cpSync(src, `public/ort/${f}`);
}
