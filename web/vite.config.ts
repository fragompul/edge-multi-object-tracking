/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `base: "./"` keeps every asset URL relative, so the same build works on a GitHub Pages
// project path (/edge-multi-object-tracking/), on a custom domain, and from `vite preview`.
// onnxruntime-web references its WASM via `new URL(..., import.meta.url)`, which makes Vite emit a
// second 21 MB copy. We serve the runtime from /ort/ (see Live.tsx `wasmPaths`), so drop it.
const dropDuplicateOrtWasm = {
  name: "drop-duplicate-ort-wasm",
  generateBundle(_: unknown, bundle: Record<string, unknown>) {
    for (const key of Object.keys(bundle)) if (/ort-wasm.*\.wasm$/.test(key) && key.startsWith("assets/")) delete bundle[key];
  },
};

export default defineConfig({
  base: "./",
  plugins: [react(), dropDuplicateOrtWasm],
  build: { target: "es2022", chunkSizeWarningLimit: 1500 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
