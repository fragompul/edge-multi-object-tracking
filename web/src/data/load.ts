import { useEffect, useState } from "react";
import type { ClipTracks, Manifest, Results } from "./types";

const base = import.meta.env.BASE_URL;
export const dataUrl = (path: string): string => `${base}data/${path}`;
export const assetUrl = (path: string): string => `${base}${path}`;

const cache = new Map<string, Promise<unknown>>();

export function fetchJson<T>(path: string): Promise<T> {
  let p = cache.get(path);
  if (!p) {
    p = fetch(dataUrl(path)).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.json();
    });
    cache.set(path, p);
  }
  return p as Promise<T>;
}

export interface Loaded<T> {
  data: T | null;
  error: string | null;
}

function useJson<T>(path: string | null): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ data: null, error: null });
  useEffect(() => {
    if (!path) return;
    let live = true;
    setState({ data: null, error: null });
    fetchJson<T>(path)
      .then((data) => live && setState({ data, error: null }))
      .catch((e: Error) => live && setState({ data: null, error: e.message }));
    return () => {
      live = false;
    };
  }, [path]);
  return state;
}

export const useManifest = () => useJson<Manifest>("manifest.json");
export const useResults = () => useJson<Results>("results.json");
export const useClipTracks = (path: string | null) => useJson<ClipTracks>(path);
