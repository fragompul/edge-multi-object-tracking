import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

/**
 * Tracks the frame index of a <video> precisely. Uses `requestVideoFrameCallback` (frame-exact
 * `mediaTime`) where available and falls back to `requestAnimationFrame` + `currentTime`.
 * `onFrame` runs for every presented frame and after every seek.
 */
export function useVideoClock(
  ref: RefObject<HTMLVideoElement>,
  fps: number,
  frames: number,
  onFrame: (index: number, now: number) => void,
) {
  const [playing, setPlaying] = useState(false);
  const [index, setIndex] = useState(0);
  const cb = useRef(onFrame);
  cb.current = onFrame;

  const emit = useCallback(
    (t: number) => {
      const i = Math.min(frames - 1, Math.max(0, Math.round(t * fps)));
      setIndex((prev) => (prev === i ? prev : i));
      cb.current(i, performance.now());
    },
    [fps, frames],
  );

  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    let handle = 0;
    let stopped = false;
    const useRvfc = typeof v.requestVideoFrameCallback === "function";
    const loop = () => {
      if (stopped) return;
      if (useRvfc) {
        handle = v.requestVideoFrameCallback((_now, meta) => {
          emit(meta.mediaTime);
          loop();
        });
      } else {
        handle = requestAnimationFrame(() => {
          emit(v.currentTime);
          loop();
        });
      }
    };
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onSeek = () => emit(v.currentTime);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("seeked", onSeek);
    v.addEventListener("loadeddata", onSeek);
    setPlaying(!v.paused);
    loop();
    return () => {
      stopped = true;
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("seeked", onSeek);
      v.removeEventListener("loadeddata", onSeek);
      if (useRvfc) v.cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(handle);
    };
  }, [ref, emit]);

  const seek = useCallback(
    (i: number) => {
      const v = ref.current;
      if (v) v.currentTime = Math.min(frames - 1, Math.max(0, i)) / fps + 1e-4;
    },
    [ref, fps, frames],
  );

  const toggle = useCallback(() => {
    const v = ref.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  }, [ref]);

  return { playing, index, seek, toggle };
}
