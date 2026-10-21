import { type RefObject, useEffect, useRef, useState } from "react";

export function useWidth<T extends HTMLElement>(initial = 720): [RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    setW(Math.max(280, Math.round(el.getBoundingClientRect().width)));
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}
