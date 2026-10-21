import { type ReactNode, useCallback, useState } from "react";

export interface TipState {
  x: number;
  y: number;
  content: ReactNode;
}

/** Hover tooltip positioned relative to the chart container. */
export function useTip() {
  const [tip, setTip] = useState<TipState | null>(null);
  const show = useCallback(
    (e: React.MouseEvent, container: HTMLElement | null, content: ReactNode) => {
      if (!container) return;
      const r = container.getBoundingClientRect();
      setTip({ x: e.clientX - r.left, y: e.clientY - r.top, content });
    },
    [],
  );
  const hide = useCallback(() => setTip(null), []);
  return { tip, show, hide };
}

export function Tip({ tip }: { tip: TipState | null }) {
  if (!tip) return null;
  return (
    <div className="tip" style={{ left: tip.x, top: tip.y }}>
      {tip.content}
    </div>
  );
}
