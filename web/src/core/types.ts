/** `[left, top, right, bottom]` in pixels. */
export type XYXY = [number, number, number, number];

export interface Detection {
  box: XYXY;
  conf: number;
}

export interface TrackedBox {
  id: number;
  box: XYXY;
  conf: number;
}

/** frame number (1-based) -> tracked boxes in that frame. */
export type FrameTracks = Map<number, TrackedBox[]>;
