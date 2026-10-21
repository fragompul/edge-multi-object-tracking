import { REGION_PRESETS, type TrackStore } from "./store";

/** Offline question router -- a port of `emot/agent/planner.py`. No LLM, no network. */
export interface ToolCallTrace {
  name: string;
  args: Record<string, unknown>;
  result: Record<string, unknown>;
}

export interface Answer {
  text: string;
  calls: ToolCallTrace[];
}

const REGIONS = Object.keys(REGION_PRESETS).filter((r) => r !== "full").join("|");
const NUM = "(\\d+(?:\\.\\d+)?)";

function regionOf(q: string): string | null {
  const m = new RegExp(`\\b(?:in|on|at|inside|within)?\\s*the\\s+(${REGIONS})(?:\\s+(?:half|side|part|area|region))?`).exec(q);
  return m ? m[1] : null;
}

function windowOf(q: string): [number | null, number | null] {
  let m = new RegExp(`between\\s+${NUM}\\s*(?:s|sec|seconds)?\\s*(?:and|to|-)\\s*${NUM}`).exec(q);
  if (m) return [parseFloat(m[1]), parseFloat(m[2])];
  m = new RegExp(`first\\s+${NUM}\\s*(?:s|sec|seconds)`).exec(q);
  if (m) return [0, parseFloat(m[1])];
  return [null, null];
}

export function askOffline(store: TrackStore, question: string): Answer {
  const q = question.toLowerCase().trim().replace(/\?+$/, "");
  const calls: ToolCallTrace[] = [];
  const call = (name: string, args: Record<string, unknown>) => {
    const result = store.call(name, args);
    calls.push({ name, args, result });
    return result as Record<string, any>;
  };

  let [start, end] = windowOf(q);
  const last = new RegExp(`last\\s+${NUM}\\s*(?:s|sec|seconds)`).exec(q);
  if (last) [start, end] = [Math.max(0, store.durationS - parseFloat(last[1])), null];
  const region = regionOf(q);
  const has = (re: RegExp) => re.test(q);

  let text: string;
  let m: RegExpExecArray | null;
  if ((m = /\b(?:track|id|person)\s*#?(\d+)\b/.exec(q))) {
    const r = call("track_info", { track_id: parseInt(m[1], 10) });
    text = r.error
      ? String(r.error)
      : `Track ${r.track_id} is visible from ${r.first_seen_s} s to ${r.last_seen_s} s (${r.frames_visible} frames), moving at a median ${r.median_speed_bh_per_s ?? 0} body-heights/s.`;
  } else if (has(/\b(cross|crossed|crossing|enter|entered|exit|exited|pass|passed)\b/)) {
    const horizontal = q.includes("horizontal");
    const coords = horizontal ? { x1: 0, y1: 0.5, x2: 1, y2: 0.5 } : { x1: 0.5, y1: 0, x2: 0.5, y2: 1 };
    const r = call("line_crossings", { ...coords, start_s: start, end_s: end });
    text = `${r.total} crossings of the ${horizontal ? "horizontal" : "vertical"} middle line: ${r.a_to_b} from side A to B and ${r.b_to_a} from B to A.`;
  } else if (has(/\b(how long|dwell|stay|stayed|linger|spent)\b/)) {
    const r = call("dwell_time", { region: region ?? "center" });
    text = `${r.people_who_entered} people entered the ${region ?? "center"} region; mean stay ${r.mean_dwell_s} s, longest ${r.max_dwell_s} s.`;
  } else if (has(/\b(crowd|crowded|packed)\b/)) {
    const n = /(\d+)\s*(?:or more|\+)?\s*(?:people|persons|pedestrians)/.exec(q);
    const r = call("crowding", { threshold: n ? parseInt(n[1], 10) : 10, region });
    const eps = (r.episodes as { start_s: number; end_s: number; peak: number }[]).slice(0, 3);
    text = `${r.count} crowding episode(s)` + (r.count ? ": " + eps.map((e) => `${e.start_s}-${e.end_s} s (peak ${e.peak})`).join(", ") : ".");
  } else if (has(/\b(busiest|peak|most people|rush|busy)\b/)) {
    const r = call("busiest_moments", { top_k: 3, region });
    const top = (r.moments as { second: number; mean_people: number; peak: number }[])[0];
    text = top ? `The busiest second is ${top.second} s with ~${top.mean_people} people (peak ${top.peak}).` : "No people were tracked.";
  } else if (has(/\b(how many|count|number of)\b/)) {
    const r = call("count_people", { start_s: start, end_s: end, region });
    text = `${r.unique_people} distinct people (track IDs); at most ${r.max_simultaneous} visible at once, ${r.mean_simultaneous} on average.`;
  } else if (has(/\b(summary|summarize|overview|describe|what happens)\b/)) {
    const r = call("scene_summary", {});
    text = `${r.duration_s} s of video with ${r.unique_tracks} distinct tracks; ${r.mean_people_per_frame} people per frame on average, peaking at ${r.peak_people} at ${r.peak_time_s} s.`;
  } else {
    text = "I can answer counts, line crossings, dwell time, crowding, busiest moments, single tracks and a summary offline; try \"How many people crossed the vertical line?\"";
  }
  return { text, calls };
}
