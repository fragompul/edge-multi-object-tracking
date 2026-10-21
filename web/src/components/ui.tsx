import type { ReactNode } from "react";

export function Seg<T extends string>({ value, options, onChange, label }: {
  value: T;
  options: { value: T; label: ReactNode; dot?: string }[];
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} className={o.value === value ? "on" : ""} onClick={() => onChange(o.value)} aria-pressed={o.value === value}>
          {o.dot && <i className="dot" style={{ background: o.dot }} />}
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {children}
    </label>
  );
}

export function Stat({ value, label, delta, color }: { value: ReactNode; label: string; delta?: ReactNode; color?: string }) {
  return (
    <div className="stat">
      <span className="v" style={color ? { color } : undefined}>{value}</span>
      <span className="l">{label}</span>
      {delta && <span className="d">{delta}</span>}
    </div>
  );
}

export function PageHead({ eyebrow, title, children }: { eyebrow: string; title: ReactNode; children?: ReactNode }) {
  return (
    <header className="page-head fade-in">
      <div className="eyebrow">{eyebrow}</div>
      <h2 style={{ fontSize: "clamp(1.7rem,3.2vw,2.4rem)" }}>{title}</h2>
      {children && <p>{children}</p>}
    </header>
  );
}

export function Loading({ what = "data" }: { what?: string }) {
  return <div className="row" style={{ padding: 40, justifyContent: "center", color: "var(--muted)" }}><span className="spinner" /> Loading {what}…</div>;
}

export function ErrorBox({ message }: { message: string }) {
  return <div className="callout warn">Could not load data: {message}</div>;
}
