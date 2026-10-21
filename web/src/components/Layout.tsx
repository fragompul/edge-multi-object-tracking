import { NavLink, Outlet } from "react-router-dom";
import { LogoMark } from "./Logo";

const links = [
  { to: "/", label: "Overview", end: true },
  { to: "/arena", label: "Tracker Arena" },
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/edge", label: "Edge Lab" },
  { to: "/studio", label: "Analytics Studio" },
  { to: "/live", label: "Live in browser" },
  { to: "/about", label: "About" },
];

export const REPO = "https://github.com/fragompul/edge-multi-object-tracking";

export function Layout() {
  return (
    <div className="shell">
      <nav className="nav">
        <div className="container nav-inner">
          <NavLink to="/" className="brand">
            <LogoMark />
            <span>EMOT <small>Studio</small></span>
          </NavLink>
          <div className="nav-links">
            {links.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end} className={({ isActive }) => (isActive ? "active" : "")}>{l.label}</NavLink>
            ))}
            <a href={REPO} target="_blank" rel="noreferrer" className="gh">GitHub ↗</a>
          </div>
        </div>
      </nav>
      <main>
        <div className="container"><Outlet /></div>
      </main>
      <footer>
        <div className="container row spread">
          <span>© Francisco Javier Gómez Pulido · MIT licence · built with YOLO, ONNX Runtime and a lot of Kalman filters</span>
          <span>Video clips: MOT17 © MOTChallenge, CC BY-NC-SA 3.0</span>
        </div>
      </footer>
    </div>
  );
}
