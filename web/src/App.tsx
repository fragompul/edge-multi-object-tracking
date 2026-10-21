import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { About } from "./pages/About";
import { Arena } from "./pages/Arena";
import { Benchmarks } from "./pages/Benchmarks";
import { EdgeLab } from "./pages/EdgeLab";
import { Home } from "./pages/Home";
import { Live } from "./pages/Live";
import { Studio } from "./pages/Studio";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="arena" element={<Arena />} />
        <Route path="benchmarks" element={<Benchmarks />} />
        <Route path="edge" element={<EdgeLab />} />
        <Route path="studio" element={<Studio />} />
        <Route path="live" element={<Live />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<Home />} />
      </Route>
    </Routes>
  );
}
