import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Sidebar } from './components/layout/Sidebar'
import { MapPage } from './pages/MapPage'
import { OptimizerPage } from './pages/OptimizerPage'
import { LabPage } from './pages/LabPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground font-sans">
        <Sidebar />
        <main className="flex-1 overflow-hidden flex flex-col">
          <Routes>
            <Route path="/" element={<Navigate to="/map" replace />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/optimizer" element={<OptimizerPage />} />
            <Route path="/lab" element={<LabPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
