import { Routes, Route } from 'react-router-dom'
import HeightPage from './pages/HeightPage'
import CapturePage from './pages/CapturePage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<HeightPage />} />
      <Route path="/capture" element={<CapturePage />} />
    </Routes>
  )
}

export default App