import { Navigate } from 'react-router-dom'
import { useMeasure } from '../context/MeasureContext'

function CapturePage() {
  const { heightCm } = useMeasure()

  // if the page was refreshed there is no height, go back
  if (heightCm === null) {
    return <Navigate to="/" replace />
  }

  return (
    <div className="page">
      <h1>Capture</h1>
      <p>Height: {heightCm} cm</p>
      <p>camera comes here</p>
    </div>
  )
}

export default CapturePage