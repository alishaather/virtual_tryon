import { Navigate } from 'react-router-dom'
import { useMeasure } from '../context/MeasureContext'

function CapturePage() {
  const { heightCm, sizing, clothing } = useMeasure()

  if (heightCm === null || sizing === null || clothing === null) {
    return <Navigate to="/" replace />
  }

  return (
    <div className="page">
      <h1>Capture</h1>
      <p>Height: {heightCm} cm</p>
      <p>Sizing: {sizing}, clothing: {clothing}</p>
      <p>camera comes here</p>
    </div>
  )
}

export default CapturePage