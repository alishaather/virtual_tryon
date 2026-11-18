import { useState } from 'react'
import { useNavigate, Navigate } from 'react-router-dom'
import { useMeasure } from '../context/MeasureContext'

const SIZING = [
  { value: 'women', label: "Women's sizing" },
  { value: 'men', label: "Men's sizing" },
]

const CLOTHING = [
  { value: 'fitted', label: 'Fitted clothes' },
  { value: 'loose', label: 'Loose clothes' },
]

function SetupPage() {
  const { heightCm, setSizing, setClothing } = useMeasure()
  const [sizingChoice, setSizingChoice] = useState(null)
  const [clothingChoice, setClothingChoice] = useState(null)
  const navigate = useNavigate()

  // refreshed page = no height, go back to start
  if (heightCm === null) {
    return <Navigate to="/" replace />
  }

  function handleContinue() {
    setSizing(sizingChoice)
    setClothing(clothingChoice)
    navigate('/capture')
  }

  return (
    <div className="page">
      <h1>A few more details</h1>

      <p className="question">Which sizing do you use?</p>
      <div className="options">
        {SIZING.map((opt) => (
          <button
            key={opt.value}
            type="button"
            className={sizingChoice === opt.value ? 'option selected' : 'option'}
            onClick={() => setSizingChoice(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <p className="question">What are you wearing in the photo?</p>
      <p className="hint">Fitted clothes usually give more accurate measurements.</p>  
      <div className="options">
        {CLOTHING.map((opt) => (
          <button
            key={opt.value}
            type="button"
            className={clothingChoice === opt.value ? 'option selected' : 'option'}
            onClick={() => setClothingChoice(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <button
        className="btn"
        disabled={!sizingChoice || !clothingChoice}
        onClick={handleContinue}
      >
        Continue
      </button>
    </div>
  )
}

export default SetupPage