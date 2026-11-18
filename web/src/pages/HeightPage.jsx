import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMeasure } from '../context/MeasureContext'

const MIN_HEIGHT = 120
const MAX_HEIGHT = 220

function HeightPage() {
  const [value, setValue] = useState('')
  const { setHeightCm } = useMeasure()
  const navigate = useNavigate()

  const height = parseFloat(value)
  let error = ''
  if (value !== '') {
    if (isNaN(height)) {
      error = 'Enter a valid number'
    } else if (height < MIN_HEIGHT || height > MAX_HEIGHT) {
      error = `Height should be between ${MIN_HEIGHT} and ${MAX_HEIGHT} cm`
    }
  }
  const isValid = value !== '' && error === ''

  function handleChange(e) {
    // only allow up to 3 digits and one decimal
    if (/^\d{0,3}\.?\d?$/.test(e.target.value)) {
      setValue(e.target.value)
    }
  }

  function handleSubmit(e) {
    e.preventDefault()
    if (!isValid) return
    setHeightCm(height)
    navigate('/setup')
  }

  return (
    <div className="page">
      <h1>What's your height?</h1>
      <p>We use it to work out your measurements from the photo.</p>

      <form onSubmit={handleSubmit}>
        <label htmlFor="height">Height (cm)</label>
        <input
          id="height"
          className="field"
          type="text"
          inputMode="decimal"
          placeholder="e.g. 165"
          value={value}
          onChange={handleChange}
        />
        {error && <p className="error">{error}</p>}
        <button className="btn" type="submit" disabled={!isValid}>
          Continue
        </button>
      </form>
    </div>
  )
}

export default HeightPage