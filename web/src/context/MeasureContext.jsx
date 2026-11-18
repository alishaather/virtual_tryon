import { createContext, useContext, useState } from 'react'

const MeasureContext = createContext(null)

export function MeasureProvider({ children }) {
  const [heightCm, setHeightCm] = useState(null)
  const [sizing, setSizing] = useState(null)
  const [clothing, setClothing] = useState(null)

  return (
    <MeasureContext.Provider
      value={{ heightCm, setHeightCm, sizing, setSizing, clothing, setClothing }}
    >
      {children}
    </MeasureContext.Provider>
  )
}

export function useMeasure() {
  return useContext(MeasureContext)
}