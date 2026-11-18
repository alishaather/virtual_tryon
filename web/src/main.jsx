import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.jsx'
import { MeasureProvider } from './context/MeasureContext.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <MeasureProvider>
        <App />
      </MeasureProvider>
    </BrowserRouter>
  </StrictMode>,
)