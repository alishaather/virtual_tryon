import { useEffect, useRef, useState } from 'react'
import { Navigate } from 'react-router-dom'
import { useMeasure } from '../context/MeasureContext'

function CapturePage() {
  const { heightCm, sizing, clothing } = useMeasure()
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const [photo, setPhoto] = useState(null)
  const [info, setInfo] = useState('')

  useEffect(() => {
    let cancelled = false

    async function startCamera() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Camera not available. The page needs HTTPS (or localhost).')
        return
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 960 },
            //aspectRatio: { ideal:9/16 },
          },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch (err) {
        if (err.name === 'NotAllowedError') {
          setError('Camera permission was denied. Allow it in your browser settings and reload.')
        } else if (err.name === 'NotFoundError') {
          setError('No camera found on this device.')
        } else if (err.name === 'NotReadableError') {
          setError('The camera is being used by another app. Close it and reload.')
        } else {
          setError('Could not start the camera: ' + err.message)
        }
      }
    }

    startCamera()

    return () => {
      cancelled = true
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  // hooks are above, so this redirect is allowed here
  if (heightCm === null || sizing === null || clothing === null) {
    return <Navigate to="/" replace />
  }

  function takePhoto() {
    const video = videoRef.current
    if (!video || !video.videoWidth) return

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    canvas.toBlob(
      (blob) => {
        if (!blob) return
        setPhoto({ blob, url: URL.createObjectURL(blob) })
      },
      'image/jpeg',
      0.92
    )
  }

  function retake() {
    if (photo) URL.revokeObjectURL(photo.url)
    setPhoto(null)
  }

  return (
    <div className="page wide">
      <h1>Capture</h1>
      <p>
        Height: {heightCm} cm · {sizing} · {clothing}
      </p>

      {info && <p className="hint">Camera: {info}</p>}

      {error && <p className="error">{error}</p>}

      <div className="camera-box">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="mirror"
          onLoadedData={(e) => {
            setReady(true)
            setInfo(e.target.videoWidth + 'x' + e.target.videoHeight)  
          }}
        />
        {photo && (
          <img src={photo.url} alt="Captured" className="mirror photo-layer" />
        )}
      </div>

      {photo ? (
        <button className="btn" onClick={retake}>
          Retake
        </button>
      ) : (
        <button className="btn" onClick={takePhoto} disabled={!ready}>
          Capture
        </button>
      )}
    </div>
  )
}

export default CapturePage