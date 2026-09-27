import React, { useEffect, useRef, useState } from 'react'
import ControlsPanel from './components/ControlsPanel.jsx'
import OrbitPlot from './components/OrbitPlot.jsx'
import MetricsPanel from './components/MetricsPanel.jsx'
import { fetchHealth, fetchPrediction } from './api.js'
import './App.css'

const DEFAULT_VALUES = {
  x0: 1.0,
  y0: 0.0,
  vx0: 0.0,
  vy0: 1.0,
  t_max: 10.0,
  n_points: 200,
}

const DEBOUNCE_MS = 250

export default function App() {
  const [values, setValues] = useState(DEFAULT_VALUES)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [backendStatus, setBackendStatus] = useState('checking') // checking | ok | no_model | unreachable
  const debounceRef = useRef(null)

  // Check backend/model availability once on mount.
  useEffect(() => {
    fetchHealth()
      .then((health) => setBackendStatus(health.model_loaded ? 'ok' : 'no_model'))
      .catch(() => setBackendStatus('unreachable'))
  }, [])

  // Debounced predict-on-change, exactly mirroring the Streamlit demo's
  // "recompute live as sliders move" behaviour.
  useEffect(() => {
    if (backendStatus !== 'ok') return
    if (debounceRef.current) clearTimeout(debounceRef.current)

    debounceRef.current = setTimeout(() => {
      setLoading(true)
      setError(null)
      fetchPrediction(values)
        .then((data) => setResult(data))
        .catch((err) => setError(err.message))
        .finally(() => setLoading(false))
    }, DEBOUNCE_MS)

    return () => clearTimeout(debounceRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, backendStatus])

  const handleChange = (key, value) => {
    setValues((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🛰️ OrbitalNet — Physics-Informed Orbit Prediction</h1>
        <p className="app-subtitle">
          Enter initial conditions for a two-body orbit and compare the PINN's prediction
          against the true numerically-integrated (RK45) trajectory.
        </p>
      </header>

      {backendStatus === 'unreachable' && (
        <div className="banner banner-error">
          Can't reach the backend API. Make sure it's running:{' '}
          <code>uvicorn web.backend.main:app --reload --port 8000</code> (run from the project
          root, in a separate terminal).
        </div>
      )}

      {backendStatus === 'no_model' && (
        <div className="banner banner-warning">
          Backend is running, but no trained PINN model was found. From the project root, run{' '}
          <code>python scripts/generate_data.py</code> then{' '}
          <code>python scripts/train_pinn.py</code>, then restart the backend.
        </div>
      )}

      {error && <div className="banner banner-error">{error}</div>}

      <main className="app-main">
        <ControlsPanel values={values} onChange={handleChange} />

        <section className="plot-section">
          {loading && <div className="loading-indicator">Predicting…</div>}
          <OrbitPlot
            trueTraj={result ? { x: result.true_x, y: result.true_y } : null}
            predTraj={result ? { x: result.pred_x, y: result.pred_y } : null}
          />
          <MetricsPanel mse={result?.mse} mae={result?.mae} />
        </section>
      </main>

      <footer className="app-footer">
        Canonical units, GM = {result?.gm ?? 1.0} (fixed across the whole dataset). Two-body
        problem only.
      </footer>
    </div>
  )
}
