// api.js
// Thin wrapper around the OrbitalNet FastAPI backend (web/backend/main.py).
// The backend URL defaults to the standard local uvicorn address; override
// with a VITE_API_BASE_URL env var (e.g. in a .env file) if you run the
// backend elsewhere.

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/api/health`)
  if (!res.ok) {
    throw new Error(`Health check failed: HTTP ${res.status}`)
  }
  return res.json()
}

export async function fetchPrediction({ x0, y0, vx0, vy0, t_max, n_points }) {
  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x0, y0, vx0, vy0, t_max, n_points }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Prediction request failed: HTTP ${res.status}`)
  }
  return res.json()
}
