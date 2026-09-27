import React from 'react'

const SLIDER_CONFIG = [
  { key: 'x0', label: 'x₀ (initial x position)', min: -2, max: 2, step: 0.05 },
  { key: 'y0', label: 'y₀ (initial y position)', min: -2, max: 2, step: 0.05 },
  { key: 'vx0', label: 'vx₀ (initial x velocity)', min: -2, max: 2, step: 0.05 },
  { key: 'vy0', label: 'vy₀ (initial y velocity)', min: -2, max: 2, step: 0.05 },
  { key: 't_max', label: 'Time horizon', min: 1, max: 30, step: 0.5 },
  { key: 'n_points', label: 'Number of points', min: 50, max: 400, step: 10 },
]

export default function ControlsPanel({ values, onChange }) {
  return (
    <div className="controls-panel">
      <h2>Initial conditions</h2>
      <p className="hint">Canonical units, central mass fixed at GM = 1.0</p>
      {SLIDER_CONFIG.map(({ key, label, min, max, step }) => (
        <div className="control-row" key={key}>
          <div className="control-label-row">
            <label htmlFor={key}>{label}</label>
            <span className="control-value">{values[key]}</span>
          </div>
          <input
            id={key}
            type="range"
            min={min}
            max={max}
            step={step}
            value={values[key]}
            onChange={(e) => onChange(key, parseFloat(e.target.value))}
          />
        </div>
      ))}
    </div>
  )
}
