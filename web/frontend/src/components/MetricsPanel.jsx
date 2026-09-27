import React from 'react'

export default function MetricsPanel({ mse, mae }) {
  if (mse == null || mae == null) return null
  return (
    <div className="metrics-panel">
      <div className="metric-card">
        <span className="metric-label">MSE</span>
        <span className="metric-value">{mse.toFixed(5)}</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">MAE</span>
        <span className="metric-value">{mae.toFixed(5)}</span>
      </div>
    </div>
  )
}
