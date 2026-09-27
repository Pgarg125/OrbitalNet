import React, { useMemo } from 'react'

const SIZE = 520
const PADDING_FRACTION = 0.15

/**
 * Computes a SQUARE data-space bounding box (so circular orbits render as
 * circles, not ellipses) covering both trajectories plus the origin.
 */
function computeSquareBounds(pointArrays) {
  const allX = [0]
  const allY = [0]
  pointArrays.forEach(({ x, y }) => {
    allX.push(...x)
    allY.push(...y)
  })
  const minX = Math.min(...allX)
  const maxX = Math.max(...allX)
  const minY = Math.min(...allY)
  const maxY = Math.max(...allY)

  const width = maxX - minX || 1
  const height = maxY - minY || 1
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const halfSpan = (Math.max(width, height) / 2) * (1 + PADDING_FRACTION)

  return {
    minX: cx - halfSpan,
    maxX: cx + halfSpan,
    minY: cy - halfSpan,
    maxY: cy + halfSpan,
  }
}

function toSvgPoints(x, y, bounds) {
  const spanX = bounds.maxX - bounds.minX
  const spanY = bounds.maxY - bounds.minY
  return x
    .map((xv, i) => {
      const sx = ((xv - bounds.minX) / spanX) * SIZE
      const sy = SIZE - ((y[i] - bounds.minY) / spanY) * SIZE // flip: SVG y grows downward
      return `${sx.toFixed(2)},${sy.toFixed(2)}`
    })
    .join(' ')
}

export default function OrbitPlot({ trueTraj, predTraj }) {
  const { bounds, truePoints, predPoints, originPoint } = useMemo(() => {
    if (!trueTraj || !predTraj) return {}
    const bounds = computeSquareBounds([trueTraj, predTraj])
    const truePoints = toSvgPoints(trueTraj.x, trueTraj.y, bounds)
    const predPoints = toSvgPoints(predTraj.x, predTraj.y, bounds)
    const [ox, oy] = toSvgPoints([0], [0], bounds).split(',').map(Number)
    return { bounds, truePoints, predPoints, originPoint: { ox, oy } }
  }, [trueTraj, predTraj])

  if (!bounds) {
    return <div className="orbit-plot-placeholder">Enter initial conditions to see a prediction.</div>
  }

  return (
    <div className="orbit-plot">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} width="100%" height="100%" role="img" aria-label="Predicted vs true orbit">
        {/* reference crosshair through the origin */}
        <line x1={originPoint.ox} y1="0" x2={originPoint.ox} y2={SIZE} className="axis-line" />
        <line x1="0" y1={originPoint.oy} x2={SIZE} y2={originPoint.oy} className="axis-line" />

        <polyline points={truePoints} className="true-orbit" />
        <polyline points={predPoints} className="pred-orbit" />

        {/* central body marker */}
        <circle cx={originPoint.ox} cy={originPoint.oy} r="7" className="central-body" />
      </svg>

      <div className="plot-legend">
        <span className="legend-item">
          <span className="legend-swatch true-swatch" /> True (RK45)
        </span>
        <span className="legend-item">
          <span className="legend-swatch pred-swatch" /> Predicted (PINN)
        </span>
        <span className="legend-item">
          <span className="legend-swatch central-swatch" /> Central body
        </span>
      </div>
    </div>
  )
}
