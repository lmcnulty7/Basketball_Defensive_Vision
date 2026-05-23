import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = ''

const TYPE_COLORS = [
  '#ef4444', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6', '#6b7280'
]

export default function ClipSummary() {
  const [clips, setClips] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get(`${API}/api/clips`)
      .then(r => { setClips(r.data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading clips…</div>
  if (!clips.length) return <div className="empty">No processed clips found.</div>

  const totals = {}
  clips.forEach(c => {
    Object.entries(c.event_types || {}).forEach(([k, v]) => {
      totals[k] = (totals[k] || 0) + v
    })
  })
  const pieData = Object.entries(totals).map(([name, value]) => ({ name, value }))

  return (
    <div className="clip-summary">
      <div className="section-header">
        <h2>Processed Clips  <span className="count">({clips.length})</span></h2>
      </div>

      <div className="clip-grid">
        {clips.map(c => (
          <div key={c.clip} className="clip-card">
            <div className="clip-name">{c.clip}</div>
            <div className="clip-meta">
              <span>{(c.duration_sec / 60).toFixed(1)} min</span>
              <span>{c.n_events} events</span>
              <span>{c.n_players} players</span>
            </div>
            <div className="clip-types">
              {Object.entries(c.event_types || {}).map(([k, v]) => (
                <span key={k} className="type-pill">{k.replace('_', ' ')}: {v}</span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {pieData.length > 0 && (
        <div className="pie-section">
          <h3>All Events by Type</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name"
                cx="50%" cy="50%" outerRadius={90} label>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={TYPE_COLORS[i % TYPE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
