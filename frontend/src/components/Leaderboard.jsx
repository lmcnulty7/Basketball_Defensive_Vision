import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts'

const API = ''

const EVENT_COLORS = {
  blocks:      '#ef4444',
  deflections: '#3b82f6',
  contested:   '#f59e0b',
  steals:      '#10b981',
}

export default function Leaderboard() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [sortKey, setSortKey] = useState('actions_per_min')

  useEffect(() => {
    axios.get(`${API}/api/leaderboard?top_n=20`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const sorted = [...data].sort((a, b) => b[sortKey] - a[sortKey])

  const chartData = sorted.slice(0, 12).map(p => ({
    name:        `#${p.track_id}`,
    blocks:      p.blocks,
    deflections: p.deflections,
    contested:   p.contested,
    steals:      p.steals,
  }))

  if (loading) return <div className="loading">Loading stats…</div>
  if (!data.length) return <div className="empty">No stats found. Run the pipeline first.</div>

  return (
    <div className="leaderboard">
      <div className="section-header">
        <h2>Defensive Leaderboard</h2>
        <select value={sortKey} onChange={e => setSortKey(e.target.value)} className="sort-select">
          <option value="actions_per_min">Actions / min</option>
          <option value="blocks">Blocks</option>
          <option value="deflections">Deflections</option>
          <option value="steals">Steals</option>
          <option value="contested">Contested shots</option>
        </select>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="blocks"      stackId="a" fill={EVENT_COLORS.blocks} />
          <Bar dataKey="deflections" stackId="a" fill={EVENT_COLORS.deflections} />
          <Bar dataKey="contested"   stackId="a" fill={EVENT_COLORS.contested} />
          <Bar dataKey="steals"      stackId="a" fill={EVENT_COLORS.steals} />
        </BarChart>
      </ResponsiveContainer>

      <table className="stats-table">
        <thead>
          <tr>
            <th>Player</th>
            <th onClick={() => setSortKey('steals')}      className="sortable">Steals</th>
            <th onClick={() => setSortKey('blocks')}      className="sortable">Blocks</th>
            <th onClick={() => setSortKey('deflections')} className="sortable">Defl.</th>
            <th onClick={() => setSortKey('contested')}   className="sortable">Cont.</th>
            <th onClick={() => setSortKey('actions_per_min')} className="sortable">Act/min</th>
            <th>Clips</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(p => (
            <tr key={p.track_id}>
              <td className="player-id">#{p.track_id}</td>
              <td>{p.steals}</td>
              <td>{p.blocks}</td>
              <td>{p.deflections}</td>
              <td>{p.contested}</td>
              <td className="highlight">{p.actions_per_min.toFixed(2)}</td>
              <td>{p.clips_seen ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
