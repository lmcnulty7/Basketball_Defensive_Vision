import { useEffect, useState } from 'react'
import axios from 'axios'

const API = ''

const TYPE_BADGE = {
  steal:          { label: 'STEAL',     color: '#10b981' },
  block:          { label: 'BLOCK',     color: '#ef4444' },
  deflection:     { label: 'DEFL',      color: '#3b82f6' },
  contested_2pt:  { label: 'CONT 2PT',  color: '#f59e0b' },
  contested_3pt:  { label: 'CONT 3PT',  color: '#8b5cf6' },
  defensive_rebound: { label: 'DREB',   color: '#6b7280' },
}

function Badge({ type }) {
  const cfg = TYPE_BADGE[type] || { label: type, color: '#6b7280' }
  return (
    <span className="badge" style={{ background: cfg.color }}>
      {cfg.label}
    </span>
  )
}

export default function EventFeed() {
  const [events, setEvents] = useState([])
  const [filter, setFilter] = useState('all')
  const [minConf, setMinConf] = useState(0.6)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const params = new URLSearchParams({ min_confidence: minConf })
    if (filter !== 'all') params.set('event_type', filter)
    axios.get(`${API}/api/events?${params}`)
      .then(r => { setEvents(r.data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [filter, minConf])

  const types = ['all', 'steal', 'block', 'deflection', 'contested_2pt', 'contested_3pt']

  if (loading) return <div className="loading">Loading events…</div>

  return (
    <div className="event-feed">
      <div className="section-header">
        <h2>Event Feed  <span className="count">({events.length})</span></h2>
        <div className="filters">
          <select value={filter} onChange={e => setFilter(e.target.value)} className="sort-select">
            {types.map(t => <option key={t} value={t}>{t === 'all' ? 'All types' : t}</option>)}
          </select>
          <label className="conf-label">
            Min conf: {minConf.toFixed(1)}
            <input type="range" min={0} max={1} step={0.05} value={minConf}
              onChange={e => setMinConf(+e.target.value)} />
          </label>
        </div>
      </div>

      {!events.length
        ? <div className="empty">No events match the current filters.</div>
        : (
          <div className="event-list">
            {events.map((ev, i) => (
              <div key={i} className="event-row">
                <Badge type={ev.event_type} />
                <span className="ev-time">{ev.timestamp_sec.toFixed(1)}s</span>
                <span className="ev-player">#{ev.primary_player_id}</span>
                {ev.secondary_player_id != null &&
                  <span className="ev-secondary">vs #{ev.secondary_player_id}</span>}
                <span className="ev-conf">{(ev.confidence * 100).toFixed(0)}%</span>
                <span className="ev-clip">{ev.clip}</span>
                {ev.metadata?.defender_dist &&
                  <span className="ev-meta">{ev.metadata.defender_dist.toFixed(1)} ft</span>}
                {ev.metadata?.angle_deg &&
                  <span className="ev-meta">{ev.metadata.angle_deg.toFixed(0)}°</span>}
              </div>
            ))}
          </div>
        )
      }
    </div>
  )
}
