import { useState } from 'react'
import Leaderboard from './components/Leaderboard'
import EventFeed from './components/EventFeed'
import ClipSummary from './components/ClipSummary'
import './App.css'

export default function App() {
  const [activeTab, setActiveTab] = useState('leaderboard')

  const tabs = [
    { id: 'leaderboard', label: 'Leaderboard' },
    { id: 'events',      label: 'Event Feed' },
    { id: 'clips',       label: 'Clips' },
  ]

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏀 NBA Defensive Vision</h1>
        <p className="subtitle">CV-extracted defensive stats from broadcast footage</p>
      </header>

      <nav className="tabs">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`tab ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="content">
        {activeTab === 'leaderboard' && <Leaderboard />}
        {activeTab === 'events'      && <EventFeed />}
        {activeTab === 'clips'       && <ClipSummary />}
      </main>
    </div>
  )
}
