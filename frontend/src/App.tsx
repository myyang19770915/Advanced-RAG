import { useState } from 'react';
import { NavLink, Route, Routes, Navigate } from 'react-router-dom';
import ProjectsPage from './pages/ProjectsPage';
import ProjectDetailPage from './pages/ProjectDetailPage';
import ChatPage from './pages/ChatPage';
import QcPage from './pages/QcPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const closeSidebar = () => setSidebarOpen(false);

  return (
    <div className="layout">
      {/* Mobile top bar – only visible on ≤768px via CSS */}
      <div className="mobile-topbar">
        <button
          className="hamburger"
          onClick={() => setSidebarOpen(o => !o)}
          aria-label="Toggle menu"
        >
          ☰
        </button>
        <span className="mobile-topbar-title">Advanced RAG</span>
      </div>

      {/* Backdrop overlay – closes sidebar when tapped on mobile */}
      <div
        className={`sidebar-overlay${sidebarOpen ? ' open' : ''}`}
        onClick={closeSidebar}
      />

      <aside className={`sidebar${sidebarOpen ? ' open' : ''}`}>
        <h1>Advanced RAG</h1>
        <nav>
          <NavLink
            to="/projects"
            className={({ isActive }) => (isActive ? 'active' : '')}
            onClick={closeSidebar}
          >
            Projects
          </NavLink>
          <NavLink
            to="/settings"
            className={({ isActive }) => (isActive ? 'active' : '')}
            onClick={closeSidebar}
          >
            Settings
          </NavLink>
        </nav>
      </aside>

      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:id" element={<ProjectDetailPage />} />
          <Route path="/projects/:id/chat" element={<ChatPage />} />
          <Route path="/projects/:id/qc" element={<QcPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}
