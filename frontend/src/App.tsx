import { NavLink, Route, Routes, Navigate } from 'react-router-dom';
import ProjectsPage from './pages/ProjectsPage';
import ProjectDetailPage from './pages/ProjectDetailPage';
import ChatPage from './pages/ChatPage';
import QcPage from './pages/QcPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>Advanced RAG</h1>
        <nav>
          <NavLink to="/projects" className={({ isActive }) => (isActive ? 'active' : '')}>
            Projects
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => (isActive ? 'active' : '')}>
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
