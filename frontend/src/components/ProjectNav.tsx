import { useEffect, useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { ProjectsAPI } from '../api/client';

interface Props {
  id: string;
  projectName?: string;
}

export default function ProjectNav({ id, projectName }: Props) {
  const [fetchedName, setFetchedName] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    if (projectName) return; // parent already supplies it
    ProjectsAPI.get(id).then(p => setFetchedName(p.name)).catch(() => {});
  }, [id, projectName]);

  const name = projectName ?? fetchedName;

  return (
    <div className="project-header">
      <div className="project-header-top">
        <button className="back-btn" onClick={() => navigate('/projects')}>
          ← Projects
        </button>
        {name && <h2 className="project-name-header">{name}</h2>}
      </div>
      <div className="phase-tabs">
        <NavLink
          to={`/projects/${id}`}
          end
          className={({ isActive }) => `phase-tab${isActive ? ' active' : ''}`}
        >
          📁 Collection
        </NavLink>
        <NavLink
          to={`/projects/${id}/chat`}
          className={({ isActive }) => `phase-tab${isActive ? ' active' : ''}`}
        >
          💬 Chat
        </NavLink>
        <NavLink
          to={`/projects/${id}/qc`}
          className={({ isActive }) => `phase-tab${isActive ? ' active' : ''}`}
        >
          🔍 QC
        </NavLink>
      </div>
    </div>
  );
}
