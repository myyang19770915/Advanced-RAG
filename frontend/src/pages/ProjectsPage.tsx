import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ProjectsAPI, Project } from '../api/client';

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState('');
  const [err, setErr] = useState('');

  const load = () => ProjectsAPI.list().then(setProjects).catch(e => setErr(String(e)));

  useEffect(() => {
    load();
  }, []);

  const create = async () => {
    if (!name.trim()) return;
    try {
      await ProjectsAPI.create(name.trim());
      setName('');
      setErr('');
      load();
    } catch (e: any) {
      setErr(e?.response?.data?.detail || String(e));
    }
  };

  const remove = async (id: string) => {
    if (!confirm('Delete this project?')) return;
    await ProjectsAPI.remove(id);
    load();
  };

  return (
    <div>
      <h2>Projects</h2>
      <div className="card">
        <div className="row">
          <input
            placeholder="New project name"
            value={name}
            onChange={e => setName(e.target.value)}
          />
          <button onClick={create}>Create</button>
        </div>
        {err && <p style={{ color: 'red' }}>{err}</p>}
      </div>
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Collection</th>
              <th>Status</th>
              <th>Created</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {projects.map(p => (
              <tr key={p.id}>
                <td>
                  <Link to={`/projects/${p.id}`}>{p.name}</Link>
                </td>
                <td>{p.collection_name}</td>
                <td>
                  <span className={`badge ${p.status}`}>{p.status}</span>
                </td>
                <td>{new Date(p.created_at).toLocaleString()}</td>
                <td>
                  <button className="ghost" onClick={() => remove(p.id)}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
