import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  DocumentsAPI,
  Document,
  ProjectsAPI,
  Project,
  RulesAPI,
  TrainingAPI,
  TrainingStatus,
} from '../api/client';
import ProjectNav from '../components/ProjectNav';

// ── Progress bar helpers ──────────────────────────────────────────────────────
const STATUS_STEPS: Record<string, { pct: number; label: string; color: string; anim: boolean }> = {
  uploaded:   { pct: 5,   label: '已上傳',      color: '#94a3b8', anim: false },
  queued:     { pct: 18,  label: '排隊中…',     color: '#f59e0b', anim: true  },
  converting: { pct: 40,  label: 'PDF 轉換中…', color: '#3b82f6', anim: true  },
  chunking:   { pct: 65,  label: '切割分塊中…', color: '#8b5cf6', anim: true  },
  embedding:  { pct: 85,  label: '向量化中…',   color: '#06b6d4', anim: true  },
  ready:      { pct: 100, label: '✓ 完成',      color: '#22c55e', anim: false },
  error:      { pct: 100, label: '✕ 錯誤',      color: '#ef4444', anim: false },
};

function StatusBar({ status, chunksDone, chunksTotal }: { status: string; chunksDone?: number; chunksTotal?: number }) {
  const s = STATUS_STEPS[status] ?? { pct: 0, label: status, color: '#94a3b8', anim: false };

  // During chunking, interpolate from 40% → 85% based on sections processed
  let pct = s.pct;
  let label = s.label;
  if (status === 'chunking' && chunksTotal && chunksTotal > 0) {
    pct = Math.round(40 + (chunksDone! / chunksTotal) * 45);
    label = `切割分塊中… ${chunksDone}/${chunksTotal}`;
  }

  return (
    <div className="status-cell">
      <span className={`badge ${status}`}>{status}</span>
      <div className="prog-track">
        <div
          className={`prog-fill${s.anim ? ' anim' : ''}`}
          style={{
            width: `${pct}%`,
            background: s.anim
              ? `linear-gradient(90deg, ${s.color} 25%, ${s.color}99 50%, ${s.color} 75%)`
              : s.color,
          }}
        />
      </div>
      <div className="status-label">{label}</div>
    </div>
  );
}

// ── Spinner helper ────────────────────────────────────────────────────────────
function Spinner({ dark }: { dark?: boolean }) {
  return <span className={`spinner${dark ? ' dark' : ''}`} />;
}

// ── Page component ────────────────────────────────────────────────────────────
export default function ProjectDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [docs, setDocs] = useState<Document[]>([]);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [domain, setDomain] = useState('');
  const [candidates, setCandidates] = useState<{ label: string; name: string; system_prompt: string }[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  // Button loading states
  const [uploading, setUploading] = useState(false);
  const [buildingRule, setBuildingRule] = useState(false);
  const [startingTrain, setStartingTrain] = useState(false);
  const [resettingTrain, setResettingTrain] = useState(false);
  const [usingRuleLabel, setUsingRuleLabel] = useState<string | null>(null);
  const [actioningDocs, setActioningDocs] = useState<Set<string>>(new Set());

  const refresh = async () => {
    if (!id) return;
    const [p, d, s] = await Promise.all([
      ProjectsAPI.get(id),
      DocumentsAPI.list(id),
      TrainingAPI.status(id),
    ]);
    setProject(p);
    setDocs(d);
    setStatus(s);
  };

  useEffect(() => { refresh(); }, [id]);

  // Poll while any document is processing
  useEffect(() => {
    if (!status || status.overall_status !== 'running') return;
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [status?.overall_status]);

  const upload = async () => {
    const files = fileRef.current?.files;
    if (!files || files.length === 0 || !id) return;
    setUploading(true);
    try {
      await DocumentsAPI.upload(id, Array.from(files));
      if (fileRef.current) fileRef.current.value = '';
      await refresh();
    } finally {
      setUploading(false);
    }
  };

  const startTrain = async () => {
    if (!id) return;
    setStartingTrain(true);
    try {
      await TrainingAPI.start(id);
      await refresh();
    } finally {
      setStartingTrain(false);
    }
  };

  const resetTrain = async () => {
    if (!id) return;
    if (!window.confirm('Reset all documents and re-train from scratch?\n(Qdrant collection will be deleted and rebuilt)')) return;
    setResettingTrain(true);
    try {
      await TrainingAPI.reset(id);
      await refresh();
    } finally {
      setResettingTrain(false);
    }
  };

  const deleteDoc = async (docId: string, filename: string) => {
    if (!id) return;
    if (!window.confirm(`Delete "${filename}" and remove all its vectors from Qdrant?`)) return;
    setActioningDocs(s => new Set(s).add(docId));
    try {
      await DocumentsAPI.remove(id, docId);
      await refresh();
    } finally {
      setActioningDocs(s => { const n = new Set(s); n.delete(docId); return n; });
    }
  };

  const retrainDoc = async (docId: string, filename: string) => {
    if (!id) return;
    if (!window.confirm(`Re-train "${filename}"?\nOld vectors will be deleted and the document will be re-processed.`)) return;
    setActioningDocs(s => new Set(s).add(docId));
    try {
      await DocumentsAPI.retrain(id, docId);
      await refresh();
    } finally {
      setActioningDocs(s => { const n = new Set(s); n.delete(docId); return n; });
    }
  };

  const buildRule = async () => {
    if (!id) return;
    setBuildingRule(true);
    try {
      const sampleIds = docs.filter(d => d.status === 'ready').slice(0, 3).map(d => d.id);
      const res = await RulesAPI.build(id, sampleIds, domain);
      setCandidates(res.candidates);
    } finally {
      setBuildingRule(false);
    }
  };

  const useRule = async (label: string, name: string, system_prompt: string) => {
    if (!id) return;
    setUsingRuleLabel(label);
    try {
      await RulesAPI.setActive(id, `${label} - ${name}`, system_prompt);
      alert('Rule activated');
    } finally {
      setUsingRuleLabel(null);
    }
  };

  const isTrainRunning = status?.overall_status === 'running';

  if (!project) return <p>Loading...</p>;

  return (
    <div>
      <ProjectNav id={id!} projectName={project.name} />

      <div className="card">
        <h3>1. Upload PDFs / text files</h3>
        <div className="row">
          <input ref={fileRef} type="file" multiple disabled={uploading} />
          <button onClick={upload} disabled={uploading}>
            {uploading ? <><Spinner />Uploading…</> : 'Upload'}
          </button>
        </div>
      </div>

      <div className="card">
        <h3>2. Rule Builder (optional)</h3>
        <div className="row">
          <input
            placeholder="domain hint (e.g. 勞動法規)"
            value={domain}
            onChange={e => setDomain(e.target.value)}
            disabled={buildingRule}
          />
          <button onClick={buildRule} disabled={buildingRule}>
            {buildingRule ? <><Spinner />Generating…</> : 'Generate candidates'}
          </button>
        </div>
        {candidates.map(c => (
          <div key={c.label} className="card" style={{ marginTop: 10 }}>
            <strong>{c.label} — {c.name}</strong>
            <pre style={{ whiteSpace: 'pre-wrap' }}>{c.system_prompt}</pre>
            <button
              onClick={() => useRule(c.label, c.name, c.system_prompt)}
              disabled={usingRuleLabel !== null}
            >
              {usingRuleLabel === c.label ? <><Spinner />Activating…</> : 'Use this rule'}
            </button>
          </div>
        ))}
      </div>

      <div className="card">
        <h3>3. Training</h3>
        <button
          onClick={startTrain}
          disabled={isTrainRunning || startingTrain}
        >
          {startingTrain
            ? <><Spinner />Starting…</>
            : isTrainRunning
              ? <><Spinner />Running…</>
              : 'Start training'}
        </button>
        {' '}
        <button
          onClick={resetTrain}
          disabled={isTrainRunning || resettingTrain}
          style={{ background: '#dc3545', color: '#fff', marginLeft: 8 }}
        >
          {resettingTrain ? <><Spinner />Resetting…</> : 'Reset & Retrain'}
        </button>
        <p style={{ marginTop: 8 }}>
          Overall: <span className={`badge ${status?.overall_status}`}>{status?.overall_status}</span>
        </p>
        <table>
          <thead>
            <tr>
              <th>Filename</th>
              <th>Ver</th>
              <th>Status</th>
              <th>Chunks</th>
              <th>Error</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {(status?.documents || []).map(d => {
              const doc = docs.find(x => x.id === d.document_id);
              const inAction = actioningDocs.has(d.document_id);
              const busy = ['queued','converting','chunking','embedding'].includes(d.status) || inAction;
              return (
                <tr key={d.document_id}>
                  <td>{d.filename}</td>
                  <td>
                    {doc && doc.version > 1
                      ? <span className="badge" style={{ background: '#6c757d' }}>v{doc.version}</span>
                      : <span style={{ color: '#aaa' }}>v1</span>
                    }
                  </td>
                  <td>
                    <StatusBar status={d.status} chunksDone={d.chunks_done} chunksTotal={d.chunks_total} />
                  </td>
                  <td>{d.chunk_count}</td>
                  <td style={{ color: 'red', fontSize: '0.8em' }}>{d.error_message}</td>
                  <td style={{ whiteSpace: 'nowrap' }}>
                    <button
                      onClick={() => retrainDoc(d.document_id, d.filename)}
                      disabled={busy}
                      style={{ marginRight: 4, fontSize: '0.8em' }}
                      title="Delete vectors and re-train this document"
                    >
                      {inAction ? <Spinner /> : '↻'} Retrain
                    </button>
                    <button
                      onClick={() => deleteDoc(d.document_id, d.filename)}
                      disabled={busy}
                      style={{ background: '#dc3545', color: '#fff', fontSize: '0.8em' }}
                      title="Delete document and remove its vectors"
                    >
                      {inAction ? <Spinner /> : '✕'} Delete
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
