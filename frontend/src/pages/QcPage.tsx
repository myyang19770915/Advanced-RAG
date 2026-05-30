import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { AuditReport, CoverageReport, QcAPI } from '../api/client';
import ProjectNav from '../components/ProjectNav';

export default function QcPage() {
  const { id } = useParams<{ id: string }>();
  const [cov, setCov] = useState<CoverageReport | null>(null);
  const [audit, setAudit] = useState<AuditReport | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!id) return;
    QcAPI.coverage(id).then(setCov);
  }, [id]);

  const runAudit = async () => {
    if (!id) return;
    setLoading(true);
    try {
      const r = await QcAPI.audit(id, 5);
      setAudit(r);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <ProjectNav id={id!} />

      <div className="card">
        <h3>Coverage</h3>
        {cov && (
          <ul>
            <li>Documents total: {cov.documents_total}</li>
            <li>Documents ready: {cov.documents_ready}</li>
            <li>Chunks in vector DB: {cov.chunks_in_db}</li>
            <li>Chunks expected: {cov.chunks_expected}</li>
            <li>
              Coverage ratio:{' '}
              <strong>{(cov.coverage_ratio * 100).toFixed(1)}%</strong>
            </li>
          </ul>
        )}
      </div>
      <div className="card">
        <h3>Audit</h3>
        <button onClick={runAudit} disabled={loading}>
          {loading
            ? <><span className="spinner" />Running audit…</>
            : 'Run audit on 5 pending QAs'}
        </button>
        {audit && (
          <div style={{ marginTop: 10 }}>
            <p>
              Passed: <span className="badge ready">{audit.passed}</span>
              {' / '}
              Failed: <span className="badge failed">{audit.failed}</span>
              {' / '}
              Total: {audit.total}
            </p>
            <table>
              <thead>
                <tr>
                  <th>Question</th>
                  <th>Answer</th>
                  <th>Verdict</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {audit.results.map(r => (
                  <tr key={r.qa_test_id}>
                    <td>{r.question}</td>
                    <td>{r.answer.slice(0, 200)}</td>
                    <td>
                      <span className={`badge ${r.passed ? 'ready' : 'failed'}`}>
                        {r.passed ? 'passed' : 'failed'}
                      </span>
                    </td>
                    <td>{r.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
