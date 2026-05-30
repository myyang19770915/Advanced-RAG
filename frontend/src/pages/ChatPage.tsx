import { useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatAPI, Citation } from '../api/client';
import ProjectNav from '../components/ProjectNav';

interface Msg {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

export default function ChatPage() {
  const { id } = useParams<{ id: string }>();
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState('');
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);

  const send = async () => {
    if (!q.trim() || !id) return;
    const question = q.trim();
    setMsgs(m => [...m, { role: 'user', content: question }]);
    setQ('');
    setLoading(true);
    setStreaming(false);
    try {
      await ChatAPI.streamAsk(id, question, sessionId, {
        onSession: (sid) => setSessionId(sid),
        onCitations: (citations) => {
          setStreaming(true);
          setMsgs(m => [...m, { role: 'assistant', content: '', citations }]);
        },
        onToken: (token) => {
          setMsgs(m => {
            const updated = [...m];
            const last = updated[updated.length - 1];
            if (last?.role === 'assistant') {
              updated[updated.length - 1] = { ...last, content: last.content + token };
            }
            return updated;
          });
        },
        onDone: () => {
          setLoading(false);
          setStreaming(false);
        },
        onError: (err) => {
          setMsgs(m => {
            const last = m[m.length - 1];
            if (last?.role === 'assistant') {
              return [...m.slice(0, -1), { ...last, content: 'Error: ' + err }];
            }
            return [...m, { role: 'assistant', content: 'Error: ' + err }];
          });
          setLoading(false);
          setStreaming(false);
        },
      });
    } catch (e: any) {
      setMsgs(m => [...m, { role: 'assistant', content: 'Error: ' + e.message }]);
      setLoading(false);
      setStreaming(false);
    }
  };

  const newConversation = () => {
    setMsgs([]);
    setSessionId(undefined);
    setQ('');
  };

  return (
    <div>
      <ProjectNav id={id!} />

      <div className="chat-toolbar">
        <button className="new-chat-btn" onClick={newConversation} title="Start a new conversation">
          ＋ New conversation
        </button>
      </div>

      <div className="card" style={{ minHeight: 300, display: 'flex', flexDirection: 'column' }}>
        {msgs.length === 0 && (
          <p style={{ color: '#9ca3af', textAlign: 'center', margin: 'auto' }}>
            Ask anything about the documents in this project…
          </p>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`chat-bubble ${m.role}`}>
            {m.role === 'assistant' ? (
              <div className="chat-md">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
              </div>
            ) : (
              <div>{m.content}</div>
            )}
            {m.citations && m.citations.length > 0 && (
              <div className="citations-section">
                <div className="citations-label">Sources ({m.citations.length})</div>
                {m.citations.map((c, j) => (
                  <div key={j} className="citation-item">
                    <div className="citation-meta">
                      <span className="citation-file">{c.original_file}</span>
                      <span className="citation-chunk">#{c.chunk_index}</span>
                      <span className="citation-score">score {c.score.toFixed(3)}</span>
                    </div>
                    <div className="citation-preview">{c.text_preview}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && !streaming && (
          <div className="chat-bubble assistant" style={{ opacity: 0.6 }}>
            <span className="spinner dark" /> Thinking…
          </div>
        )}
      </div>
      <div className="card">
        <div className="row">
          <input
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="Ask anything..."
            onKeyDown={e => e.key === 'Enter' && !loading && send()}
            disabled={loading}
          />
          <button onClick={send} disabled={loading} style={{ minWidth: 72 }}>
            {loading ? <><span className="spinner" />…</> : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
