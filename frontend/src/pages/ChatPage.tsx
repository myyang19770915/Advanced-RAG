import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, NavLink, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatAPI, Citation, ChatSessionSummary, ProjectsAPI } from '../api/client';
import PdfPreviewModal from '../components/PdfPreviewModal';


interface ModalTarget {
  pdfUrl: string;
  page: number;
  bbox: number[] | null;
  pageWidth: number | null;
  pageHeight: number | null;
  textPreview: string;
}

/** Process a plain text string and replace [file#N] markers with clickable badge buttons */
function processCiteText(
  text: string,
  citations: Citation[],
  openModal: (t: ModalTarget) => void,
  keyPrefix: string
): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  const re = /\[([^\]]+?)#(\d+)\]/g;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const filename = m[1];
    const chunkIdx = parseInt(m[2], 10);
    const cit = citations.find(
      c => c.original_file === filename && c.chunk_index === chunkIdx
    );
    const canPreview = cit?.document_id && cit?.page != null;
    const badge = cit ? getTypeBadge(cit) : null;
    const showIcon = badge && (cit?.primary_type || 'text') !== 'text';
    parts.push(
      <button
        key={`${keyPrefix}-${m.index}`}
        className="cite-badge"
        title={canPreview ? `Open PDF — ${filename} chunk #${chunkIdx}${badge ? ` (${badge.label})` : ''}` : `${filename} chunk #${chunkIdx}`}
        style={{ cursor: canPreview ? 'pointer' : 'default' }}
        onClick={() => {
          if (!cit?.document_id || cit?.page == null) return;
          openModal({
            pdfUrl: `/api/files/${cit.document_id}/download`,
            page: cit.page,
            bbox: cit.bbox ?? null,
            pageWidth: cit.page_width ?? null,
            pageHeight: cit.page_height ?? null,
            textPreview: cit.text_preview ?? '',
          });
        }}
      >
        {showIcon ? `${badge!.icon} ` : ''}{filename.replace(/\.pdf$/i, '')} #{chunkIdx}
      </button>
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length > 1 ? <React.Fragment>{parts}</React.Fragment> : parts[0] ?? text;
}

/** Build a renderer that turns [file#N] markers into clickable badges.
 *  Uses React.Children.map to handle mixed-content arrays (bold + citation tag)
 *  without manual recursion that could cause stack overflows. */
function makeRenderer(citations: Citation[], openModal: (t: ModalTarget) => void) {
  return function renderWithCiteBadges(children: React.ReactNode): React.ReactNode {
    return React.Children.map(children, (child, i) => {
      if (typeof child === 'string') {
        return processCiteText(child, citations, openModal, String(i));
      }
      return child;
    }) ?? children;
  };
}

/** Map a Citation primary_type to an emoji + human label for source badges. */
const TYPE_BADGES: Record<string, { icon: string; label: string }> = {
  text:    { icon: '📝', label: 'text'    },
  picture: { icon: '🖼️', label: 'picture' },
  table:   { icon: '📊', label: 'table'   },
  formula: { icon: '🔢', label: 'formula' },
  code:    { icon: '💻', label: 'code'    },
  heading: { icon: '📑', label: 'heading' },
};

function getTypeBadge(c: Citation): { icon: string; label: string } {
  const t = (c.primary_type || 'text').toLowerCase();
  return TYPE_BADGES[t] ?? TYPE_BADGES.text;
}

interface Msg {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  mode?: 'answer' | 'clarify';
}

export default function ChatPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState('');
  const [projectName, setProjectName] = useState('');
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [modalTarget, setModalTarget] = useState<ModalTarget | null>(null);
  const [expandedCitations, setExpandedCitations] = useState<Set<number>>(new Set());
  const [chatSidebarOpen, setChatSidebarOpen] = useState(false);
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!id) return;
    ProjectsAPI.get(id).then(p => setProjectName(p.name)).catch(() => {});
  }, [id]);

  const refreshSessions = async () => {
    if (!id) return;
    try { setSessions(await ChatAPI.listSessions(id)); } catch { /* ignore */ }
  };

  useEffect(() => { refreshSessions(); }, [id]);

  useEffect(() => {
    const el = chatAreaRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [msgs]);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    }
  }, []);

  const send = async () => {
    if (!q.trim() || !id || loading) return;
    const question = q.trim();
    setMsgs(m => [...m, { role: 'user', content: question }]);
    setQ('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setLoading(true);
    setStreaming(false);
    let currentMode: 'answer' | 'clarify' = 'answer';
    try {
      await ChatAPI.streamAsk(id, question, sessionId, {
        onSession: (sid) => setSessionId(sid),
        onMeta: (meta) => { currentMode = meta.mode; },
        onCitations: (citations) => {
          setStreaming(true);
          setMsgs(m => [...m, { role: 'assistant', content: '', citations, mode: currentMode }]);
        },
        onToken: (token) => {
          setMsgs(m => {
            const updated = [...m];
            const last = updated[updated.length - 1];
            if (last?.role === 'assistant') {
              updated[updated.length - 1] = { ...last, content: last.content + token, mode: currentMode };
            }
            return updated;
          });
        },
        onDone: () => {
          setLoading(false);
          setStreaming(false);
          refreshSessions();
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
    setExpandedCitations(new Set());
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const loadSession = async (sid: string) => {
    try {
      const detail = await ChatAPI.getSession(sid);
      setSessionId(detail.id);
      setMsgs(detail.messages.map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
        citations: m.citations,
        mode: 'answer' as const,
      })));
      setExpandedCitations(new Set());
    } catch { /* ignore */ }
  };

  const deleteSession = async (sid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await ChatAPI.deleteSession(sid);
    if (sid === sessionId) newConversation();
    refreshSessions();
  };

  const toggleCitations = (idx: number) => {
    setExpandedCitations(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx); else next.add(idx);
      return next;
    });
  };

  return (
    <div className="chat-page-root">
      {/* Mobile backdrop – closes chat sidebar when tapped */}
      {chatSidebarOpen && (
        <div
          className="chat-sidebar-mobile-overlay"
          onClick={() => setChatSidebarOpen(false)}
        />
      )}

      {/* ── Left sidebar ── */}
      <aside className={`chat-sidebar${chatSidebarOpen ? ' open' : ''}`}>
        <div className="chat-sidebar-top">
          <button className="chat-back-btn" onClick={() => navigate('/projects')}>
            ← Projects
          </button>
          <span className="chat-project-name" title={projectName}>{projectName}</span>
        </div>

        <div className="chat-phase-nav">
          <NavLink to={`/projects/${id}`} end className={({ isActive }) => `chat-phase-link${isActive ? ' active' : ''}`}>
            <span className="chat-phase-icon">📁</span> Collection
          </NavLink>
          <NavLink to={`/projects/${id}/chat`} className={({ isActive }) => `chat-phase-link${isActive ? ' active' : ''}`}>
            <span className="chat-phase-icon">💬</span> Chat
          </NavLink>
          <NavLink to={`/projects/${id}/qc`} className={({ isActive }) => `chat-phase-link${isActive ? ' active' : ''}`}>
            <span className="chat-phase-icon">🔍</span> QC
          </NavLink>
        </div>

        <div className="chat-sidebar-divider" />

        <button className="new-chat-sidebar-btn" onClick={newConversation}>
          <span>＋</span> New conversation
        </button>

        <div className="chat-history-section">
          <div className="chat-history-label">Recent</div>
          <div className="chat-history-list">
            {sessions.length === 0 && (
              <div className="chat-history-empty">No conversations yet</div>
            )}
            {sessions.map(s => (
              <div
                key={s.id}
                className={`chat-history-item${s.id === sessionId ? ' active' : ''}`}
                onClick={() => loadSession(s.id)}
              >
                <span className="chat-history-title">{s.title}</span>
                <button
                  className="chat-history-delete"
                  onClick={e => deleteSession(s.id, e)}
                  title="Delete conversation"
                >✕</button>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* ── Main chat area ── */}
      <div className="chat-main">
        {/* Mobile header – only visible on ≤768px via CSS */}
        <div className="chat-mobile-header">
          <button
            className="chat-history-toggle-btn"
            onClick={() => setChatSidebarOpen(o => !o)}
          >
            ☰ History
          </button>
          <span style={{ flex: 1 }} />
          <button
            className="chat-history-toggle-btn"
            onClick={newConversation}
          >
            ＋ New
          </button>
        </div>

        {/* Messages */}
        <div className="chat-messages-area" ref={chatAreaRef}>
          {msgs.length === 0 ? (
            <div className="chat-empty-state">
              <div className="chat-empty-icon">💬</div>
              <h2 className="chat-empty-title">What would you like to know?</h2>
              <p className="chat-empty-subtitle">
                Ask anything about the documents in this project.<br />
                I'll find relevant context and cite sources inline.
              </p>
            </div>
          ) : (
            msgs.map((m, i) => (
              <div key={i} className={`chat-msg-row ${m.role}`}>
                {m.role === 'assistant' && (
                  <div className="chat-msg-avatar">AI</div>
                )}
                <div className="chat-msg-body">
                  {m.mode === 'clarify' && (
                    <div className="clarify-badge">⚠ Need more info</div>
                  )}
                  {m.role === 'assistant' ? (
                    <div className="chat-md">
                      {(() => {
                        const renderWithCiteBadges = makeRenderer(m.citations ?? [], setModalTarget);
                        return (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              p:      ({ children }) => <p>{renderWithCiteBadges(children)}</p>,
                              li:     ({ children }) => <li>{renderWithCiteBadges(children)}</li>,
                              td:     ({ children }) => <td>{renderWithCiteBadges(children)}</td>,
                              strong: ({ children }) => <strong>{renderWithCiteBadges(children)}</strong>,
                              em:     ({ children }) => <em>{renderWithCiteBadges(children)}</em>,
                              h1:     ({ children }) => <h1>{renderWithCiteBadges(children)}</h1>,
                              h2:     ({ children }) => <h2>{renderWithCiteBadges(children)}</h2>,
                              h3:     ({ children }) => <h3>{renderWithCiteBadges(children)}</h3>,
                              h4:     ({ children }) => <h4>{renderWithCiteBadges(children)}</h4>,
                            }}
                          >{m.content}</ReactMarkdown>
                        );
                      })()}
                    </div>
                  ) : (
                    <div className="chat-user-text">{m.content}</div>
                  )}
                  {m.citations && m.citations.length > 0 && (
                    <div className="chat-citations-wrap">
                      <button
                        className="citations-toggle-btn"
                        onClick={() => toggleCitations(i)}
                      >
                        {expandedCitations.has(i) ? '▾' : '▸'}
                        &nbsp;{m.citations.length} source{m.citations.length > 1 ? 's' : ''}
                      </button>
                      {expandedCitations.has(i) && (
                        <div className="citations-expanded">
                          {m.citations.map((c, j) => {
                            const badge = getTypeBadge(c);
                            return (
                              <div key={j} className="citation-item">
                                <div className="citation-meta">
                                  <span
                                    className={`citation-type-badge type-${(c.primary_type || 'text').toLowerCase()}`}
                                    title={`Source type: ${badge.label}`}
                                  >
                                    {badge.icon} {badge.label}
                                  </span>
                                  <span className="citation-file">{c.original_file}</span>
                                  <span className="citation-chunk">#{c.chunk_index}</span>
                                  <span className="citation-score">score {c.score.toFixed(3)}</span>
                                </div>
                                <div className="citation-preview">
                                  {(c.text_preview || '').length > 240
                                    ? (c.text_preview || '').slice(0, 240) + '…'
                                    : c.text_preview}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {loading && !streaming && (
            <div className="chat-msg-row assistant">
              <div className="chat-msg-avatar">AI</div>
              <div className="chat-msg-body">
                <div className="chat-thinking">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input bar */}
        <div className="chat-input-area">
          <div className="chat-input-box">
            <textarea
              ref={textareaRef}
              value={q}
              onChange={e => { setQ(e.target.value); autoResize(); }}
              placeholder="Ask anything…"
              disabled={loading}
              rows={1}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey && !loading) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <button
              className="chat-send-btn"
              onClick={send}
              disabled={!q.trim() || loading}
              title="Send (Enter)"
            >
              {loading ? <span className="spinner" /> : '↑'}
            </button>
          </div>
          <div className="chat-input-hint">
            Enter to send · Shift+Enter for new line · Click <span className="cite-badge" style={{cursor:'default',pointerEvents:'none'}}>doc #1</span> badges to preview PDF sources
          </div>
        </div>
      </div>

      {modalTarget && (
        <PdfPreviewModal
          pdfUrl={modalTarget.pdfUrl}
          page={modalTarget.page}
          bbox={modalTarget.bbox}
          pageWidth={modalTarget.pageWidth}
          pageHeight={modalTarget.pageHeight}
          textPreview={modalTarget.textPreview}
          onClose={() => setModalTarget(null)}
        />
      )}
    </div>
  );
}

