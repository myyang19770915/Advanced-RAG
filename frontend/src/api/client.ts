import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
});

export interface Project {
  id: string;
  name: string;
  collection_name: string;
  status: string;
  description: string;
  created_at: string;
}

export interface Document {
  id: string;
  project_id: string;
  original_filename: string;
  status: string;
  chunk_count: number;
  error_message: string;
  created_at: string;
  version: number;
}

export interface DocumentStatus {
  document_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  error_message: string;
  chunks_done: number;
  chunks_total: number;
}

export interface TrainingStatus {
  project_id: string;
  overall_status: string;
  documents: DocumentStatus[];
}

export interface Citation {
  original_file: string;
  chunk_index: number;
  score: number;
  text_preview: string;
  download_url?: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  session_id: string;
}

export interface CoverageReport {
  project_id: string;
  documents_total: number;
  documents_ready: number;
  chunks_in_db: number;
  chunks_expected: number;
  coverage_ratio: number;
}

export interface AuditResult {
  qa_test_id: string;
  question: string;
  answer: string;
  passed: boolean;
  reason: string;
}

export interface AuditReport {
  project_id: string;
  total: number;
  passed: number;
  failed: number;
  results: AuditResult[];
}

export const ProjectsAPI = {
  list: () => api.get<Project[]>('/projects').then(r => r.data),
  create: (name: string, description = '') =>
    api.post<Project>('/projects', { name, description }).then(r => r.data),
  get: (id: string) => api.get<Project>(`/projects/${id}`).then(r => r.data),
  remove: (id: string) => api.delete(`/projects/${id}`),
};

export const DocumentsAPI = {
  list: (pid: string) =>
    api.get<Document[]>(`/projects/${pid}/documents`).then(r => r.data),
  upload: (pid: string, files: File[]) => {
    const fd = new FormData();
    files.forEach(f => fd.append('files', f));
    return axios
      .post<Document[]>(`/api/projects/${pid}/documents`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then(r => r.data);
  },
  remove: (pid: string, docId: string) =>
    api.delete(`/projects/${pid}/documents/${docId}`),
  retrain: (pid: string, docId: string) =>
    api.post<{ job_id: string; document_ids: string[] }>(
      `/projects/${pid}/documents/${docId}/retrain`
    ).then(r => r.data),
  downloadUrl: (docId: string) => `/api/files/${docId}/download`,
};

export const TrainingAPI = {
  start: (pid: string) =>
    api.post<{ job_id: string; document_ids: string[] }>(`/projects/${pid}/training/start`).then(r => r.data),
  reset: (pid: string) =>
    api.post<{ job_id: string; document_ids: string[] }>(`/projects/${pid}/training/reset`).then(r => r.data),
  status: (pid: string) =>
    api.get<TrainingStatus>(`/projects/${pid}/training/status`).then(r => r.data),
};

export const ChatAPI = {
  ask: (project_id: string, question: string, session_id?: string) =>
    api.post<ChatResponse>('/chat', { project_id, question, session_id }).then(r => r.data),

  streamAsk: async (
    project_id: string,
    question: string,
    callbacks: {
      onCitations: (citations: Citation[]) => void;
      onToken: (token: string) => void;
      onDone: () => void;
      onError: (err: string) => void;
    },
  ): Promise<void> => {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id, question }),
    });
    if (!resp.ok || !resp.body) {
      callbacks.onError(`HTTP ${resp.status}`);
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop() ?? '';
      for (const part of parts) {
        if (!part.trim()) continue;
        let event = '';
        let data = '';
        for (const line of part.split('\n')) {
          if (line.startsWith('event: ')) event = line.slice(7);
          else if (line.startsWith('data: ')) data = line.slice(6);
        }
        if (event === 'citations') callbacks.onCitations(JSON.parse(data));
        else if (event === 'token') callbacks.onToken(JSON.parse(data).t);
        else if (event === 'done') callbacks.onDone();
        else if (event === 'error') callbacks.onError(data);
      }
    }
  },
};

export const QcAPI = {
  coverage: (pid: string) =>
    api.get<CoverageReport>(`/projects/${pid}/qc/coverage`).then(r => r.data),
  audit: (pid: string, limit = 10) =>
    api.post<AuditReport>(`/projects/${pid}/qc/audit?limit=${limit}`).then(r => r.data),
};

export const RulesAPI = {
  build: (pid: string, sample_document_ids: string[], domain_hint = '') =>
    api.post<{ candidates: { label: string; name: string; system_prompt: string }[] }>(
      `/projects/${pid}/rule/build`,
      { sample_document_ids, domain_hint }
    ).then(r => r.data),
  setActive: (pid: string, name: string, system_prompt: string) =>
    api.post(`/projects/${pid}/rule`, { name, system_prompt }).then(r => r.data),
};

export const SettingsAPI = {
  get: () => api.get('/settings').then(r => r.data),
};

export default api;
