/**
 * PdfPreviewModal — opens a PDF at a specific page and overlays a
 * semi-transparent highlight rectangle over the cited chunk region.
 *
 * Coordinate system from backend:
 *   BOTTOMLEFT origin (PDF native, points). bbox = [l, t, r, b] with t > b.
 *
 * Conversion to CSS (TOPLEFT origin) at the rendered scale:
 *   scaleX  = renderedW_px / pageWidth_pt
 *   scaleY  = renderedH_px / pageHeight_pt
 *   cssTop  = (pageHeight - t) * scaleY
 *   cssLeft = l * scaleX
 *   cssW    = (r - l) * scaleX
 *   cssH    = (t - b) * scaleY
 */
import { useEffect, useLayoutEffect, useRef, useState, useCallback } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
// Vite-friendly worker URL — emits a hashed asset and returns the public path.
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

interface Props {
  pdfUrl: string;
  page: number;
  bbox: number[] | null;
  pageWidth: number | null;
  pageHeight: number | null;
  textPreview: string;
  onClose: () => void;
}

export default function PdfPreviewModal({
  pdfUrl, page, bbox, pageWidth, pageHeight, textPreview, onClose,
}: Props) {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(page);
  const [renderWidth, setRenderWidth] = useState<number>(800);
  const [renderedSize, setRenderedSize] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState<number>(1);
  const bodyRef = useRef<HTMLDivElement>(null);
  const pageContainerRef = useRef<HTMLDivElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  // Close on Escape
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  // Fit page width to body
  useLayoutEffect(() => {
    const update = () => {
      if (bodyRef.current) {
        setRenderWidth(Math.max(400, bodyRef.current.clientWidth - 48));
      }
    };
    update();
    const ro = new ResizeObserver(update);
    if (bodyRef.current) ro.observe(bodyRef.current);
    return () => ro.disconnect();
  }, []);

  // Reset rendered size on page/zoom change
  useEffect(() => { setRenderedSize(null); }, [currentPage, zoom]);

  const onPageRenderSuccess = useCallback(() => {
    if (pageContainerRef.current) {
      const canvas = pageContainerRef.current.querySelector('canvas');
      if (canvas) setRenderedSize({ w: canvas.offsetWidth, h: canvas.offsetHeight });
    }
  }, []);

  const overlayStyle = (): React.CSSProperties | null => {
    if (!bbox || !pageWidth || !pageHeight || !renderedSize) return null;
    if (currentPage !== page) return null;
    const [l, t, r, b] = bbox;
    const sx = renderedSize.w / pageWidth;
    const sy = renderedSize.h / pageHeight;
    return {
      position: 'absolute',
      top:  (pageHeight - t) * sy,
      left: l * sx,
      width:  (r - l) * sx,
      height: (t - b) * sy,
      background: 'rgba(250, 204, 21, 0.32)',
      border: '2px solid rgba(234, 88, 12, 0.9)',
      boxShadow: '0 0 0 1px rgba(234,88,12,0.4), 0 4px 12px rgba(234,88,12,0.25)',
      borderRadius: 3,
      pointerEvents: 'none',
      zIndex: 10,
    };
  };
  const overlay = overlayStyle();

  return (
    <div
      className="pdf-modal-backdrop"
      ref={modalRef}
      onClick={(e) => { if (e.target === modalRef.current) onClose(); }}
    >
      <div className="pdf-modal">
        <div className="pdf-modal-header">
          <div className="pdf-modal-title">
            PDF Preview — page {currentPage} / {numPages || '…'}
          </div>
          <div className="pdf-modal-nav">
            <button className="ghost" disabled={currentPage <= 1}
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}>◀</button>
            <input type="number" min={1} max={numPages || 1} value={currentPage}
              onChange={e => {
                const v = Number(e.target.value);
                if (v >= 1 && v <= numPages) setCurrentPage(v);
              }}
              style={{ width: 52, textAlign: 'center', padding: '2px 4px' }} />
            <button className="ghost" disabled={currentPage >= numPages}
              onClick={() => setCurrentPage(p => Math.min(numPages, p + 1))}>▶</button>
            <span style={{ width: 12 }} />
            <button className="ghost"
              onClick={() => setZoom(z => Math.max(0.5, +(z - 0.1).toFixed(2)))}>−</button>
            <span style={{ fontSize: '0.78rem', minWidth: 42, textAlign: 'center' }}>
              {(zoom * 100).toFixed(0)}%
            </span>
            <button className="ghost"
              onClick={() => setZoom(z => Math.min(3, +(z + 0.1).toFixed(2)))}>+</button>
            <button className="ghost" title="Jump back to cited page"
              onClick={() => { setCurrentPage(page); setZoom(1); }}>↺ cited</button>
          </div>
          <button className="pdf-modal-close" onClick={onClose} title="Close (Esc)">✕</button>
        </div>

        {textPreview && (
          <div className="pdf-citation-bar">
            <span className="pdf-citation-label">Cited text:</span>{' '}{textPreview}
          </div>
        )}

        <div className="pdf-modal-body" ref={bodyRef}>
          <Document
            file={pdfUrl}
            onLoadSuccess={({ numPages }) => setNumPages(numPages)}
            onLoadError={(err) => console.error('PDF load error:', err)}
            loading={<div className="pdf-loading">Loading PDF…</div>}
            error={<div className="pdf-loading" style={{ color: '#ef4444' }}>Failed to load PDF.</div>}
          >
            <div ref={pageContainerRef}
                 style={{ position: 'relative', display: 'inline-block', margin: '0 auto' }}>
              <Page
                pageNumber={currentPage}
                width={renderWidth * zoom}
                onRenderSuccess={onPageRenderSuccess}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
              {overlay && <div style={overlay} />}
            </div>
          </Document>
        </div>
      </div>
    </div>
  );
}
