// PDF export utility — connects to jsPDF + html2canvas
// In prototype mode this shows an informational message when the backend is not connected.

import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

export interface PDFExportOptions {
  filename?: string;
  onError?: (msg: string) => void;
}

/**
 * Capture a DOM element as a PDF and trigger a browser download.
 * Falls back to an alert if capture fails.
 */
export async function exportElementToPDF(
  element: HTMLElement,
  options: PDFExportOptions = {},
): Promise<void> {
  const { filename = 'accident-intel-report.pdf', onError } = options;

  try {
    const canvas = await html2canvas(element, {
      scale: 2,
      useCORS: true,
      backgroundColor: '#0a0f1e',
      logging: false,
    });

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'px',
      format: [canvas.width / 2, canvas.height / 2],
    });

    pdf.addImage(imgData, 'PNG', 0, 0, canvas.width / 2, canvas.height / 2);
    pdf.save(filename);
  } catch (err) {
    const message =
      'PDF export is in prototype state. Connect to the backend for full PDF generation.';
    if (onError) {
      onError(message);
    } else {
      alert(message);
    }
    console.error('PDF export error:', err);
  }
}

/** Format seconds as MM:SS for display in reports */
export function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
