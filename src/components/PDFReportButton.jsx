import { FileText } from 'lucide-react';
import { jsPDF } from 'jspdf';
import toast from 'react-hot-toast';

export default function PDFReportButton({ result, stats }) {
  const generate = () => {
    const pdf = new jsPDF({ unit: 'mm', format: 'a4' });
    pdf.setFillColor(15, 23, 42); pdf.rect(0, 0, 210, 297, 'F');
    pdf.setTextColor(255, 255, 255); pdf.setFontSize(24); pdf.text('RoadAI Prediction Report', 15, 20);
    pdf.setTextColor(203, 213, 225); pdf.setFontSize(10); pdf.text(`Generated: ${new Date(stats.metadata.timestamp).toLocaleString()}`, 15, 28);
    const lines = [`File: ${stats.metadata.filename}`, `Model: ${stats.model}`, `Road coverage: ${stats.coverage}`, `Road pixels: ${stats.roadPixels}`, `Estimated road length: ${stats.roadLength}`, `Confidence: ${stats.confidence}`, `Image resolution: ${stats.size}`, `Prediction time: ${stats.time}`, 'Road length uses a 0.5 meter-per-pixel approximation.'];
    pdf.setFontSize(11); lines.forEach((line, index) => pdf.text(line, 15, 40 + index * 7));
    pdf.addImage(result.original, 'PNG', 15, 108, 85, 64); pdf.addImage(result.mask, 'PNG', 110, 108, 85, 64);
    pdf.addImage(result.overlay, 'PNG', 15, 180, 180, 96);
    pdf.setFontSize(9); pdf.setTextColor(148, 163, 184); pdf.text('RoadAI • TensorFlow U-Net road segmentation', 15, 286);
    pdf.save('roadai-prediction-report.pdf'); toast.success('PDF report generated');
  };
  return <button onClick={generate} className="action-button"><FileText size={16}/>PDF Report</button>;
}
