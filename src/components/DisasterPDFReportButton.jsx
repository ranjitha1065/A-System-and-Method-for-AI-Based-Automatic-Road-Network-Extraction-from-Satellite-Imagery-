import { FileText } from 'lucide-react';
import { jsPDF } from 'jspdf';
import toast from 'react-hot-toast';

export default function DisasterPDFReportButton({ result }) {
  const generate = () => {
    const stats = result.statistics;
    const pdf = new jsPDF({ unit: 'mm', format: 'a4' });
    pdf.setFillColor(15, 23, 42); pdf.rect(0, 0, 210, 297, 'F');
    pdf.setTextColor(255, 255, 255); pdf.setFontSize(22); pdf.text('RoadAI Disaster Analysis Report', 15, 19);
    pdf.setTextColor(203, 213, 225); pdf.setFontSize(10); pdf.text(`Generated: ${new Date().toLocaleString()}`, 15, 27);
    const lines = [`Model: ${stats.model}`, `Road pixels before: ${stats.road_pixels_before.toLocaleString()}`, `Road pixels after: ${stats.road_pixels_after.toLocaleString()}`, `Road pixels lost: ${stats.road_pixels_lost.toLocaleString()}`, `Damage percentage: ${stats.damage_percentage}%`, `Estimated road length lost: ${stats.estimated_length_lost.toLocaleString()} m`, `Prediction time: ${stats.prediction_time} s`, 'Estimate uses a 0.5 meter-per-pixel approximation.'];
    pdf.setFontSize(10); lines.forEach((line, index) => pdf.text(line, 15, 38 + index * 6));
    pdf.addImage(result.before_image, 'PNG', 15, 94, 85, 64); pdf.addImage(result.before_mask, 'PNG', 110, 94, 85, 64);
    pdf.addImage(result.after_image, 'PNG', 15, 166, 85, 64); pdf.addImage(result.after_mask, 'PNG', 110, 166, 85, 64);
    pdf.addImage(result.difference_map, 'PNG', 55, 238, 100, 45);
    pdf.setTextColor(148, 163, 184); pdf.setFontSize(8); pdf.text('Green: unchanged roads - Red: missing roads - Yellow: changed/new regions', 15, 290);
    pdf.save('roadai-disaster-analysis-report.pdf'); toast.success('Disaster report generated');
  };
  return <button onClick={generate} className="action-button"><FileText size={16}/>Generate PDF</button>;
}
