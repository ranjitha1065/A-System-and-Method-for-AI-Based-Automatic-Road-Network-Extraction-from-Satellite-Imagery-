import { motion } from 'framer-motion';
import DisasterAnalysisCard from '../components/DisasterAnalysisCard';

export default function DisasterAnalysis() { return <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mx-auto max-w-6xl"><div className="mb-7"><p className="text-xs font-bold uppercase tracking-[.2em] text-orange-300">RoadAI resilience workspace</p><h1 className="mt-2 text-4xl font-extrabold">Disaster Analysis</h1><p className="mt-3 text-slate-400">Compare satellite scenes before and after an event to estimate road-network disruption.</p></div><DisasterAnalysisCard/></motion.div>; }
