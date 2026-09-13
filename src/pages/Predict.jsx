import { motion } from 'framer-motion';
import PredictionCard from '../components/PredictionCard';
export default function Predict(){return <motion.div initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} className="mx-auto max-w-4xl"><div className="mb-7"><p className="text-xs font-bold uppercase tracking-[.2em] text-orange-300">RoadAI workspace</p><h1 className="mt-2 text-4xl font-extrabold">Prediction Studio</h1><p className="mt-3 text-slate-400">Turn raw satellite imagery into clean road-network intelligence.</p></div><PredictionCard/></motion.div>}
