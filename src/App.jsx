import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import { ArrowUp } from 'lucide-react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Predict from './pages/Predict';
import About from './pages/About';
import DisasterAnalysis from './pages/DisasterAnalysis';

export default function App() {
  const [page, setPage] = useState('home');
  return <div className="min-h-screen overflow-hidden bg-ink text-slate-100"><Toaster position="top-right" toastOptions={{style:{background:'#172033',color:'#f8fafc',border:'1px solid rgba(255,255,255,.14)'}}}/><div className="aurora aurora-one"/><div className="aurora aurora-two"/><Navbar active={page} onNavigate={setPage}/><Sidebar active={page} onNavigate={setPage}/><main className="relative z-10 mx-auto max-w-7xl px-5 pb-14 pt-24 lg:pl-28 lg:pt-28"><AnimatePresence mode="wait"><motion.div key={page} initial={{opacity:0,y:14}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-10}} transition={{duration:.32}}>{page === 'home' && <Home onNavigate={setPage}/>} {page === 'predict' && <Predict/>} {page === 'disaster' && <DisasterAnalysis/>} {page === 'about' && <About/>}</motion.div></AnimatePresence></main><button onClick={()=>window.scrollTo({top:0,behavior:'smooth'})} className="fixed bottom-6 right-5 z-30 grid h-12 w-12 place-items-center rounded-2xl gradient-button shadow-glow" title="Scroll to top"><ArrowUp size={19}/></button><Footer/></div>;
}
