import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { FileImage, UploadCloud } from 'lucide-react';

export default function UploadCard({ onFile, file }) {
  const input = useRef(); const [dragging, setDragging] = useState(false); const [preview, setPreview] = useState();
  useEffect(() => { if (!file) { setPreview(); return; } const url = URL.createObjectURL(file); setPreview(url); return () => URL.revokeObjectURL(url); }, [file]);
  const choose = files => files?.[0] && onFile(files[0]);
  return <motion.div whileHover={{scale:1.005}} animate={{boxShadow:dragging?'0 0 55px rgba(168,85,247,.45)':'0 0 0 rgba(0,0,0,0)'}} onDragOver={event => {event.preventDefault();setDragging(true);}} onDragLeave={() => setDragging(false)} onDrop={event => {event.preventDefault();setDragging(false);choose(event.dataTransfer.files);}} onClick={() => input.current.click()} className={`cursor-pointer rounded-3xl border-2 border-dashed p-8 text-center transition ${dragging ? 'border-orange-300 bg-orange-300/10' : 'border-violet-300/35 bg-violet-300/5 hover:bg-violet-300/10'}`}><input ref={input} type="file" accept=".tif,.tiff,image/*" hidden onChange={event => choose(event.target.files)}/>{preview ? <img src={preview} alt="Selected satellite preview" className="mx-auto mb-5 h-40 w-full max-w-sm rounded-2xl object-cover"/> : <UploadCloud className="mx-auto mb-4 text-violet-300" size={40}/>}<p className="font-semibold">Drag & drop your satellite image</p><p className="mt-2 text-sm text-slate-400">or <span className="text-orange-300">browse file</span> · TIFF, GeoTIFF, PNG, JPG</p>{file && <div className="mt-5 inline-flex items-center gap-2 rounded-xl bg-white/10 px-3 py-2 text-xs text-slate-200"><FileImage size={15}/>{file.name}</div>}</motion.div>;
}
