import { useNavigate } from 'react-router-dom';
import { useState, useRef } from 'react';
// @ts-ignore
import CanvasDraw from 'react-canvas-draw';
import { parseAndValidate, validateWithDetails } from '../utils/HandwritingValidator';
import { LETTER_CHECKPOINTS } from '../utils/LetterCheckpoints';

export default function TracingPad({ debug = true }: { debug?: boolean }) {
  const navigate = useNavigate();
  const [targetLetter, setTargetLetter] = useState('A');
  const [progress, setProgress] = useState(1);
  const [showSuccess, setShowSuccess] = useState(false);
  const canvasRef = useRef<any>(null);

  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

  const handleClear = () => {
    if (canvasRef.current) {
      canvasRef.current.clear();
    }
  };

  const handleSubmit = () => {
    if (canvasRef.current) {
      const data = canvasRef.current.getSaveData();
      
      // Use the checkpoint validation algorithm
      const isValid = parseAndValidate(data, targetLetter);
      const details = validateWithDetails(data, targetLetter);
      
      console.log('Validation Details:', details);
      
      if (isValid) {
        setShowSuccess(true);
        setTimeout(() => {
          setShowSuccess(false);
          const currentIndex = letters.indexOf(targetLetter);
          if (currentIndex < letters.length - 1) {
            setTargetLetter(letters[currentIndex + 1]);
            setProgress(progress + 1);
            handleClear();
          }
        }, 1500);
      } else {
        alert(`Try again! Checkpoints hit: ${details.checkpointsHit}/${details.totalCheckpoints}`);
      }
    }
  };

  const handleUndo = () => {
    if (canvasRef.current) {
      canvasRef.current.undo();
    }
  };

  return (
    <div className="relative w-full max-w-md h-screen flex flex-col bg-background-light dark:bg-background-dark mx-auto shadow-2xl overflow-hidden">
      {/* Top App Bar */}
      <header className="flex items-center justify-between p-4 pt-6 z-10">
        <button onClick={() => navigate('/')} className="flex size-10 shrink-0 items-center justify-center rounded-full bg-white/50 dark:bg-black/20 hover:bg-white dark:hover:bg-black/40 transition-colors text-slate-700 dark:text-slate-200">
          <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>close</span>
        </button>
        <h1 className="text-slate-800 dark:text-white text-lg font-bold leading-tight tracking-[-0.015em] flex-1 text-center">
          Trace the Letter
        </h1>
        <div className="flex size-10 items-center justify-center">
          <div className="relative flex items-center justify-center size-10">
            <svg className="size-full -rotate-90" viewBox="0 0 36 36">
              <path className="text-slate-200 dark:text-slate-700" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="4" />
              <path className="text-primary" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeDasharray={`${(progress / 26) * 100}, 100`} strokeLinecap="round" strokeWidth="4" />
            </svg>
            <span className="absolute text-[10px] font-bold text-slate-600 dark:text-slate-300">{progress}/26</span>
          </div>
        </div>
      </header>

      {/* Main Canvas Area */}
      <main className="flex-1 flex flex-col items-center justify-center p-6 w-full relative">
        <div className="bg-white dark:bg-[#1a2c2b] w-full aspect-[4/5] max-h-[60vh] rounded-3xl shadow-soft relative flex items-center justify-center overflow-hidden border border-slate-100 dark:border-slate-800">
          {/* Background Grid */}
          <div className="absolute inset-0 opacity-30 pointer-events-none" style={{ backgroundImage: 'radial-gradient(#cbd5e1 1px, transparent 1px)', backgroundSize: '24px 24px' }} />
          
          {/* Letter Guide */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-0">
            <span className="text-[280px] font-bold text-slate-100 dark:text-slate-700/30 leading-none select-none" style={{ WebkitTextStroke: '2px #e2e8f0', color: 'transparent' }}>
              {targetLetter}
            </span>
          </div>

          {/* Canvas Layer */}
          <div className="absolute inset-0 z-10">
            <CanvasDraw
              ref={canvasRef}
              brushColor="#0F766E"
              brushRadius={10}
              lazyRadius={0}
              canvasWidth={400}
              canvasHeight={500}
              hideGrid
              backgroundColor="transparent"
              className="w-full h-full"
              onChange={() => {
                if (debug && canvasRef.current) {
                  const data = JSON.parse(canvasRef.current.getSaveData());
                  if (data.lines && data.lines.length > 0) {
                    const lastLine = data.lines[data.lines.length - 1];
                    if (lastLine.points && lastLine.points.length > 0) {
                      const lastPoint = lastLine.points[lastLine.points.length - 1];
                      const nearestCheckpoint = LETTER_CHECKPOINTS[targetLetter]?.[0] || { x: 50, y: 50 };
                      const dist = Math.sqrt(
                        Math.pow(lastPoint.x - nearestCheckpoint.x, 2) + 
                        Math.pow(lastPoint.y - nearestCheckpoint.y, 2)
                      );
                      console.log('User at:', lastPoint.x, lastPoint.y, '| Distance to Target:', dist.toFixed(2));
                    }
                  }
                }
              }}
            />
          </div>

          {/* Debug: Checkpoint Markers */}
          {debug && LETTER_CHECKPOINTS[targetLetter]?.map((checkpoint, index) => (
            <div
              key={index}
              className="absolute w-10 h-10 bg-red-500 rounded-full opacity-50 pointer-events-none z-20 flex items-center justify-center text-white text-xs font-bold"
              style={{
                left: `${checkpoint.x}px`,
                top: `${checkpoint.y}px`,
                transform: 'translate(-50%, -50%)'
              }}
            >
              {index + 1}
            </div>
          ))}

          {/* Success Animation */}
          {showSuccess && (
            <div className="absolute inset-0 flex items-center justify-center bg-primary/20 backdrop-blur-sm z-20 animate-[fadeIn_0.3s_ease-in-out]">
              <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-2xl flex flex-col items-center gap-3 animate-[scaleIn_0.3s_ease-out]">
                <span className="material-symbols-outlined text-primary text-6xl">check_circle</span>
                <p className="text-slate-800 dark:text-white text-xl font-bold">Great Job!</p>
              </div>
            </div>
          )}

          {/* Helper Text */}
          <p className="absolute bottom-6 text-slate-400 dark:text-slate-500 text-sm font-medium tracking-wide uppercase pointer-events-none z-5">
            Trace the letter
          </p>
        </div>
      </main>

      {/* Floating Tool Dock */}
      <div className="w-full px-6 pb-8 pt-2 z-20">
        <div className="bg-white dark:bg-[#1a2c2b] rounded-2xl shadow-float p-2 flex items-center justify-between border border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-1">
            <button onClick={handleUndo} aria-label="Undo last stroke" className="size-12 rounded-xl flex items-center justify-center text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-white/10 transition-colors active:scale-95">
              <span className="material-symbols-outlined">undo</span>
            </button>
            <button onClick={handleClear} aria-label="Clear canvas" className="size-12 rounded-xl flex items-center justify-center text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-white/10 transition-colors active:scale-95">
              <span className="material-symbols-outlined">ink_eraser</span>
            </button>
          </div>
          <div className="w-px h-8 bg-slate-200 dark:bg-slate-700 mx-2" />
          <button onClick={handleSubmit} className="flex-1 h-12 bg-primary hover:bg-[#0b5c55] active:bg-[#094d47] text-white rounded-xl flex items-center justify-center gap-2 font-bold shadow-lg shadow-primary/30 transition-all active:scale-[0.98]">
            <span className="material-symbols-outlined" style={{ fontWeight: 700 }}>check</span>
            <span>Submit</span>
          </button>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes scaleIn {
          from { transform: scale(0.8); opacity: 0; }
          to { transform: scale(1); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
