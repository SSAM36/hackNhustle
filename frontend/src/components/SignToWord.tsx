import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { authAPI } from '../services/api';

export default function SignToWord() {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const [detectedSign, setDetectedSign] = useState('Ready');
  const [confidence, setConfidence] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  // Simple camera setup
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Camera access error:', error);
      }
    };

    startCamera();
  }, []);

  // Capture and process frames
  const captureFrame = async () => {
    if (!videoRef.current || isProcessing) return;
    
    setIsProcessing(true);
    
    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      
      ctx.drawImage(videoRef.current, 0, 0);
      
      canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        
        try {
          const response = await fetch('http://localhost:8000/recognize/image?top_k=1', {
            method: 'POST',
            body: formData
          });
          
          if (response.ok) {
            const data = await response.json();
            if (data.predictions && data.predictions.length > 0) {
              setDetectedSign(data.predictions[0].label.toUpperCase());
              setConfidence(Math.round(data.predictions[0].confidence * 100));
            }
          }
        } catch (error) {
          console.error('Recognition error:', error);
        }
      }, 'image/jpeg', 0.8);
    } catch (error) {
      console.error('Capture error:', error);
    } finally {
      setTimeout(() => setIsProcessing(false), 2000);
    }
  };

  // Auto capture every 3 seconds
  useEffect(() => {
    const interval = setInterval(captureFrame, 3000);
    return () => clearInterval(interval);
  }, [isProcessing]);

  const speakText = () => {
    if ('speechSynthesis' in window && detectedSign !== 'Ready') {
      const utterance = new SpeechSynthesisUtterance(detectedSign);
      utterance.rate = 0.8;
      utterance.pitch = 1;
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="flex h-full w-full flex-col group/design-root">
      {/* Video Container */}
      <div className="absolute inset-0 z-0 bg-black">
        <video
          ref={videoRef}
          className="absolute inset-0 h-full w-full object-cover"
          autoPlay
          playsInline
          muted
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-transparent to-black/60 pointer-events-none" />
      </div>

      {/* Top Navigation Bar */}
      <div className="relative z-20 flex items-center justify-between p-4 pt-12">
        <button 
          onClick={() => navigate(-1)} 
          className="flex size-12 shrink-0 items-center justify-center rounded-full bg-black/20 backdrop-blur-md border border-white/10 active:scale-95 transition-all hover:bg-black/40"
        >
          <span className="material-symbols-outlined text-white">arrow_back</span>
        </button>
        
        <div className="flex h-10 shrink-0 items-center justify-center gap-x-2 rounded-full bg-black/40 backdrop-blur-md border border-neon-teal/30 pl-3 pr-5 shadow-lg shadow-neon-teal/10">
          <span className="material-symbols-outlined text-neon-teal text-[20px]">{isProcessing ? 'hourglass_empty' : 'videocam'}</span>
          <p className="text-white text-sm font-medium tracking-wide">
            {isProcessing ? 'Processing...' : 'Camera Active'}
          </p>
        </div>
        
        <div className="w-12" />
      </div>

      {/* Central Focus Overlay */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center pointer-events-none pb-32">
        <div className="relative w-[70%] aspect-[3/4] max-w-sm rounded-xl border-2 border-dashed border-neon-teal/60 bg-neon-teal/5">
          {/* Corner Indicators */}
          <div className="absolute -top-[2px] -left-[2px] h-6 w-6 rounded-tl-xl border-l-4 border-t-4 border-neon-teal" />
          <div className="absolute -top-[2px] -right-[2px] h-6 w-6 rounded-tr-xl border-r-4 border-t-4 border-neon-teal" />
          <div className="absolute -bottom-[2px] -left-[2px] h-6 w-6 rounded-bl-xl border-l-4 border-b-4 border-neon-teal" />
          <div className="absolute -bottom-[2px] -right-[2px] h-6 w-6 rounded-br-xl border-r-4 border-b-4 border-neon-teal" />
          
          {/* Helper Text */}
          <div className="absolute -bottom-16 w-full flex justify-center">
            <p className="text-white/90 text-sm font-medium bg-black/30 backdrop-blur-sm px-4 py-2 rounded-lg border border-white/5">
              Show hands in frame
            </p>
          </div>
        </div>
      </div>

      {/* Result Bottom Sheet */}
      <div className="absolute bottom-0 left-0 z-30 w-full">
        <div className="flex flex-col bg-white dark:bg-[#1A1D1D] rounded-t-3xl shadow-[0_-8px_30px_rgba(0,0,0,0.5)] p-6 pb-8 transition-colors">
          {/* Handle */}
          <div className="flex w-full items-center justify-center pb-6">
            <div className="h-1.5 w-12 rounded-full bg-gray-300 dark:bg-gray-600" />
          </div>
          
          {/* Content Grid */}
          <div className="flex items-center justify-between gap-4 mb-8">
            {/* Text Result */}
            <div className="flex flex-col gap-1">
              <span className="text-xs font-bold uppercase tracking-widest text-gray-500 dark:text-gray-400">
                Status
              </span>
              <h1 className="text-5xl font-bold text-gray-900 dark:text-white tracking-tight">
                {detectedSign}
              </h1>
            </div>
            
            {/* Confidence Score Ring */}
            <div className="relative flex items-center justify-center size-[72px]">
              <svg className="h-full w-full rotate-[-90deg]" viewBox="0 0 36 36">
                <path 
                  className="text-gray-100 dark:text-gray-800" 
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="3" 
                />
                <path 
                  className="text-neon-teal drop-shadow-[0_0_4px_rgba(20,184,166,0.4)]" 
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeDasharray={`${confidence}, 100`} 
                  strokeLinecap="round" 
                  strokeWidth="3" 
                />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="text-sm font-bold text-gray-900 dark:text-white leading-none">
                  {confidence}%
                </span>
              </div>
            </div>
          </div>
          
          {/* Primary Action Button */}
          <button 
            onClick={speakText}
            disabled={detectedSign === 'Ready'}
            className="relative w-full overflow-hidden rounded-2xl bg-primary p-4 shadow-xl shadow-primary/20 active:scale-[0.99] transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="flex items-center justify-center gap-3">
              <div className="flex items-center justify-center size-8 rounded-full bg-white/20">
                <span className="material-symbols-outlined text-white text-[20px]">volume_up</span>
              </div>
              <span className="text-lg font-bold text-white tracking-wide">
                {detectedSign !== 'Ready' ? `Speak "${detectedSign}"` : 'Ready to detect signs'}
              </span>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}