import React, { useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, Zap, TrendingDown, ArrowDown } from "lucide-react";

interface HeroPreviewProps {
  onSelectPreset: (type: string) => void;
}

export default function HeroPreview({ onSelectPreset }: HeroPreviewProps) {
  const [query, setQuery] = useState("Explain dryland PM-KISAN eligibility...");
  const [metrics, setMetrics] = useState({
    input: "184 Tokens",
    compressed: "52 Tokens",
    saved: "⚡ 72% Saved",
    speedup: "+38ms faster"
  });

  const handleQuickPreset = (type: string) => {
    onSelectPreset(type);
    if (type === "civic") {
      setQuery("Who is eligible for PM-KISAN dryland farming?");
      setMetrics({
        input: "155 Tokens",
        compressed: "46 Tokens",
        saved: "⚡ 70% Saved",
        speedup: "+34ms faster"
      });
    } else if (type === "tech") {
      setQuery("How does write-through caching affect database commit latency?");
      setMetrics({
        input: "172 Tokens",
        compressed: "48 Tokens",
        saved: "⚡ 72% Saved",
        speedup: "+38ms faster"
      });
    }
    
    // Smooth scroll down to interactive playground
    setTimeout(() => {
      const element = document.getElementById("demo");
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    }, 150);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="relative rounded-2xl border border-emerald-500/30 bg-zinc-900/60 backdrop-blur-xl p-6 shadow-[0_24px_60px_-30px_rgba(16,185,129,0.4)] tech-border glow-orange"
    >
      {/* Top Bar pill badge */}
      <div className="flex items-center justify-between pb-4 border-b border-zinc-800/80 mb-4">
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
          <Sparkles className="w-3 h-3 text-emerald-400 animate-pulse" />
          Live Compression Engine
        </span>
        <span className="text-[10px] font-mono text-zinc-500 uppercase">Interactive Preview</span>
      </div>

      {/* Interactive Box */}
      <div className="space-y-4 font-sans">
        <div className="space-y-2">
          <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider font-mono">Try Quick Presets</label>
          <div className="flex flex-wrap gap-2">
            <button 
              type="button"
              onClick={() => handleQuickPreset("civic")}
              className="text-[10px] px-3 py-1.5 rounded-lg border border-zinc-800 hover:border-emerald-500/40 hover:bg-emerald-500/5 text-zinc-300 font-sans font-medium transition-all"
            >
              Analyze Civic Claim
            </button>
            <button 
              type="button"
              onClick={() => handleQuickPreset("tech")}
              className="text-[10px] px-3 py-1.5 rounded-lg border border-zinc-800 hover:border-emerald-500/40 hover:bg-emerald-500/5 text-zinc-300 font-sans font-medium transition-all"
            >
              Compress System Architecture
            </button>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider font-mono">Sample Query</label>
          <input 
            type="text" 
            value={query}
            readOnly
            className="w-full bg-black border border-zinc-800 rounded-lg px-3 py-2 text-xs text-zinc-100 font-sans focus:outline-none"
          />
        </div>

        {/* Real-Time Visual Metric Output */}
        <div className="grid grid-cols-3 gap-2 bg-black border border-zinc-800/80 rounded-xl p-3 text-center font-mono">
          <div className="min-w-0">
            <div className="text-[9px] uppercase tracking-wider text-zinc-500">Input Context</div>
            <div className="text-xs font-bold text-zinc-300 mt-1 truncate">{metrics.input}</div>
          </div>
          <div className="min-w-0 border-x border-zinc-800/80">
            <div className="text-[9px] uppercase tracking-wider text-zinc-500">TokenDiet</div>
            <div className="text-xs font-bold text-emerald-400 mt-1 truncate">{metrics.compressed}</div>
            <div className="text-[8px] text-emerald-500 mt-0.5">{metrics.saved}</div>
          </div>
          <div className="min-w-0">
            <div className="text-[9px] uppercase tracking-wider text-zinc-500">Est. TTFT</div>
            <div className="text-xs font-bold text-emerald-400 mt-1 truncate">{metrics.speedup}</div>
            <div className="text-[8px] text-zinc-500 mt-0.5">Latency Cut</div>
          </div>
        </div>

        {/* Action button */}
        <button 
          type="button"
          onClick={() => {
            const element = document.getElementById("demo");
            if (element) {
              element.scrollIntoView({ behavior: "smooth" });
            }
          }}
          className="w-full flex items-center justify-center gap-2 text-[11px] font-bold text-emerald-400 hover:text-emerald-300 transition-colors uppercase tracking-wider pt-2 group"
        >
          <span>Try Full Interactive Sandbox</span>
          <ArrowDown className="w-3.5 h-3.5 animate-bounce group-hover:translate-y-0.5 transition-transform text-emerald-400" />
        </button>
      </div>
    </motion.div>
  );
}
