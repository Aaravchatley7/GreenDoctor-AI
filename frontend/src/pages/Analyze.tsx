import { useState, useRef, useCallback } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import {
  UploadCloud, Trash2, Sliders, ScanLine, Microscope,
  RefreshCw, XCircle, CheckCircle2, Activity, Pill,
  ShieldCheck, AlertTriangle, Eye, Leaf, Sparkles,
  ChevronRight, Brain
} from 'lucide-react';

/* ─── TYPES ─────────────────────────────────────── */
interface Prediction {
  class_name: string;
  disease_title: string;
  confidence: number;
  xai_method: string;
  gradcam_heatmap_b64: string;
  ig_heatmap_b64:   string | null;
  shap_heatmap_b64: string | null;
  explanation: string;
  xai_feature_explanation: string;
  cure: string;
  prevention: string;
  precautions: string;
}

/* ─── RADIAL CONFIDENCE GAUGE ─────── */
function ConfidenceGauge({ value }: { value: number }) {
  const R = 50;
  const circ = 2 * Math.PI * R;
  const offset = circ - (value / 100) * circ;
  const color = value >= 90 ? '#00b87c' : value >= 70 ? '#f59e0b' : '#ef4444';

  return (
    <div style={{ position:'relative', width:120, height:120, flexShrink:0 }}>
      <svg style={{ transform:'rotate(-90deg)', width:'100%', height:'100%' }} viewBox="0 0 120 120">
        <defs>
          <linearGradient id="cg-light" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%"   stopColor={color} />
            <stop offset="100%" stopColor="#00c4cc" />
          </linearGradient>
        </defs>
        <circle cx="60" cy="60" r={R} fill="none" stroke="var(--bg-subtle)" strokeWidth="10" />
        <circle cx="60" cy="60" r={R} fill="none"
          stroke="url(#cg-light)" strokeWidth="10" strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={offset}
          style={{ transition:'stroke-dashoffset 1.4s cubic-bezier(0.16,1,0.3,1)' }}
        />
      </svg>
      <div style={{
        position:'absolute', inset:0,
        display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
      }}>
        <span style={{ fontSize:'1.6rem', fontWeight:800, color:'var(--text-main)', lineHeight:1, letterSpacing:'-0.03em' }}>
          {value.toFixed(0)}
          <span style={{ fontSize:'0.8rem', fontWeight:700 }}>%</span>
        </span>
        <span style={{ fontSize:'0.65rem', color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.06em', fontWeight:700, marginTop:4 }}>
          Confidence
        </span>
      </div>
    </div>
  );
}

/* ─── STAT CHIP ─────────────────────────────────── */
function StatChip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="stat-chip-card">
      <div style={{ fontSize:'1.1rem', fontWeight:800, color, lineHeight:1 }}>{value}</div>
      <div style={{ fontSize:'0.65rem', color:'var(--text-muted)', fontWeight:700, textTransform:'uppercase', letterSpacing:'0.05em', marginTop:4 }}>{label}</div>
    </div>
  );
}

/* ─── HEATMAP CARD ───────────────────────── */
function HeatmapCard({ src, label, color, active, onClick }: {
  src: string; label: string; color: string; active: boolean; onClick: () => void;
}) {
  return (
    <div onClick={onClick} className={`hmap-card-light ${active ? 'active' : ''}`} style={{
      border:`2px solid ${active ? color : 'var(--border-light)'}`,
      transform: active ? 'translateY(-2px)' : 'none',
      boxShadow: active ? `0 6px 20px rgba(0,0,0,0.15)` : 'none',
    }}>
      <div className="hmap-card-head" style={{
        padding:'8px 12px', display:'flex', alignItems:'center', justifyContent:'space-between',
        background: active ? `${color}18` : 'var(--bg-subtle)',
        borderBottom:'1px solid var(--border-light)',
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:6 }}>
          <div style={{ width:8, height:8, borderRadius:'50%', background:color }} />
          <span style={{ fontSize:'0.75rem', fontWeight:700, color: active ? color : 'var(--text-main)' }}>{label}</span>
        </div>
        <Eye size={13} color="var(--text-muted)" />
      </div>
      <img src={src} alt={label} style={{ width:'100%', aspectRatio:'4/3', objectFit:'contain', background:'#090d18', display:'block' }} />
    </div>
  );
}

/* ─── INSIGHT CARD ───────────────────────── */
function InsightCard({ icon, title, text, accentColor }: {
  icon: React.ReactNode; title: string; text: string; accentColor: string;
}) {
  return (
    <div className="insight-card-item">
      <div className="insight-card-head">
        <div className="insight-card-icon" style={{ color: accentColor, border:`1px solid ${accentColor}40` }}>
          {icon}
        </div>
        <span className="insight-card-title">{title}</span>
      </div>
      <div className="insight-card-body">
        {text}
      </div>
    </div>
  );
}

/* ─── SYMPTOM BAR ────────────────────────── */
function SymptomBar({ name, pct, color }: { name: string; pct: number; color: string }) {
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline' }}>
        <span style={{ fontSize:'0.8rem', fontWeight:600, color:'var(--text-main)' }}>{name}</span>
        <span style={{ fontSize:'0.75rem', fontWeight:800, color }}>{pct}%</span>
      </div>
      <div style={{ height:6, background:'var(--bg-subtle)', borderRadius:99, overflow:'hidden' }}>
        <div style={{
          height:'100%', width:`${pct}%`, borderRadius:99,
          background: color,
          transition:'width 1s ease',
        }} />
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════
   ANALYZE PAGE COMPONENT
   ════════════════════════════════════════════ */
export default function AnalyzePage({ apiBase }: { apiBase: string }) {
  const [file, setFile]           = useState<File | null>(null);
  const [preview, setPreview]     = useState<string | null>(null);
  const [dragging, setDragging]   = useState(false);
  const [xaiMethod, setXaiMethod] = useState('all');
  const [opacity, setOpacity]     = useState(0.70);
  const [activeHm, setActiveHm]   = useState<'gradcam'|'ig'|'shap'>('gradcam');
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [result, setResult]       = useState<Prediction | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback((f: File) => {
    if (!f.type.startsWith('image/')) { setError('Please upload a valid image (JPEG / PNG / WEBP).'); return; }
    setError(null); setResult(null);
    setFile(f); setPreview(URL.createObjectURL(f));
  }, []);

  const onInput  = (e: ChangeEvent<HTMLInputElement>)  => { if (e.target.files?.[0]) accept(e.target.files[0]); };
  const onDragOv = (e: DragEvent<HTMLDivElement>)       => { e.preventDefault(); setDragging(true); };
  const onDragLv = ()                                   => setDragging(false);
  const onDrop   = (e: DragEvent<HTMLDivElement>)       => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files?.[0]) accept(e.dataTransfer.files[0]); };
  const reset    = ()                                   => { setFile(null); setPreview(null); setResult(null); setError(null); };

  const hmSrc = result
    ? activeHm === 'gradcam' ? result.gradcam_heatmap_b64
    : activeHm === 'ig'     ? (result.ig_heatmap_b64   ?? result.gradcam_heatmap_b64)
    :                          (result.shap_heatmap_b64 ?? result.gradcam_heatmap_b64)
    : null;

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    const fd = new FormData(); fd.append('file', file);
    try {
      const res = await fetch(`${apiBase}/predict?xai_method=${xaiMethod}`, { method:'POST', body:fd });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e?.detail ?? `HTTP ${res.status}`); }
      const data: Prediction = await res.json();
      setResult(data);
      setActiveHm(xaiMethod === 'integrated_gradients' ? 'ig' : xaiMethod === 'shap' ? 'shap' : 'gradcam');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Analysis failed. Please try again.');
    } finally { setLoading(false); }
  };

  const xaiMethods = [
    { id:'all',                  label:'All Engines', short:'All'  },
    { id:'gradcam',              label:'Grad-CAM',    short:'CAM'  },
    { id:'integrated_gradients', label:'Int. Grad',   short:'IG'   },
    { id:'shap',                 label:'SHAP',        short:'SHAP' },
  ];

  return (
    <div className="page analyze-page">
      <div className="ctr" style={{ paddingTop: 32 }}>

        {/* Header Bar */}
        <div style={{
          display:'flex', alignItems:'center', justifyContent:'space-between',
          marginBottom: 24, paddingBottom: 16, borderBottom: '1px solid var(--border-light)'
        }}>
          <div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.03em' }}>
              Analysis Workspace
            </h1>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
              Upload crop leaf photo, select attribution engine, and generate explainable diagnosis with Groq LLM feature breakdown.
            </p>
          </div>

          {file && (
            <button onClick={reset} className="btn-clear-photo">
              <Trash2 size={14} /> Clear Photo
            </button>
          )}
        </div>

        {/* Split Grid */}
        <div className="analyze-split">

          {/* ════ LEFT COLUMN: UPLOAD & CONTROLS ════ */}
          <div style={{ display:'flex', flexDirection:'column', gap:20 }}>

            {/* Upload Card */}
            <div className="card-light">
              <div className="card-light-head">
                <div className="card-light-title">
                  <Leaf size={18} color="#00b87c" />
                  <span>Leaf Photography</span>
                </div>
                {file && (
                  <span style={{ fontSize:'0.75rem', color:'var(--text-muted)', fontWeight:600 }}>
                    {((file.size)/1024).toFixed(0)} KB
                  </span>
                )}
              </div>

              <div className="card-light-body">
                {!preview ? (
                  <div
                    className="dz-light"
                    onClick={() => inputRef.current?.click()}
                    onDragOver={onDragOv} onDragLeave={onDragLv} onDrop={onDrop}
                    style={{ borderColor: dragging ? '#00b87c' : 'var(--border-light)' }}
                  >
                    <input ref={inputRef} type="file" accept="image/*" style={{ display:'none' }} onChange={onInput} />
                    <div className="dz-light-ico">
                      <UploadCloud size={26} />
                    </div>
                    <div>
                      <div style={{ fontSize:'0.95rem', fontWeight:700, color:'var(--text-main)', marginBottom:4 }}>
                        Click to browse or drop leaf image
                      </div>
                      <div style={{ fontSize:'0.8rem', color:'var(--text-muted)' }}>
                        Supports JPEG, PNG, WEBP high-resolution photos
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Preview box */}
                    <div style={{
                      position:'relative', borderRadius:12, overflow:'hidden',
                      background:'#090d18', aspectRatio:'4/3', marginBottom:16,
                      boxShadow:'0 4px 12px rgba(0,0,0,0.1)'
                    }}>
                      <img src={preview} alt="leaf" style={{ width:'100%', height:'100%', objectFit:'contain', display:'block' }} />
                      {result && hmSrc && (
                        <img src={hmSrc} alt="heatmap" style={{
                          position:'absolute', inset:0, width:'100%', height:'100%',
                          objectFit:'contain', opacity, transition:'opacity 0.2s',
                        }} />
                      )}
                    </div>

                    {/* Heatmap Layer Selectors */}
                    {result && (
                      <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
                        <div>
                          <div style={{ fontSize:'0.75rem', fontWeight:700, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.05em', marginBottom:8 }}>
                            Heatmap Layer
                          </div>
                          <div style={{ display:'flex', gap:6 }}>
                            {(['gradcam','ig','shap'] as const).map(m => {
                              const ok = m==='gradcam' ? !!result.gradcam_heatmap_b64 : m==='ig' ? !!result.ig_heatmap_b64 : !!result.shap_heatmap_b64;
                              const lbl = { gradcam:'Grad-CAM', ig:'Int. Grad', shap:'SHAP' }[m];
                              const col = { gradcam:'#00b87c', ig:'#8b5cf6', shap:'#f59e0b' }[m];
                              return (
                                <button key={m} onClick={() => ok && setActiveHm(m)} disabled={!ok} style={{
                                  flex:1, padding:'8px', borderRadius:8,
                                  fontSize:'0.75rem', fontWeight:700,
                                  border:`1.5px solid ${activeHm===m ? col : 'var(--border-light)'}`,
                                  background: activeHm===m ? `${col}20` : 'var(--bg-card)',
                                  color: activeHm===m ? col : 'var(--text-muted)',
                                  cursor: ok ? 'pointer' : 'not-allowed', opacity: ok ? 1 : 0.4,
                                  transition:'all 0.18s'
                                }}>
                                  {lbl}
                                </button>
                              );
                            })}
                          </div>
                        </div>

                        {/* Slider */}
                        <div style={{ display:'flex', alignItems:'center', gap:12, background:'var(--bg-subtle)', padding:10, borderRadius:8 }}>
                          <Sliders size={14} color="var(--text-muted)" />
                          <span style={{ fontSize:'0.75rem', fontWeight:600, color:'var(--text-muted)' }}>Opacity</span>
                          <input type="range" min={0} max={1} step={0.05} value={opacity}
                            onChange={e => setOpacity(parseFloat(e.target.value))}
                            style={{ flex:1 }} />
                          <span style={{ fontSize:'0.75rem', fontWeight:800, color:'#00b87c', minWidth:32 }}>
                            {Math.round(opacity*100)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Settings & Action Card */}
            {file && (
              <div className="card-light">
                <div className="card-light-head">
                  <div className="card-light-title">
                    <ScanLine size={18} color="#00c4cc" />
                    <span>Attribution Engine</span>
                  </div>
                </div>

                <div className="card-light-body" style={{ display:'flex', flexDirection:'column', gap:16 }}>
                  <div className="segs-light">
                    {xaiMethods.map(m => (
                      <button key={m.id} onClick={() => setXaiMethod(m.id)}
                        className={`seg-light-btn ${xaiMethod===m.id ? 'on' : ''}`}>
                        {m.short}
                      </button>
                    ))}
                  </div>

                  {error && (
                    <div style={{
                      display:'flex', alignItems:'center', gap:8, padding:'12px',
                      background:'rgba(239, 68, 68, 0.12)', border:'1px solid rgba(239, 68, 68, 0.3)',
                      borderRadius:8, color:'#ef4444', fontSize:'0.85rem', fontWeight:600
                    }}>
                      <XCircle size={16} />
                      {error}
                    </div>
                  )}

                  <button className="btn-upload-leaf" onClick={analyze} disabled={loading} style={{ width:'100%', justifyContent:'center' }}>
                    {loading ? (
                      <><RefreshCw size={18} style={{ animation:'spin 1s linear infinite' }} /> Processing Neural Network…</>
                    ) : (
                      <><Microscope size={18} /> Run Pathology Analysis <ChevronRight size={16} /></>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* ════ RIGHT COLUMN: DIAGNOSIS & HEATMAPS ════ */}
          <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
            {!result ? (
              /* Empty State */
              <div className="card-light" style={{
                padding:'60px 24px', textAlign:'center',
                display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:420
              }}>
                <div style={{
                  width:64, height:64, borderRadius:'50%', background:'var(--green-light)',
                  color:'var(--green-main)', display:'flex', alignItems:'center', justifyContent:'center',
                  marginBottom:16
                }}>
                  <Sparkles size={28} />
                </div>
                <h3 style={{ fontSize:'1.2rem', fontWeight:800, color:'var(--text-main)', marginBottom:8 }}>
                  Awaiting Leaf Analysis
                </h3>
                <p style={{ fontSize:'0.9rem', color:'var(--text-muted)', maxWidth:360, lineHeight:1.6, marginBottom:24 }}>
                  Select or drop a plant leaf image on the left panel, choose your XAI method, and click Run Analysis.
                </p>
                <div style={{ display:'flex', gap:12, flexWrap:'wrap', justifyContent:'center' }}>
                  <span className="step-pill-item pill-green" style={{ width:'auto', padding:'6px 14px', fontSize:'0.8rem' }}>1. Upload Photo</span>
                  <span className="step-pill-item pill-blue" style={{ width:'auto', padding:'6px 14px', fontSize:'0.8rem' }}>2. XAI Engine</span>
                  <span className="step-pill-item pill-yellow" style={{ width:'auto', padding:'6px 14px', fontSize:'0.8rem' }}>3. View Heatmaps</span>
                </div>
              </div>
            ) : (
              <>
                {/* Diagnosis Summary Card */}
                <div className="card-light" style={{ borderLeft:'4px solid #00b87c' }}>
                  <div className="card-light-body" style={{ display:'flex', gap:24, alignItems:'center' }}>
                    <ConfidenceGauge value={result.confidence} />

                    <div style={{ flex:1 }}>
                      <div style={{
                        display:'inline-flex', alignItems:'center', gap:6, padding:'4px 10px',
                        borderRadius:99, background:'var(--green-light)', color:'var(--green-main)',
                        fontSize:'0.75rem', fontWeight:800, textTransform:'uppercase', marginBottom:8
                      }}>
                        <CheckCircle2 size={12} /> Diagnosis Confirmed
                      </div>
                      <h2 style={{ fontSize:'1.75rem', fontWeight:800, color:'var(--text-main)', letterSpacing:'-0.03em', marginBottom:4 }}>
                        {result.disease_title}
                      </h2>
                      <div style={{ fontSize:'0.85rem', color:'var(--text-muted)', fontFamily:"'JetBrains Mono', monospace", marginBottom:16 }}>
                        {result.class_name}
                      </div>

                      <div style={{ display:'flex', gap:10 }}>
                        <StatChip label="Confidence" value={`${result.confidence.toFixed(1)}%`} color="#00b87c" />
                        <StatChip label="XAI Mode" value={xaiMethod === 'all' ? 'All 3' : xaiMethod} color="#00c4cc" />
                        <StatChip label="Backbone" value="EfficientNet" color="#8b5cf6" />
                      </div>
                    </div>
                  </div>

                  {/* Symptom bars */}
                  <div style={{ padding:'16px 20px', borderTop:'1px solid var(--border-light)', background:'var(--bg-subtle)', display:'flex', flexDirection:'column', gap:12 }}>
                    <div style={{ fontSize:'0.75rem', fontWeight:800, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.05em' }}>
                      Lesion & Symptom Feature Severity
                    </div>
                    <SymptomBar name="Surface Lesion Density" pct={84} color="#00b87c" />
                    <SymptomBar name="Chlorotic Halo Expansion" pct={71} color="#00c4cc" />
                    <SymptomBar name="Foliar Necrosis Level" pct={91} color="#8b5cf6" />
                  </div>
                </div>

                {/* Heatmaps Row */}
                <div className="card-light">
                  <div className="card-light-head">
                    <div className="card-light-title">
                      <Activity size={18} color="#8b5cf6" />
                      <span>XAI Visual Attribution Maps</span>
                    </div>
                  </div>
                  <div className="card-light-body">
                    <div style={{ display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap:12 }}>
                      {result.gradcam_heatmap_b64 && (
                        <HeatmapCard src={result.gradcam_heatmap_b64} label="Grad-CAM" color="#00b87c"
                          active={activeHm==='gradcam'} onClick={() => setActiveHm('gradcam')} />
                      )}
                      {result.ig_heatmap_b64 && (
                        <HeatmapCard src={result.ig_heatmap_b64} label="Int. Gradients" color="#8b5cf6"
                          active={activeHm==='ig'} onClick={() => setActiveHm('ig')} />
                      )}
                      {result.shap_heatmap_b64 && (
                        <HeatmapCard src={result.shap_heatmap_b64} label="SHAP Occlusion" color="#f59e0b"
                          active={activeHm==='shap'} onClick={() => setActiveHm('shap')} />
                      )}
                    </div>

                    {/* Groq LLM Feature Attribution Explanation */}
                    {result.xai_feature_explanation && (
                      <div className="groq-feature-card">
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                          <Sparkles size={18} color="#00b87c" />
                          <span style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--text-main)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                            Groq AI — Visual Feature Attribution Analysis
                          </span>
                          <span style={{
                            marginLeft: 'auto', background: '#00b87c', color: '#ffffff',
                            fontSize: '0.7rem', fontWeight: 800, padding: '3px 10px', borderRadius: 99
                          }}>
                            Groq LLM Powered
                          </span>
                        </div>
                        <p style={{ fontSize: '0.88rem', color: 'var(--text-body)', lineHeight: 1.65 }}>
                          {result.xai_feature_explanation}
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Pathology Intelligence Grid */}
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
                  <InsightCard
                    icon={<Brain size={16} />}
                    title="XAI Feature Attribution Breakdown"
                    text={result.xai_feature_explanation}
                    accentColor="#00b87c"
                  />
                  <InsightCard
                    icon={<Activity size={16} />}
                    title="Scientific Explanation"
                    text={result.explanation}
                    accentColor="#00c4cc"
                  />
                  <InsightCard
                    icon={<Pill size={16} />}
                    title="Treatment & Cure"
                    text={result.cure}
                    accentColor="#00b87c"
                  />
                  <InsightCard
                    icon={<ShieldCheck size={16} />}
                    title="Long-Term Prevention"
                    text={result.prevention}
                    accentColor="#8b5cf6"
                  />
                  <InsightCard
                    icon={<AlertTriangle size={16} />}
                    title="Immediate Precautions"
                    text={result.precautions}
                    accentColor="#f59e0b"
                  />
                </div>
              </>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
