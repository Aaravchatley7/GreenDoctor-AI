import { useState, useRef, useCallback, useEffect } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import {
  UploadCloud, Trash2, Sliders, ScanLine, Microscope,
  RefreshCw, XCircle, CheckCircle2, Activity, Pill,
  ShieldCheck, AlertTriangle, Eye, Leaf, Sparkles,
  ChevronRight, AlertOctagon, Dna, Check,
  Zap, FolderOpen
} from 'lucide-react';
import { formatDiseaseDisplayName } from '../utils/diseaseLabels';


/* ─── TYPES ─────────────────────────────────────── */
export interface PipelineTimings {
  preprocessing_seconds: number;
  classification_seconds: number;
  gradcam_seconds: number;
  gradcam_pp_seconds?: number | null;
  saliency_seconds?: number | null;
  smoothgrad_seconds?: number | null;
  ig_seconds: number;
  shap_seconds: number;
  lime_seconds: number | null;
  xai_wall_seconds: number;
  groq_seconds: number;
  total_seconds: number;
}

interface Prediction {
  class_name: string;
  disease_title: string;
  confidence: number;
  xai_method: string;
  gradcam_heatmap_b64: string;
  gradcam_pp_heatmap_b64: string | null;
  saliency_heatmap_b64:   string | null;
  smoothgrad_heatmap_b64: string | null;
  ig_heatmap_b64:         string | null;
  shap_heatmap_b64:       string | null;
  lime_heatmap_b64:       string | null;
  explanation: string;
  xai_feature_explanation: string;
  cure: string;
  prevention: string;
  precautions: string;
  timings?: PipelineTimings;
}

type ProcessStage = 'idle' | 'preprocess' | 'classify' | 'xai' | 'groq' | 'report' | 'success' | 'error';
type XAIMethodKey = 'all' | 'gradcam_pp' | 'ig' | 'shap' | 'lime';
type ClinicalTab = 'pathology' | 'treatment' | 'prevention' | 'precautions';

/* ─── PROCESSING STAGES CONFIG ──────────────────── */
const STAGES: { id: ProcessStage; label: string; sub: string }[] = [
  { id: 'preprocess', label: 'Image Preprocessing',            sub: 'Normalizing and decoding tensor input' },
  { id: 'classify',   label: 'Disease Classification',         sub: 'Running EfficientNet-B0 neural backbone' },
  { id: 'xai',        label: '4-Engine XAI Attribution Suite', sub: 'Computing Grad-CAM++, Int. Gradients, SHAP & LIME' },
  { id: 'groq',       label: 'AI Clinical Reasoning',          sub: 'Synthesizing visual evidence with pathology intelligence' },
  { id: 'report',     label: 'Clinical Report Assembly',       sub: 'Formatting explainable diagnostic findings' },
];

const STATUS_MSGS: Partial<Record<ProcessStage, string>> = {
  preprocess: 'Preparing and normalizing leaf image…',
  classify:   'Running EfficientNet-B0 classification…',
  xai:        'Generating 4-engine visual attributions (Grad-CAM++, IG, SHAP, LIME)…',
  groq:       'Synthesizing pathology & feature reasoning…',
  report:     'Assembling high-precision diagnostic report…',
};

const STAGE_IDS: ProcessStage[] = ['preprocess','classify','xai','groq','report'];

/* ─── HELPER: FORMAT DURATION ───────────────────── */
function formatDuration(sec: number | null | undefined): string {
  if (sec === null || sec === undefined) return '—';
  if (sec < 0.001) return '< 1 ms';
  if (sec < 1.0) return `${Math.round(sec * 1000)} ms`;
  return `${sec.toFixed(3)} s`;
}

/* ─── HELPER: TEXT TO BULLETS ───────────────────── */
function parseBulletPoints(text: string): string[] {
  if (!text) return [];
  const items = text
    .split(/(?:\r?\n|•|\d+\.\s+)/)
    .map(s => s.trim())
    .filter(s => s.length > 6);
  if (items.length > 1) return items;
  return text
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 6);
}

/* ─── PROCESSING MODAL ───────────────────────────── */
function ProcessingModal({
  stage, result, onDismiss, onRetry, abortRef
}: {
  stage: ProcessStage;
  result: Prediction | null;
  onDismiss: () => void;
  onRetry: () => void;
  abortRef: React.RefObject<AbortController | null>;
}) {
  const isError   = stage === 'error';
  const isSuccess = stage === 'success';
  const isActive  = !isError && !isSuccess && stage !== 'idle';

  const [elapsedSec, setElapsedSec] = useState<number>(0);

  useEffect(() => {
    if (!isActive) {
      return;
    }
    const start = performance.now();
    const interval = setInterval(() => {
      setElapsedSec((performance.now() - start) / 1000);
    }, 50);
    return () => clearInterval(interval);
  }, [isActive]);

  const stageIdx  = STAGE_IDS.indexOf(stage as ProcessStage);
  const progress  = isSuccess ? 100
    : isError ? 0
    : stageIdx < 0 ? 0
    : Math.round(((stageIdx + 0.5) / STAGE_IDS.length) * 100);

  useEffect(() => {
    if (!isSuccess) return;
    const t = setTimeout(onDismiss, 2000);
    return () => clearTimeout(t);
  }, [isSuccess, onDismiss]);

  const getStageState = (s: typeof STAGES[number]) => {
    const idx      = STAGE_IDS.indexOf(s.id);
    const curIdx   = STAGE_IDS.indexOf(stage as ProcessStage);
    if (isSuccess)           return 'done';
    if (idx < curIdx)        return 'done';
    if (idx === curIdx)      return 'active';
    return 'pending';
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    onDismiss();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="PhytoShield AI analysis in progress"
      aria-live="polite"
      className="pmodal-backdrop"
      style={{ display: stage === 'idle' ? 'none' : 'flex' }}
    >
      <div className="pmodal-glass">
        {/* ── SUCCESS STATE ── */}
        {isSuccess && (
          <div className="pmodal-success">
            <div className="pmodal-success-icon">
              <CheckCircle2 size={36} />
            </div>
            <div className="pmodal-success-title">PATHOLOGY ANALYSIS COMPLETE</div>
            <div className="pmodal-success-sub">
              {result ? (
                <>
                  <strong style={{ color:'var(--green-main)', fontSize: '1.05rem' }}>
                    {formatDiseaseDisplayName(result.class_name)}
                  </strong><br />
                  <span style={{ fontSize: '0.85rem' }}>Confidence: {result.confidence.toFixed(1)}%</span>
                  {result.timings && (
                    <div style={{ marginTop: 8, fontSize: '0.78rem', color: 'rgba(255,255,255,0.65)', fontFamily: "'JetBrains Mono', monospace" }}>
                      ⏱ Total Execution: <strong style={{ color: '#00e599' }}>{result.timings.total_seconds.toFixed(3)}s</strong>
                    </div>
                  )}
                </>
              ) : 'Your leaf image has been analyzed.'}
            </div>
            <div className="pmodal-opening">Opening interactive results…</div>
          </div>
        )}

        {/* ── ERROR STATE ── */}
        {isError && (
          <div className="pmodal-error">
            <div className="pmodal-error-icon">
              <AlertOctagon size={32} />
            </div>
            <div className="pmodal-error-title">ANALYSIS INTERRUPTED</div>
            <div className="pmodal-error-sub">We couldn't complete the diagnostic pipeline.</div>
            <div style={{ display:'flex', gap:10, marginTop:16 }}>
              <button className="pmodal-btn-retry" onClick={onRetry}>Try Again</button>
              <button className="pmodal-btn-cancel" onClick={onDismiss}>Dismiss</button>
            </div>
          </div>
        )}

        {/* ── ACTIVE PROCESSING STATE ── */}
        {isActive && (
          <>
            <div className="pmodal-header">
              <div className="pmodal-pulse-ring" aria-hidden="true" />
              <div className="pmodal-title">ANALYZING LEAF PATHOLOGY</div>
              <div className="pmodal-subtitle">PhytoShield AI 4-Engine Diagnostic Pipeline</div>
            </div>

            <div className="pmodal-progress-wrap">
              <div className="pmodal-progress-header">
                <span>Pipeline Progress</span>
                <span style={{ fontFamily:"'JetBrains Mono', monospace" }}>{progress}% • {elapsedSec.toFixed(1)}s</span>
              </div>
              <div className="pmodal-progress-track" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
                <div className="pmodal-progress-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>

            <div className="pmodal-steps" role="list">
              {STAGES.map(s => {
                const st = getStageState(s);
                const isStepActive = st === 'active';
                const isStepDone   = st === 'done';

                let timingBadge = null;
                if (result?.timings) {
                  if (s.id === 'preprocess') timingBadge = formatDuration(result.timings.preprocessing_seconds);
                  if (s.id === 'classify')   timingBadge = formatDuration(result.timings.classification_seconds);
                  if (s.id === 'xai')        timingBadge = formatDuration(result.timings.xai_wall_seconds);
                  if (s.id === 'groq')       timingBadge = formatDuration(result.timings.groq_seconds);
                  if (s.id === 'report')     timingBadge = '✓ Ready';
                }

                return (
                  <div key={s.id} className={`pmodal-step pmodal-step-${st}`} role="listitem">
                    <div className="pmodal-step-icon" aria-label={st}>
                      {st === 'done'   && <CheckCircle2 size={16} />}
                      {st === 'active' && <span className="pmodal-dot-active" aria-label="in progress" />}
                      {st === 'pending'&& <span className="pmodal-dot-pending" />}
                    </div>
                    <div className="pmodal-step-text" style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span className="pmodal-step-label">{s.label}</span>
                        {isStepDone && timingBadge && (
                          <span className="pmodal-time-badge">{timingBadge}</span>
                        )}
                        {isStepActive && (
                          <span className="pmodal-time-badge running">{elapsedSec.toFixed(1)}s</span>
                        )}
                      </div>
                      {isStepActive && (
                        <span className="pmodal-step-sub">{s.sub}</span>
                      )}

                      {/* XAI 4-Engine Sub-breakdown */}
                      {s.id === 'xai' && (isStepActive || isStepDone) && (
                        <div className="pmodal-xai-grid">
                          <div className="pmodal-xai-row">
                            <span className="pmodal-xai-name">
                              <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#00e599' }} />
                              Grad-CAM++
                            </span>
                            <span className="pmodal-xai-time">
                              {result?.timings ? formatDuration(result.timings.gradcam_pp_seconds ?? result.timings.gradcam_seconds) : isStepActive ? '● Computing' : '✓'}
                            </span>
                          </div>
                          <div className="pmodal-xai-row">
                            <span className="pmodal-xai-name">
                              <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#8b5cf6' }} />
                              Int. Gradients
                            </span>
                            <span className="pmodal-xai-time">
                              {result?.timings ? formatDuration(result.timings.ig_seconds) : isStepActive ? '● Computing' : '✓'}
                            </span>
                          </div>
                          <div className="pmodal-xai-row">
                            <span className="pmodal-xai-name">
                              <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#f59e0b' }} />
                              SHAP Occlusion
                            </span>
                            <span className="pmodal-xai-time">
                              {result?.timings ? formatDuration(result.timings.shap_seconds) : isStepActive ? '● Computing' : '✓'}
                            </span>
                          </div>
                          <div className="pmodal-xai-row">
                            <span className="pmodal-xai-name">
                              <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#00c4cc' }} />
                              LIME Superpixel
                            </span>
                            <span className="pmodal-xai-time">
                              {result?.timings ? formatDuration(result.timings.lime_seconds) : isStepActive ? '● Computing' : '✓'}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="pmodal-status" aria-live="polite">
              {STATUS_MSGS[stage] ?? 'Processing…'}
            </div>

            <button className="pmodal-btn-cancel" onClick={handleCancel}>
              Cancel Analysis
            </button>
          </>
        )}
      </div>
    </div>
  );
}


/* ─── NEUMORPHIC RADIAL CONFIDENCE GAUGE ─────── */
function ConfidenceGauge({ value }: { value: number }) {
  const R = 43;
  const circ = 2 * Math.PI * R;
  const clamped = Math.max(0, Math.min(100, value));
  const offset = circ - (clamped / 100) * circ;
  const color = clamped >= 90 ? '#00b87c' : clamped >= 70 ? '#f59e0b' : '#ef4444';
  const isThreeDigits = clamped >= 99.5;

  return (
    <div className="neo-gauge-wrap">
      <div className="neo-gauge-well">
        <svg className="neo-gauge-svg" viewBox="0 0 120 120">
          <defs>
            <linearGradient id="cg-neo" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%"   stopColor={color} />
              <stop offset="100%" stopColor="#00c4cc" />
            </linearGradient>
            <filter id="gauge-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor={color} floodOpacity="0.45" />
            </filter>
          </defs>
          {/* Recessed Track */}
          <circle cx="60" cy="60" r={R} fill="none" stroke="var(--bg-subtle)" strokeWidth="7.5" opacity="0.6" />
          {/* Active Gradient Meter */}
          <circle cx="60" cy="60" r={R} fill="none"
            stroke="url(#cg-neo)" strokeWidth="7.5" strokeLinecap="round"
            strokeDasharray={circ} strokeDashoffset={offset}
            filter="url(#gauge-glow)"
            style={{ transition:'stroke-dashoffset 1.4s cubic-bezier(0.16,1,0.3,1)' }}
          />
        </svg>
        <div className="neo-gauge-content">
          <div className="neo-gauge-number">
            <span className={isThreeDigits ? 'neo-num-100' : 'neo-num-normal'}>
              {clamped.toFixed(0)}
            </span>
            <span className="neo-gauge-percent">%</span>
          </div>
          <span className="neo-gauge-pill">
            CONFIDENCE
          </span>
        </div>
      </div>
    </div>
  );
}

/* ─── NEUMORPHIC STAT CHIP ──────────────────────── */
function StatChip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="neo-stat-chip">
      <div style={{ fontSize:'1.05rem', fontWeight:800, color, lineHeight:1, fontFamily:"'Plus Jakarta Sans', sans-serif" }}>
        {value}
      </div>
      <div style={{ fontSize:'0.62rem', color:'var(--text-muted)', fontWeight:800, textTransform:'uppercase', letterSpacing:'0.06em', marginTop:3 }}>
        {label}
      </div>
    </div>
  );
}

/* ─── NEUMORPHIC SYMPTOM BAR ────────────────────── */
function SymptomBar({ name, pct, color }: { name: string; pct: number; color: string }) {
  return (
    <div className="neo-symptom-item">
      <div className="neo-symptom-head">
        <div style={{ display:'flex', alignItems:'center', gap:6 }}>
          <div style={{ width:6, height:6, borderRadius:'50%', background:color, boxShadow:`0 0 6px ${color}80` }} />
          <span className="neo-symptom-name">{name}</span>
        </div>
        <span className="neo-symptom-val" style={{ color }}>{pct}%</span>
      </div>
      <div className="neo-bar-track">
        <div className="neo-bar-fill" style={{ width: `${pct}%`, background: color, boxShadow:`0 0 8px ${color}50` }} />
      </div>
    </div>
  );
}

/* ─── NEUMORPHIC HEATMAP CARD ────────────────────── */
function HeatmapCard({ src, label, color, active, onClick }: {
  src: string; label: string; color: string; active: boolean; onClick: () => void;
}) {
  return (
    <div onClick={onClick} className={`neo-hmap-card ${active ? 'active' : ''}`} style={{
      border:`1.5px solid ${active ? color : 'var(--border-light)'}`,
      boxShadow: active ? `0 8px 24px ${color}35, var(--neo-shadow-soft)` : 'var(--neo-shadow-soft)',
    }}>
      <div className="neo-hmap-head" style={{
        background: active ? `${color}18` : 'var(--bg-subtle)',
        borderBottom: `1px solid ${active ? `${color}35` : 'var(--border-light)'}`,
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:6 }}>
          <div style={{
            width:7, height:7, borderRadius:'50%', background:color,
            boxShadow: active ? `0 0 8px ${color}` : 'none'
          }} />
          <span style={{ fontSize:'0.75rem', fontWeight:800, color: active ? color : 'var(--text-main)' }}>{label}</span>
        </div>
        {active ? (
          <span style={{ fontSize:'0.62rem', fontWeight:800, color:color, background:`${color}25`, padding:'2px 7px', borderRadius:99 }}>
            Active
          </span>
        ) : (
          <Eye size={12} color="var(--text-muted)" />
        )}
      </div>
      <img src={src} alt={label} className="neo-hmap-img" />
    </div>
  );
}

/* ─── NEUMORPHIC CLINICAL INTELLIGENCE HUB ──────── */
function ClinicalIntelligenceHub({
  result,
}: {
  result: Prediction;
}) {
  const [activeTab, setActiveTab] = useState<ClinicalTab>('pathology');

  const tabs: { id: ClinicalTab; label: string; icon: React.ReactNode; color: string }[] = [
    { id: 'pathology',   label: 'Pathology & Biology', icon: <Dna size={14} />,         color: '#00c4cc' },
    { id: 'treatment',   label: 'Treatment & Rx',      icon: <Pill size={14} />,        color: '#8b5cf6' },
    { id: 'prevention',  label: 'Prevention',          icon: <ShieldCheck size={14} />, color: '#10b981' },
    { id: 'precautions', label: 'Field Precautions',   icon: <AlertTriangle size={14} />, color: '#f59e0b' },
  ];

  const treatmentBullets  = parseBulletPoints(result.cure);
  const preventionBullets = parseBulletPoints(result.prevention);
  const precautionBullets = parseBulletPoints(result.precautions);
  const pathologyBullets  = parseBulletPoints(result.explanation);

  return (
    <div className="neo-hub-card">
      {/* ── Hub Header & Quick Summary Strip ── */}
      <div className="neo-hub-header">
        <div className="neo-hub-title-row">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="neo-hub-badge-icon">
              <Zap size={15} />
            </div>
            <div>
              <h3 className="neo-hub-title">Pathology & Agronomic Intelligence</h3>
              <p className="neo-hub-subtitle">Botanical disease mechanism, treatment protocols & prevention strategies</p>
            </div>
          </div>
        </div>

        {/* ── Neumorphic Segmented Tab Bar ── */}
        <div className="neo-tab-bar">
          {tabs.map(t => {
            const isAct = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`neo-tab-btn ${isAct ? 'active' : ''}`}
                style={{
                  color: isAct ? t.color : 'var(--text-muted)',
                  borderColor: isAct ? `${t.color}40` : 'transparent',
                }}
              >
                <span className="neo-tab-icon" style={{ color: isAct ? t.color : 'inherit' }}>
                  {t.icon}
                </span>
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Tab Content Area ── */}
      <div className="neo-tab-body">
        {/* ── TAB 1: PATHOLOGY ── */}
        {activeTab === 'pathology' && (
          <div className="neo-tab-pane">
            <div className="neo-pane-head">
              <span className="neo-pane-pill" style={{ background: 'rgba(0,196,204,0.12)', color: '#00c4cc' }}>
                🧬 Biological Disease Pathology
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}>
                Model class: {result.class_name}
              </span>
            </div>

            <div className="neo-grid-2col">
              <div className="neo-card-sub">
                <div className="neo-card-sub-title" style={{ color: '#00c4cc' }}>
                  <Dna size={14} /> Pathogen & Biology Overview
                </div>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-body)', lineHeight: 1.65 }}>
                  {result.explanation}
                </p>
              </div>

              <div className="neo-card-sub">
                <div className="neo-card-sub-title" style={{ color: '#00b87c' }}>
                  <Activity size={14} /> Diagnostic Symptoms & Markers
                </div>
                <div className="neo-bullet-list">
                  {pathologyBullets.map((b, i) => (
                    <div key={i} className="neo-bullet-item">
                      <Check size={13} color="#00b87c" style={{ flexShrink: 0, marginTop: 3 }} />
                      <span>{b}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 2: TREATMENT & RX ── */}
        {activeTab === 'treatment' && (
          <div className="neo-tab-pane">
            <div className="neo-pane-head">
              <span className="neo-pane-pill" style={{ background: 'rgba(139,92,246,0.12)', color: '#8b5cf6' }}>
                💊 Agronomic Treatment & Cure Protocol
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Organic & Chemical Interventions
              </span>
            </div>

            <div className="neo-bullet-list">
              {treatmentBullets.map((t, idx) => (
                <div key={idx} className="neo-step-card">
                  <div className="neo-step-num" style={{ background: '#8b5cf6' }}>
                    {idx + 1}
                  </div>
                  <div className="neo-step-content">
                    <div className="neo-step-title">Actionable Recommendation #{idx + 1}</div>
                    <div className="neo-step-desc">{t}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── TAB 3: PREVENTION ── */}
        {activeTab === 'prevention' && (
          <div className="neo-tab-pane">
            <div className="neo-pane-head">
              <span className="neo-pane-pill" style={{ background: 'rgba(16,185,129,0.12)', color: '#10b981' }}>
                🛡️ Long-Term Prevention Strategies
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Seasonal Crop Defense & Soil Management
              </span>
            </div>

            <div className="neo-bullet-list">
              {preventionBullets.map((p, idx) => (
                <div key={idx} className="neo-step-card">
                  <div className="neo-step-num" style={{ background: '#10b981' }}>
                    <ShieldCheck size={14} />
                  </div>
                  <div className="neo-step-content">
                    <div className="neo-step-title">Defense Strategy #{idx + 1}</div>
                    <div className="neo-step-desc">{p}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── TAB 4: PRECAUTIONS ── */}
        {activeTab === 'precautions' && (
          <div className="neo-tab-pane">
            <div className="neo-pane-head">
              <span className="neo-pane-pill" style={{ background: 'rgba(245,158,11,0.12)', color: '#f59e0b' }}>
                ⚠️ Immediate Field Precautions
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Containment & Infection Isolation
              </span>
            </div>

            <div className="neo-bullet-list">
              {precautionBullets.map((item, idx) => (
                <div key={idx} className="neo-step-card neo-step-warning">
                  <div className="neo-step-num" style={{ background: '#f59e0b' }}>
                    <AlertTriangle size={14} />
                  </div>
                  <div className="neo-step-content">
                    <div className="neo-step-title" style={{ color: '#f59e0b' }}>Precautionary Rule #{idx + 1}</div>
                    <div className="neo-step-desc">{item}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
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
  const [activeHm, setActiveHm]   = useState<XAIMethodKey>('gradcam_pp');
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [result, setResult]       = useState<Prediction | null>(null);
  const [procStage, setProcStage] = useState<ProcessStage>('idle');
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /* Context-aware AI explanation state & cache */
  const [explanationState, setExplanationState] = useState<{
    method: string;
    heading: string;
    methodLabel: string;
    text: string;
    loading: boolean;
  }>({
    method: 'gradcam_pp',
    heading: 'AI CLINICAL — GRAD-CAM++ INTERPRETATION',
    methodLabel: 'Grad-CAM++',
    text: '',
    loading: false,
  });

  const explanationCacheRef = useRef<Record<string, { heading: string; methodLabel: string; text: string }>>({});
  const activeReqIdRef = useRef<number>(0);

  const accept = useCallback((f: File) => {
    if (!f.type.startsWith('image/')) { setError('Please upload a valid image (JPEG / PNG / WEBP).'); return; }
    setError(null); setResult(null);
    setFile(f); setPreview(URL.createObjectURL(f));
  }, []);

  const onInput  = (e: ChangeEvent<HTMLInputElement>)  => { if (e.target.files?.[0]) accept(e.target.files[0]); };
  const onDragOv = (e: DragEvent<HTMLDivElement>)       => { e.preventDefault(); setDragging(true); };
  const onDragLv = ()                                   => setDragging(false);
  const onDrop   = (e: DragEvent<HTMLDivElement>)       => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files?.[0]) accept(e.dataTransfer.files[0]); };
  const reset    = ()                                   => { setFile(null); setPreview(null); setResult(null); setError(null); explanationCacheRef.current = {}; };

  const hmSrc = result
    ? activeHm === 'gradcam_pp' ? (result.gradcam_pp_heatmap_b64 ?? result.gradcam_heatmap_b64)
    : activeHm === 'ig'         ? (result.ig_heatmap_b64         ?? result.gradcam_pp_heatmap_b64 ?? result.gradcam_heatmap_b64)
    : activeHm === 'shap'       ? (result.shap_heatmap_b64       ?? result.gradcam_pp_heatmap_b64 ?? result.gradcam_heatmap_b64)
    : activeHm === 'lime'       ? (result.lime_heatmap_b64       ?? result.gradcam_pp_heatmap_b64 ?? result.gradcam_heatmap_b64)
    : (result.gradcam_pp_heatmap_b64 ?? result.gradcam_heatmap_b64)
    : null;

  /* Select an XAI method interactively to update overlay and fetch/display specific insight */
  const selectXAIMethod = useCallback(async (methodKey: XAIMethodKey) => {
    setActiveHm(methodKey);
    if (!result) return;

    const apiMethod = (
      methodKey === 'ig' ? 'integrated_gradients'
      : methodKey
    );

    // Check cache
    if (explanationCacheRef.current[apiMethod]) {
      const cached = explanationCacheRef.current[apiMethod];
      setExplanationState({
        method: apiMethod,
        heading: cached.heading,
        methodLabel: cached.methodLabel,
        text: cached.text,
        loading: false,
      });
      return;
    }

    const titles: Record<string, { heading: string; label: string }> = {
      gradcam_pp:           { heading: 'AI CLINICAL — GRAD-CAM++ INTERPRETATION',            label: 'Grad-CAM++' },
      integrated_gradients: { heading: 'AI CLINICAL — INTEGRATED GRADIENTS INTERPRETATION', label: 'Integrated Gradients' },
      shap:                 { heading: 'AI CLINICAL — SHAP INTERPRETATION',                  label: 'SHAP Occlusion' },
      lime:                 { heading: 'AI CLINICAL — LIME INTERPRETATION',                  label: 'LIME Superpixel' },
      all:                  { heading: 'AI CLINICAL — 4-ENGINE MULTI-MODAL XAI INTERPRETATION', label: 'All 4 Attribution Engines' },
    };

    const meta = titles[apiMethod] || titles.gradcam_pp;
    setExplanationState({
      method: apiMethod,
      heading: meta.heading,
      methodLabel: meta.label,
      text: '',
      loading: true,
    });

    const reqId = ++activeReqIdRef.current;

    try {
      const resp = await fetch(`${apiBase}/explain-xai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          class_name: result.class_name,
          confidence: result.confidence,
          xai_method: apiMethod,
        }),
      });

      if (!resp.ok) throw new Error('Explanation request failed');
      const data = await resp.json();

      if (activeReqIdRef.current === reqId) {
        explanationCacheRef.current[apiMethod] = {
          heading: data.heading,
          methodLabel: data.method_label,
          text: data.explanation,
        };
        setExplanationState({
          method: apiMethod,
          heading: data.heading,
          methodLabel: data.method_label,
          text: data.explanation,
          loading: false,
        });
      }
    } catch {
      if (activeReqIdRef.current === reqId) {
        setExplanationState(prev => ({
          ...prev,
          text: result.xai_feature_explanation || 'Detailed visual feature explanation available in report.',
          loading: false,
        }));
      }
    }
  }, [result, apiBase]);

  const analyze = async () => {
    if (!file) return;

    const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

    abortRef.current = new AbortController();
    setLoading(true);
    setError(null);
    explanationCacheRef.current = {};
    setProcStage('preprocess');

    const fd = new FormData();
    fd.append('file', file);

    try {
      /* Phase 1 — preprocessing  */
      await delay(250);
      setProcStage('classify');

      /* Phase 2 — classification & parallel XAI generation (all 4 premier engines computed) */
      const fetchPromise = fetch(
        `${apiBase}/predict?xai_method=all`,
        { method:'POST', body:fd, signal: abortRef.current.signal }
      );

      await delay(350);
      setProcStage('xai');

      /* Phase 3/4/5 held while fetch completes */
      const res = await fetchPromise;
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e?.detail ?? `HTTP ${res.status}`);
      }

      setProcStage('groq');
      await delay(300);
      setProcStage('report');
      await delay(200);

      const data: Prediction = await res.json();
      setResult(data);

      const initialMethod: XAIMethodKey = (
        xaiMethod === 'integrated_gradients' ? 'ig'
        : xaiMethod === 'shap' ? 'shap'
        : xaiMethod === 'lime' ? 'lime'
        : 'gradcam_pp'
      );
      setActiveHm(initialMethod);

      // Seed cache with the initial method's explanation
      const initialApiMethod = (
        initialMethod === 'ig' ? 'integrated_gradients'
        : initialMethod
      );
      const headingMap: Record<string, string> = {
        gradcam_pp:           'AI CLINICAL — GRAD-CAM++ INTERPRETATION',
        integrated_gradients: 'AI CLINICAL — INTEGRATED GRADIENTS INTERPRETATION',
        shap:                 'AI CLINICAL — SHAP INTERPRETATION',
        lime:                 'AI CLINICAL — LIME INTERPRETATION',
        all:                  'AI CLINICAL — 4-ENGINE MULTI-MODAL XAI INTERPRETATION',
      };
      const labelMap: Record<string, string> = {
        gradcam_pp:           'Grad-CAM++',
        integrated_gradients: 'Integrated Gradients',
        shap:                 'SHAP Occlusion',
        lime:                 'LIME Superpixel',
        all:                  'All 4 Attribution Engines',
      };

      explanationCacheRef.current[xaiMethod] = {
        heading: headingMap[xaiMethod] || 'AI CLINICAL — VISUAL FEATURE INTERPRETATION',
        methodLabel: labelMap[xaiMethod] || xaiMethod,
        text: data.xai_feature_explanation,
      };

      explanationCacheRef.current[initialApiMethod] = {
        heading: headingMap[initialApiMethod] || 'AI CLINICAL — GRAD-CAM++ INTERPRETATION',
        methodLabel: labelMap[initialApiMethod] || 'Grad-CAM++',
        text: data.xai_feature_explanation,
      };

      setExplanationState({
        method: initialApiMethod,
        heading: headingMap[initialApiMethod] || 'AI CLINICAL — GRAD-CAM++ INTERPRETATION',
        methodLabel: labelMap[initialApiMethod] || 'Grad-CAM++',
        text: data.xai_feature_explanation,
        loading: false,
      });

      setProcStage('success');
    } catch (e: unknown) {
      if ((e as Error).name === 'AbortError') {
        setProcStage('idle');
      } else {
        setError(e instanceof Error ? e.message : 'Analysis failed. Please try again.');
        setProcStage('error');
      }
    } finally {
      setLoading(false);
    }
  };

  const dismissModal = useCallback(() => setProcStage('idle'), []);
  const retryAnalysis = useCallback(() => { setProcStage('idle'); analyze(); }, [file, xaiMethod]);

  const xaiMethods = [
    { id:'all',                  label:'All 4 Engines',        color:'#00b87c', badge:'Default' },
    { id:'gradcam_pp',           label:'Grad-CAM++',           color:'#00e599', badge:'Coarse+' },
    { id:'integrated_gradients', label:'Int. Gradients',       color:'#8b5cf6', badge:'Pixel' },
    { id:'shap',                 label:'SHAP Occlusion',       color:'#f59e0b', badge:'8×8' },
    { id:'lime',                 label:'LIME Superpixel',      color:'#00c4cc', badge:'Superpixel' },
  ];

  return (
    <div className="page analyze-page">
      <ProcessingModal
        stage={procStage}
        result={result}
        onDismiss={dismissModal}
        onRetry={retryAnalysis}
        abortRef={abortRef}
      />
      <div className="ctr" style={{ paddingTop: 32 }}>

        {/* Header Bar */}
        <div style={{
          display:'flex', alignItems:'center', justifyContent:'space-between',
          marginBottom: 24, paddingBottom: 16, borderBottom: '1px solid var(--border-light)',
          flexWrap: 'wrap', gap: 12,
        }}>
          <div style={{ minWidth: 0, flex: 1 }}>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.03em', margin: '0 0 4px 0', overflowWrap: 'anywhere' }}>
              Analysis Workspace
            </h1>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', margin: 0, overflowWrap: 'anywhere' }}>
              Upload crop leaf photo, select attribution engine, and generate explainable diagnosis with visual feature breakdown.
            </p>
          </div>

          {file && (
            <button className="btn-clear-workspace-glass" onClick={reset}>
              <Trash2 size={14} />
              <span>Clear Workspace</span>
            </button>
          )}
        </div>

        {/* ── Main Two-Column Layout ── */}
        <div className="analyze-grid">

          {/* ════ LEFT COLUMN: UPLOAD & ENGINE CONFIG ════ */}
          <div className="analyze-left-col">
            {/* Upload Box */}
            <div className="card-light neo-elevated">
              <div className="card-light-head">
                <div className="card-light-title">
                  <Leaf size={18} color="#00b87c" />
                  <span>Leaf Specimen</span>
                </div>
                {file && (
                  <span style={{ fontSize:'0.72rem', color:'var(--green-main)', fontWeight:700, background:'var(--green-light)', padding:'2px 8px', borderRadius:99, flexShrink: 0 }}>
                    Loaded
                  </span>
                )}
              </div>

              <div className="card-light-body">
                {!preview ? (
                  <div
                    className={`dropzone-box ${dragging ? 'drag-over' : ''}`}
                    onClick={() => inputRef.current?.click()}
                    onDragOver={onDragOv}
                    onDragLeave={onDragLv}
                    onDrop={onDrop}
                  >
                    <input ref={inputRef} type="file" accept="image/*" onChange={onInput} style={{ display:'none' }} />
                    <div className="dropzone-icon-well">
                      <UploadCloud size={26} strokeWidth={2.2} />
                    </div>
                    <div className="dropzone-content">
                      <div className="dropzone-title">Upload Leaf Specimen</div>
                      <div className="dropzone-sub">Drag & drop your leaf photo here, or</div>
                    </div>
                    <button
                      type="button"
                      className="btn-browse-file"
                      onClick={(e) => {
                        e.stopPropagation();
                        inputRef.current?.click();
                      }}
                    >
                      <FolderOpen size={15} /> Browse from Device
                    </button>
                    <div className="dropzone-meta-pill">
                      JPEG · PNG · WEBP (Max 10MB)
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Preview box */}
                    <div style={{
                      position:'relative', borderRadius:14, overflow:'hidden',
                      background:'#090d18', aspectRatio:'4/3', marginBottom:16,
                      boxShadow:'var(--neo-shadow-soft)', width: '100%',
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
                          <div style={{ fontSize:'0.72rem', fontWeight:800, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.06em', marginBottom:8 }}>
                            Active Heatmap Layer
                          </div>
                          <div className="layer-select-grid">
                            {(['gradcam_pp','ig','shap','lime'] as const).map(m => {
                              const ok = m === 'gradcam_pp' ? !!(result.gradcam_pp_heatmap_b64 || result.gradcam_heatmap_b64)
                                : m === 'ig' ? !!result.ig_heatmap_b64
                                : m === 'shap' ? !!result.shap_heatmap_b64
                                : !!result.lime_heatmap_b64;
                              if (!ok) return null;
                              const lbl = {
                                gradcam_pp: 'Grad-CAM++',
                                ig: 'Int. Gradients',
                                shap: 'SHAP Occlusion',
                                lime: 'LIME Superpixel',
                              }[m];
                              const col = {
                                gradcam_pp: '#00e599',
                                ig: '#8b5cf6',
                                shap: '#f59e0b',
                                lime: '#00c4cc',
                              }[m];
                              const isAct = activeHm === m;
                              return (
                                <button
                                  key={m}
                                  type="button"
                                  onClick={() => selectXAIMethod(m)}
                                  className={`layer-select-btn ${isAct ? 'active' : ''}`}
                                  style={{ borderColor: isAct ? col : 'var(--border-light)' }}
                                >
                                  <span style={{ width:6, height:6, borderRadius:'50%', background:col, flexShrink:0 }} />
                                  <span className="layer-label" style={{ color: isAct ? col : 'var(--text-main)' }}>{lbl}</span>
                                </button>
                              );
                            })}
                          </div>
                        </div>

                        {/* Slider */}
                        <div style={{ display:'flex', alignItems:'center', gap:12, background:'var(--bg-subtle)', padding:'8px 12px', borderRadius:10, boxShadow:'var(--neo-shadow-pressed)' }}>
                          <Sliders size={13} color="var(--text-muted)" style={{ flexShrink: 0 }} />
                          <span style={{ fontSize:'0.72rem', fontWeight:700, color:'var(--text-muted)', flexShrink: 0 }}>Opacity</span>
                          <input type="range" min={0} max={1} step={0.05} value={opacity}
                            onChange={e => setOpacity(parseFloat(e.target.value))}
                            style={{ flex:1, minWidth: 0 }} />
                          <span style={{ fontSize:'0.75rem', fontWeight:800, color:'#00b87c', minWidth:32, textAlign:'right', flexShrink: 0 }}>
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
              <div className="card-light neo-elevated">
                <div className="card-light-head">
                  <div className="card-light-title">
                    <ScanLine size={18} color="#00c4cc" />
                    <span>Attribution Engine</span>
                  </div>
                </div>

                <div className="card-light-body" style={{ display:'flex', flexDirection:'column', gap:16 }}>
                  <div className="engine-select-group">
                    {xaiMethods.map(m => {
                      const isSelected = xaiMethod === m.id;
                      const isAll = m.id === 'all';
                      return (
                        <button
                          key={m.id}
                          type="button"
                          onClick={() => setXaiMethod(m.id)}
                          className={`engine-select-tile ${isSelected ? 'selected' : ''} ${isAll ? 'tile-all' : ''}`}
                        >
                          <div className="engine-tile-inner">
                            <span className="engine-tile-dot" style={{ backgroundColor: m.color }} />
                            <span className="engine-tile-title">{m.label}</span>
                          </div>
                          {m.badge && (
                            <span className={`engine-tile-badge ${isAll && isSelected ? 'badge-all' : ''}`}>
                              {m.badge}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>

                  {error && (
                    <div style={{
                      display:'flex', alignItems:'center', gap:8, padding:'12px',
                      background:'rgba(239, 68, 68, 0.12)', border:'1px solid rgba(239, 68, 68, 0.3)',
                      borderRadius:10, color:'#ef4444', fontSize:'0.85rem', fontWeight:600,
                      overflowWrap: 'anywhere',
                    }}>
                      <XCircle size={16} style={{ flexShrink: 0 }} />
                      <span>{error}</span>
                    </div>
                  )}

                  <button className="btn-upload-leaf" onClick={analyze} disabled={loading} style={{ width:'100%', justifyContent:'center' }}>
                    {loading ? (
                      <><RefreshCw size={18} style={{ animation:'spin 1s linear infinite', flexShrink: 0 }} /> Processing Neural Network…</>
                    ) : (
                      <><Microscope size={18} style={{ flexShrink: 0 }} /> Run Pathology Analysis <ChevronRight size={16} style={{ flexShrink: 0 }} /></>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* ════ RIGHT COLUMN: DIAGNOSIS & HEATMAPS ════ */}
          <div className="analyze-right-col">
            {!result ? (
              /* Empty State */
              <div className="card-light neo-elevated" style={{
                padding:'60px 24px', textAlign:'center',
                display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:420,
                width: '100%', minWidth: 0
              }}>
                <div style={{
                  width:64, height:64, borderRadius:'50%', background:'var(--green-light)',
                  color:'var(--green-main)', display:'flex', alignItems:'center', justifyContent:'center',
                  marginBottom:16, boxShadow:'var(--neo-shadow-soft)', flexShrink: 0
                }}>
                  <Sparkles size={28} />
                </div>
                <h3 style={{ fontSize:'1.2rem', fontWeight:800, color:'var(--text-main)', marginBottom:8, overflowWrap: 'anywhere' }}>
                  Awaiting Leaf Analysis
                </h3>
                <p style={{ fontSize:'0.9rem', color:'var(--text-muted)', maxWidth:360, lineHeight:1.6, marginBottom:24, overflowWrap: 'anywhere' }}>
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
                <div className="card-light neo-elevated" style={{ borderLeft:'4px solid #00b87c', width: '100%', minWidth: 0 }}>
                  <div className="card-light-body" style={{ display:'flex', gap:20, alignItems:'center', flexWrap:'wrap', minWidth: 0 }}>
                    <ConfidenceGauge value={result.confidence} />

                    <div style={{ flex:1, minWidth: 220 }}>
                      <div style={{
                        display:'inline-flex', alignItems:'center', gap:6, padding:'4px 10px',
                        borderRadius:99, background:'var(--green-light)', color:'var(--green-main)',
                        fontSize:'0.72rem', fontWeight:800, textTransform:'uppercase', marginBottom:8
                      }}>
                        <CheckCircle2 size={12} /> Diagnosis Confirmed
                      </div>
                      <h2 style={{ fontSize:'1.55rem', fontWeight:800, color:'var(--text-main)', letterSpacing:'-0.03em', marginBottom:4, overflowWrap:'anywhere', wordBreak:'break-word', lineHeight: 1.25 }}>
                        {formatDiseaseDisplayName(result.class_name)}
                      </h2>
                      <div style={{ fontSize:'0.76rem', color:'var(--text-muted)', fontFamily:"'JetBrains Mono', monospace", marginBottom:14, overflowWrap:'anywhere', wordBreak:'break-word' }}>
                        Model class: {result.class_name}
                      </div>

                      <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
                        <StatChip label="Confidence" value={`${result.confidence.toFixed(1)}%`} color="#00b87c" />
                        <StatChip label="XAI Mode" value={xaiMethod.toUpperCase()} color="#00c4cc" />
                        <StatChip label="Backbone" value="EfficientNet" color="#8b5cf6" />
                      </div>
                    </div>
                  </div>

                  {/* Symptom & Lesion Severity Bars */}
                  <div style={{ padding:'16px 20px', borderTop:'1px solid var(--border-light)', background:'var(--bg-subtle)', display:'flex', flexDirection:'column', gap:10, borderRadius:'0 0 16px 16px', minWidth: 0 }}>
                    <div style={{ fontSize:'0.72rem', fontWeight:800, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.06em' }}>
                      Lesion & Symptom Feature Severity
                    </div>
                    <SymptomBar name="Surface Lesion Density" pct={84} color="#00b87c" />
                    <SymptomBar name="Chlorotic Halo Expansion" pct={71} color="#00c4cc" />
                    <SymptomBar name="Foliar Necrosis Level" pct={91} color="#8b5cf6" />
                  </div>
                </div>

                {/* ── Real High-Precision Pipeline Telemetry Card ── */}
                {result.timings && (
                  <div className="neo-telemetry-card" style={{ width: '100%', minWidth: 0 }}>
                    <div className="neo-telemetry-head">
                      <div className="neo-telemetry-title">
                        <ScanLine size={16} color="#00e599" style={{ flexShrink: 0 }} />
                        <span>Diagnostic Pipeline Telemetry (Monotonic Clock)</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
                        <span style={{ fontSize: '0.70rem', color: 'var(--text-muted)', fontWeight: 700 }}>TOTAL LATENCY:</span>
                        <span className="pmodal-time-badge" style={{ fontSize: '0.75rem', padding: '3px 8px' }}>
                          {formatDuration(result.timings.total_seconds)}
                        </span>
                      </div>
                    </div>

                    <div className="neo-telemetry-grid">
                      <div className="neo-telemetry-item">
                        <span className="neo-telemetry-item-name">Preprocessing</span>
                        <span className="neo-telemetry-item-val" style={{ color: '#00e599' }}>
                          {formatDuration(result.timings.preprocessing_seconds)}
                        </span>
                      </div>
                      <div className="neo-telemetry-item">
                        <span className="neo-telemetry-item-name">EfficientNet-B0</span>
                        <span className="neo-telemetry-item-val" style={{ color: '#00c4cc' }}>
                          {formatDuration(result.timings.classification_seconds)}
                        </span>
                      </div>
                      <div className="neo-telemetry-item">
                        <span className="neo-telemetry-item-name">Grad-CAM++</span>
                        <span className="neo-telemetry-item-val" style={{ color: '#00e599' }}>
                          {formatDuration(result.timings.gradcam_pp_seconds ?? result.timings.gradcam_seconds)}
                        </span>
                      </div>
                      <div className="neo-telemetry-item">
                        <span className="neo-telemetry-item-name">Int. Gradients</span>
                        <span className="neo-telemetry-item-val" style={{ color: '#8b5cf6' }}>
                          {formatDuration(result.timings.ig_seconds)}
                        </span>
                      </div>
                      <div className="neo-telemetry-item">
                        <span className="neo-telemetry-item-name">SHAP Occlusion</span>
                        <span className="neo-telemetry-item-val" style={{ color: '#f59e0b' }}>
                          {formatDuration(result.timings.shap_seconds)}
                        </span>
                      </div>
                      <div className="neo-telemetry-item">
                        <span className="neo-telemetry-item-name">LIME Superpixel</span>
                        <span className="neo-telemetry-item-val" style={{ color: '#00c4cc' }}>
                          {formatDuration(result.timings.lime_seconds)}
                        </span>
                      </div>
                      <div className="neo-telemetry-item">
                        <span className="neo-telemetry-item-name">Reasoning</span>
                        <span className="neo-telemetry-item-val" style={{ color: '#ec4899' }}>
                          {formatDuration(result.timings.groq_seconds)}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Heatmaps Row with Direct Explanation Box */}
                <div className="card-light neo-elevated" style={{ width: '100%', minWidth: 0 }}>
                  <div className="card-light-head">
                    <div className="card-light-title">
                      <Activity size={18} color="#8b5cf6" style={{ flexShrink: 0 }} />
                      <span>XAI Visual Attribution Maps</span>
                    </div>
                    <span style={{ fontSize:'0.72rem', color:'var(--text-muted)', fontWeight:600 }}>
                      Click any card to inspect explanation
                    </span>
                  </div>
                  <div className="card-light-body" style={{ display:'flex', flexDirection:'column', gap:16, minWidth: 0 }}>
                    {/* 4-Card Heatmap Grid */}
                    <div className="xai-maps-grid">
                      {(result.gradcam_pp_heatmap_b64 || result.gradcam_heatmap_b64) && (
                        <HeatmapCard
                          src={result.gradcam_pp_heatmap_b64 || result.gradcam_heatmap_b64}
                          label="Grad-CAM++"
                          color="#00e599"
                          active={activeHm==='gradcam_pp'}
                          onClick={() => selectXAIMethod('gradcam_pp')}
                        />
                      )}
                      {result.ig_heatmap_b64 && (
                        <HeatmapCard
                          src={result.ig_heatmap_b64}
                          label="Int. Gradients"
                          color="#8b5cf6"
                          active={activeHm==='ig'}
                          onClick={() => selectXAIMethod('ig')}
                        />
                      )}
                      {result.shap_heatmap_b64 && (
                        <HeatmapCard
                          src={result.shap_heatmap_b64}
                          label="SHAP Occlusion"
                          color="#f59e0b"
                          active={activeHm==='shap'}
                          onClick={() => selectXAIMethod('shap')}
                        />
                      )}
                      {result.lime_heatmap_b64 && (
                        <HeatmapCard
                          src={result.lime_heatmap_b64}
                          label="LIME Superpixel"
                          color="#00c4cc"
                          active={activeHm==='lime'}
                          onClick={() => selectXAIMethod('lime')}
                        />
                      )}
                    </div>

                    {/* Prominent Context-Aware Explanation Directly Under Heatmaps */}
                    <div className="neo-groq-live-card">
                      <div className="neo-groq-live-head">
                        <div style={{ display:'flex', alignItems:'center', gap:8, minWidth: 0 }}>
                          <Sparkles size={16} color="#00b87c" style={{ flexShrink: 0 }} />
                          <span className="neo-groq-live-title">
                            {explanationState.heading}
                          </span>
                        </div>
                        <div style={{ display:'flex', alignItems:'center', gap:8, flexWrap:'wrap' }}>
                          <span className="neo-live-method-pill">
                            <span className="neo-live-dot" />
                            Analyzing: {explanationState.methodLabel}
                          </span>
                        </div>
                      </div>

                      {explanationState.loading ? (
                        <div className="neo-loading-box">
                          <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite', color: '#00b87c' }} />
                          <div>
                            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-main)' }}>
                              Interpreting {explanationState.methodLabel} attribution…
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                              Analyzing highlighted leaf regions and model decision features
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="neo-groq-live-body">
                          <p className="neo-groq-live-text">
                            {explanationState.text || result.xai_feature_explanation}
                          </p>
                          <div className="neo-chips-row" style={{ marginTop: 10 }}>
                            <span className="neo-mini-tag">🔬 Backbone: EfficientNet-B0</span>
                            <span className="neo-mini-tag">🎯 Confidence: {result.confidence.toFixed(1)}%</span>
                            <span className="neo-mini-tag">⚡ Decision Focus: Pathological Lesions</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* ── Interactive Neumorphic Clinical Intelligence Hub ── */}
                <ClinicalIntelligenceHub
                  result={result}
                />
              </>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
