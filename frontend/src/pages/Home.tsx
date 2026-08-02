import { useNavigate } from 'react-router-dom';
import { 
  Upload, ArrowRight, CheckCircle2, Flame, Brain, Leaf, 
  Activity, ShieldCheck, Microscope, BarChart2, Sparkles
} from 'lucide-react';

const STATS = [
  { val: '99.48%', lbl: 'Accuracy' },
  { val: '15',     lbl: 'Plant Classes' },
  { val: '<1s',    lbl: 'Inference Time' },
];

const FEATURES = [
  { c: 'c-em', icon: <Leaf size={22} />,       t: 'Deep Learning Diagnosis', d: 'EfficientNet-B0 model trained on PlantVillage dataset delivers 99.48% classification accuracy across 15 plant pathology classes.' },
  { c: 'c-cy', icon: <Flame size={22} />,      t: 'Grad-CAM Attention',      d: 'Class Activation Mapping renders high-resolution heatmaps indicating exact leaf lesion regions that influenced model outputs.' },
  { c: 'c-vi', icon: <Activity size={22} />,   t: 'Integrated Gradients',   d: 'Path integral attribution calculates baseline-to-input pixel gradients to satisfy completeness and implementation invariance axioms.' },
  { c: 'c-am', icon: <BarChart2 size={22} />,  t: 'SHAP Occlusion',         d: 'Shapley value patch occlusion quantifies true spatial feature importance using game theory principles.' },
  { c: 'c-em', icon: <Sparkles size={22} />,   t: 'Clinical Prescription',  d: 'Generates medical-grade treatment plans, chemical/biological cures, and long-term prevention protocols for every diagnosis.' },
  { c: 'c-cy', icon: <ShieldCheck size={22} />, t: 'Production API',        d: 'FastAPI backend with Pydantic schema validation, CORS support, async request handling, and interactive OpenAPI docs.' },
];

const PIPELINE_STEPS = [
  { n: '01', t: 'Upload Leaf',      d: 'Select or drag-and-drop any leaf photo in JPEG, PNG, or WEBP format.' },
  { n: '02', t: 'Preprocess & Scale', d: 'Image is resized to 224×224, normalized with ImageNet statistics.' },
  { n: '03', t: 'CNN Inference',    d: 'EfficientNet-B0 calculates class probabilities and returns top diagnosis.' },
  { n: '04', t: 'XAI Visualizations', d: 'Grad-CAM, Integrated Gradients, and SHAP overlay heatmaps onto the leaf.' },
];

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="page">
      {/* ── HERO SECTION MATCHING PlantXAI INSPIRATION ── */}
      <section className="hero-sec">
        <div className="ctr">
          <div className="hero-grid">
            
            {/* LEFT HERO TEXT & CTAS */}
            <div>
              <h1 className="hero-h1">
                AI Powered <span className="text-green">Plant</span><br />
                <span className="text-teal">Disease</span> Detection<br />
                <span className="text-light">with Explainable AI</span>
              </h1>

              <p className="hero-p">
                Upload a plant leaf image and receive an instant disease prediction
                together with a <strong>Grad-CAM visualization</strong> showing exactly
                which parts of the leaf influenced the AI decision.
              </p>

              <div className="hero-actions">
                <button className="btn-upload-leaf" onClick={() => navigate('/analyze')}>
                  <Upload size={18} />
                  Upload Leaf
                </button>

                <button className="btn-demo" onClick={() => navigate('/model')}>
                  View Demo
                  <ArrowRight size={16} />
                </button>
              </div>

              {/* STATS ROW */}
              <div className="hero-stats-row">
                {STATS.map(s => (
                  <div key={s.lbl} className="hero-stat-item">
                    <div className="hero-stat-val">{s.val}</div>
                    <div className="hero-stat-lbl">{s.lbl}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* RIGHT PROCESS FLOW SHOWCASE CARD */}
            <div className="flow-showcase-card">
              
              {/* TOP CARD BAR */}
              <div className="showcase-top-bar">
                <div className="model-backbone-tag">
                  <strong>EfficientNet-B0</strong>
                  <span>CNN Backbone</span>
                </div>

                <div className="thumb-preview">
                  <span className="thumb-dot" />
                  Tomato Leaf • Input
                </div>
              </div>

              {/* STACK OF PROCESS FLOW PILLS */}
              <div className="pipeline-stack">
                
                {/* 1. LEAF IMAGE */}
                <div className="step-pill-item pill-green">
                  <div className="left">
                    <Leaf size={16} />
                    <span>Leaf Image</span>
                  </div>
                </div>

                <div className="down-arrow">↓</div>

                {/* 2. CNN PREDICTION */}
                <div className="step-pill-item pill-blue">
                  <div className="left">
                    <Brain size={16} />
                    <span>CNN Prediction</span>
                  </div>
                </div>

                <div className="down-arrow">↓</div>

                {/* 3. GRAD-CAM HEATMAP */}
                <div className="step-pill-item pill-yellow">
                  <div className="left">
                    <Flame size={16} />
                    <span>Grad-CAM Heatmap</span>
                  </div>
                </div>

                <div className="down-arrow">↓</div>

                {/* 4. DISEASE RESULT */}
                <div className="step-pill-item pill-pink">
                  <div className="left">
                    <Microscope size={16} />
                    <span>Disease Result</span>
                  </div>
                </div>

                <div className="down-arrow">↓</div>

                {/* 5. CONFIDENCE SCORE */}
                <div className="step-pill-item pill-purple">
                  <div className="left">
                    <BarChart2 size={16} />
                    <span>Confidence Score</span>
                  </div>
                  <span style={{
                    background: '#00b87c', color: '#ffffff',
                    padding: '2px 8px', borderRadius: 99, fontSize: '0.72rem', fontWeight: 800
                  }}>
                    99.48%
                  </span>
                </div>

                <div className="down-arrow">↓</div>

                {/* 6. OUTCOME CARD */}
                <div className="result-card-outcome">
                  <div>
                    <div className="title">Early Blight Detected</div>
                    <div className="sub">Confidence: 99.48% • High</div>
                  </div>
                  <div className="result-check-badge">
                    <CheckCircle2 size={20} />
                  </div>

                  <div className="floating-tooltip-badge">
                    Grad-CAM XAI Visual Explanation
                  </div>
                </div>

              </div>

            </div>

          </div>
        </div>
      </section>

      {/* ── CORE CAPABILITIES ── */}
      <section className="sec" style={{ background: 'var(--bg-card)', borderTop: '1px solid var(--border-light)', borderBottom: '1px solid var(--border-light)' }}>
        <div className="ctr">
          <p className="sec-eye">Features</p>
          <h2 className="sec-h2">High Precision Pathology & Explainability</h2>
          <p className="sec-desc">
            A research-grade computer vision pipeline designed for modern agricultural intelligence.
          </p>

          <div className="feat-grid">
            {FEATURES.map(f => (
              <div key={f.t} className="feat-card">
                <div className={`fi ${f.c}`}>{f.icon}</div>
                <div className="feat-tt">{f.t}</div>
                <div className="feat-bd">{f.d}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <section className="sec">
        <div className="ctr">
          <p className="sec-eye">How It Works</p>
          <h2 className="sec-h2">Simple 4-Step Analysis Pipeline</h2>
          <p className="sec-desc">
            From raw field photography to an explainable, medical-grade plant diagnosis in seconds.
          </p>

          <div className="pipe-row">
            {PIPELINE_STEPS.map(p => (
              <div className="pipe-step" key={p.n}>
                <div className="pipe-n">{p.n}</div>
                <div className="pipe-tt">{p.t}</div>
                <div className="pipe-dd">{p.d}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA SECTION ── */}
      <section className="sec" style={{ background: 'var(--bg-dark)', color: '#ffffff', textAlign: 'center' }}>
        <div className="ctr">
          <h2 className="sec-h2" style={{ color: '#ffffff', marginBottom: 12 }}>
            Ready to Test Plant Diagnostics?
          </h2>
          <p className="sec-desc" style={{ color: '#94a3b8', marginBottom: 32 }}>
            Upload any crop leaf photo and get instant AI pathology analysis with visual heatmaps.
          </p>
          <button className="btn-upload-leaf" style={{ margin: '0 auto' }} onClick={() => navigate('/analyze')}>
            <Upload size={18} />
            Start Leaf Analysis
            <ArrowRight size={16} />
          </button>
        </div>
      </section>
    </div>
  );
}
