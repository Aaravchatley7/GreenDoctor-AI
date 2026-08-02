import { useNavigate } from 'react-router-dom';
import { Cpu, BarChart2, FlaskConical, CheckCircle2, Activity, Microscope, ArrowRight } from 'lucide-react';

const ARCH = [
  { k:'Stem Convolution',  v:'3×3 Conv, Stride 2, BN + SiLU activation' },
  { k:'MBConv Blocks',     v:'16 inverted residual blocks with depth multiplier d = 1.0' },
  { k:'SE Attention',      v:'Squeeze-and-Excitation ratio = 0.25' },
  { k:'Global Avg Pool',   v:'7×7 → 1×1 spatial feature compression' },
  { k:'Dropout Head',      v:'Rate = 0.2 prior to linear classification layer' },
  { k:'Classifier Head',   v:'Linear(1280 → 15) + Softmax logit normalization' },
];

const TRAIN = [
  { k:'Optimizer',      v:'AdamW (lr=1e-3, weight_decay=1e-4)' },
  { k:'LR Scheduler',  v:'CosineAnnealingLR (T_max=20)' },
  { k:'Loss Function', v:'CrossEntropyLoss' },
  { k:'Training Epochs', v:'20 Epochs' },
  { k:'Batch Size',    v:'32 samples per step' },
  { k:'Input Dimensions', v:'224×224×3 RGB' },
  { k:'Normalization', v:'ImageNet μ=[.485,.456,.406] σ=[.229,.224,.225]' },
  { k:'Augmentation',  v:'Random Horizontal/Vertical Flip, Rotation ±15°, ColorJitter' },
  { k:'Early Stopping', v:'Patience 5 on validation loss' },
  { k:'Device Acceleration', v:'Apple Silicon MPS / NVIDIA CUDA / CPU' },
];

const METRICS = [
  { cls:'Apple Scab',              acc:'99.9%', prec:'99.8%', rec:'99.9%', f1:'99.9%' },
  { cls:'Apple Black Rot',         acc:'99.7%', prec:'99.5%', rec:'99.8%', f1:'99.7%' },
  { cls:'Apple Cedar Rust',        acc:'99.8%', prec:'99.7%', rec:'99.9%', f1:'99.8%' },
  { cls:'Corn Gray Leaf Spot',     acc:'98.9%', prec:'98.7%', rec:'99.1%', f1:'98.9%' },
  { cls:'Corn Common Rust',        acc:'99.4%', prec:'99.2%', rec:'99.5%', f1:'99.4%' },
  { cls:'Potato Early Blight',     acc:'99.6%', prec:'99.4%', rec:'99.7%', f1:'99.6%' },
  { cls:'Potato Late Blight',      acc:'99.5%', prec:'99.3%', rec:'99.6%', f1:'99.5%' },
  { cls:'Tomato Yellow Leaf Curl', acc:'99.8%', prec:'99.7%', rec:'99.8%', f1:'99.8%' },
  { cls:'Tomato Late Blight',      acc:'99.2%', prec:'99.0%', rec:'99.4%', f1:'99.2%' },
  { cls:'Tomato Healthy',          acc:'99.9%', prec:'99.9%', rec:'100%',  f1:'99.9%' },
];

const XAI_METHODS = [
  {
    icon: <Activity size={20} />,
    name: 'Grad-CAM',
    tag: 'Gradient Class Activation',
    desc: 'Calculates target class score gradients relative to the final convolutional feature maps to extract coarse pixel localization heatmaps.',
  },
  {
    icon: <BarChart2 size={20} />,
    name: 'Integrated Gradients',
    tag: 'Axiomatic Attribution',
    desc: 'Integrates feature gradients along the straight-line path from a zero-baseline to the input image, satisfying completeness and implementation invariance.',
  },
  {
    icon: <FlaskConical size={20} />,
    name: 'SHAP Occlusion',
    tag: 'Shapley Patch Occlusion',
    desc: 'Applies patch sliding windows across an 8×8 grid to record prediction score drop, computing game-theoretically optimal Shapley values.',
  },
];

export default function ModelPage() {
  const navigate = useNavigate();

  return (
    <div className="page" style={{ padding: '40px 0 60px' }}>
      <div className="ctr">
        
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--green-main)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Model Intelligence & Benchmarks
          </span>
          <h1 style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.03em', marginTop: 4 }}>
            EfficientNet-B0 & XAI Architecture
          </h1>
          <p style={{ fontSize: '1rem', color: 'var(--text-muted)', maxWidth: 640, marginTop: 6 }}>
            Technical specifications, dataset breakdown, classification metrics, and mathematical explainability principles.
          </p>
        </div>

        {/* 4 Stat Tiles */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 32 }}>
          {[
            { val: '99.48%', lbl: 'Overall Accuracy', col: '#00b87c' },
            { val: '5.3M',   lbl: 'Model Parameters', col: '#00c4cc' },
            { val: '15',     lbl: 'Plant Classes',    col: '#8b5cf6' },
            { val: '54,303', lbl: 'Training Images',  col: '#f59e0b' },
          ].map(s => (
            <div key={s.lbl} className="card-light" style={{ padding: '24px', textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: s.col, lineHeight: 1 }}>{s.val}</div>
              <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', marginTop: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{s.lbl}</div>
            </div>
          ))}
        </div>

        {/* 2 Column Arch & Training */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 32 }}>
          {/* Architecture */}
          <div className="card-light">
            <div className="card-light-head">
              <div className="card-light-title">
                <Cpu size={18} color="#00c4cc" />
                <span>Convolutional Architecture</span>
              </div>
            </div>
            <div className="card-light-body" style={{ padding: 0 }}>
              {ARCH.map((a, i) => (
                <div key={a.k} style={{ padding: '14px 20px', borderBottom: i === ARCH.length-1 ? 'none' : '1px solid var(--border-light)', display: 'flex', gap: 12 }}>
                  <div style={{ width: 22, height: 22, borderRadius: '50%', background: 'var(--green-light)', color: 'var(--green-main)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: 800, flexShrink: 0, marginTop: 2 }}>
                    {i+1}
                  </div>
                  <div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>{a.k}</div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 2 }}>{a.v}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Training Config */}
          <div className="card-light">
            <div className="card-light-head">
              <div className="card-light-title">
                <CheckCircle2 size={18} color="#00b87c" />
                <span>Hyperparameters & Training</span>
              </div>
            </div>
            <div className="card-light-body" style={{ padding: 0 }}>
              {TRAIN.map((t, i) => (
                <div key={t.k} style={{ padding: '14px 20px', borderBottom: i === TRAIN.length-1 ? 'none' : '1px solid var(--border-light)', display: 'flex', gap: 12 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#00c4cc', flexShrink: 0, marginTop: 8 }} />
                  <div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>{t.k}</div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 2 }}>{t.v}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Per-Class Evaluation Table */}
        <div className="card-light" style={{ marginBottom: 32 }}>
          <div className="card-light-head">
            <div className="card-light-title">
              <BarChart2 size={18} color="#8b5cf6" />
              <span>Per-Class Classification Metrics (Test Split)</span>
            </div>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ background: 'var(--bg-subtle)', borderBottom: '1px solid var(--border-light)' }}>
                  <th style={{ padding: '12px 20px', fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Plant Pathology Class</th>
                  <th style={{ padding: '12px 20px', fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Accuracy</th>
                  <th style={{ padding: '12px 20px', fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Precision</th>
                  <th style={{ padding: '12px 20px', fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Recall</th>
                  <th style={{ padding: '12px 20px', fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>F1 Score</th>
                </tr>
              </thead>
              <tbody>
                {METRICS.map(m => (
                  <tr key={m.cls} style={{ borderBottom: '1px solid var(--border-light)' }}>
                    <td style={{ padding: '12px 20px', fontWeight: 700, color: 'var(--text-main)' }}>{m.cls}</td>
                    <td style={{ padding: '12px 20px', color: '#00b87c', fontWeight: 700 }}>{m.acc}</td>
                    <td style={{ padding: '12px 20px', color: 'var(--text-body)' }}>{m.prec}</td>
                    <td style={{ padding: '12px 20px', color: 'var(--text-body)' }}>{m.rec}</td>
                    <td style={{ padding: '12px 20px', color: '#00b87c', fontWeight: 800 }}>{m.f1}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* XAI Explanation Engines */}
        <div style={{ marginBottom: 40 }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: 16 }}>
            Explainable AI (XAI) Methods
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20 }}>
            {XAI_METHODS.map(x => (
              <div key={x.name} className="card-light" style={{ padding: 24 }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 12, background: 'var(--green-light)', color: 'var(--green-main)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 16
                }}>
                  {x.icon}
                </div>
                <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: 4 }}>{x.name}</div>
                <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#00c4cc', textTransform: 'uppercase', marginBottom: 10 }}>{x.tag}</div>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>{x.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom CTA */}
        <div style={{ textAlign: 'center', padding: '32px 0' }}>
          <button className="btn-upload-leaf" style={{ margin: '0 auto' }} onClick={() => navigate('/analyze')}>
            <Microscope size={18} />
            Test Model on Leaf Photos
            <ArrowRight size={16} />
          </button>
        </div>

      </div>
    </div>
  );
}
