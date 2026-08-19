import { BrowserRouter, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { Leaf, ArrowRight, Printer, Sun, Moon } from 'lucide-react';
import HomePage from './pages/Home';
import AnalyzePage from './pages/Analyze';
import ModelPage from './pages/Model';
import AmbientBackground from './components/AmbientBackground';

const API_BASE = 'http://localhost:8000/api';

const NAV_LINKS = [
  { path: '/',        label: 'Home' },
  { path: '/analyze', label: 'Analyze' },
  { path: '/model',   label: 'Model Info' },
];

function Nav({
  apiOnline,
  showPrint,
  theme,
  onToggleTheme,
}: {
  apiOnline: boolean | null;
  showPrint: boolean;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
}) {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <nav className="nav">
      <div className="nav-inner">
        {/* Brand Logo & Wordmark matching PlantXAI style */}
        <div className="nav-brand" onClick={() => navigate('/')}>
          <div className="nav-logo">
            <Leaf size={20} strokeWidth={2.5} />
          </div>
          <div className="nav-wordmark">
            PhytoShield<span className="highlight">AI</span>
          </div>
        </div>

        {/* Links */}
        <div className="nav-links">
          {NAV_LINKS.map(({ path, label }) => (
            <button
              key={path}
              className={`nav-link ${location.pathname === path ? 'active' : ''}`}
              onClick={() => navigate(path)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Right CTA / Status / Dark Mode Toggle */}
        <div className="nav-right">
          {/* Dark / Light Mode Toggle Button */}
          <button
            className="btn-theme-toggle"
            onClick={onToggleTheme}
            title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle Theme"
          >
            {theme === 'dark' ? <Sun size={18} color="#fde047" /> : <Moon size={18} color="#0f172a" />}
          </button>

          {apiOnline !== null && (
            <div className={`api-status-badge ${apiOnline ? 'on' : 'off'}`}>
              <span className="api-dot" />
              {apiOnline ? 'Model Ready 99.48%' : 'API Offline'}
            </div>
          )}

          {showPrint ? (
            <button className="btn-try" onClick={() => window.print()}>
              <Printer size={15} />
              Export PDF
            </button>
          ) : (
            <button className="btn-try" onClick={() => navigate('/analyze')}>
              Try Now
              <ArrowRight size={15} />
            </button>
          )}
        </div>
      </div>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-in">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="nav-logo" style={{ width: 28, height: 28 }}>
            <Leaf size={15} strokeWidth={2.5} />
          </div>
          <span className="footer-copy">
            © 2025 PhytoShield AI · Explainable Plant Pathology Intelligence System
          </span>
        </div>
        <div className="footer-meta">
          <span>v1.0.0</span>
          <span>EfficientNet-B0</span>
          <span>15 Classes</span>
          <span>Grad-CAM · IG · SHAP</span>
        </div>
      </div>
    </footer>
  );
}

function AppInner() {
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('phyto_theme');
    return saved === 'dark' ? 'dark' : 'light';
  });

  const location = useLocation();

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('phyto_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  useEffect(() => {
    const check = () =>
      fetch(`${API_BASE}/health`)
        .then(r => r.json())
        .then(d => setApiOnline(d.status === 'healthy' && d.model_loaded))
        .catch(() => setApiOnline(false));
    check();
    const id = setInterval(check, 30000);
    return () => clearInterval(id);
  }, []);

  const showPrint = location.pathname === '/analyze';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative' }}>
      <AmbientBackground />
      <Nav
        apiOnline={apiOnline}
        showPrint={showPrint}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      <main style={{ flex: 1, position: 'relative', zIndex: 1 }}>
        <Routes>
          <Route path="/"        element={<HomePage />} />
          <Route path="/analyze" element={<AnalyzePage apiBase={API_BASE} />} />
          <Route path="/model"   element={<ModelPage />} />
          <Route path="*"        element={<HomePage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppInner />
    </BrowserRouter>
  );
}
