# 🌿 PhytoShield AI — Explainable Plant Disease Detection System

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React + Vite](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB.svg)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/ML%20Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.48%25-00b87c.svg)](#metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PhytoShield AI** is a research-grade, production-ready Explainable AI (XAI) Plant Pathology Intelligence System. It identifies crop diseases from leaf photography with **99.48% accuracy** across 15 agricultural classes, providing multi-modal visual attribution heatmaps (**Grad-CAM**, **Integrated Gradients**, **SHAP**) alongside **Groq LLM** visual feature explanations.

---

## 🌟 Key Features

- **High-Precision Neural Backbone**: EfficientNet-B0 architecture trained on 54,303 leaf images.
- **Triple XAI Attribution Engine**:
  - 🟢 **Grad-CAM**: Convolutional feature activation heatmaps.
  - 🟣 **Integrated Gradients**: Axiomatic straight-line path gradient attributions.
  - 🟠 **SHAP Patch Occlusion**: Game-theoretic Shapley value spatial sensitivity mapping.
- **Groq LLM Pathology Intelligence**: Natural language feature breakdowns explaining how heatmapped leaf regions (chlorotic halos, concentric lesions, necrotic spot density) influenced the model diagnosis.
- **Modern SaaS Web UI**:
  - PlantXAI-inspired aesthetic with 8pt symmetrical grid system.
  - Light & Dark mode theme toggle with persistent preferences.
  - Interactive opacity slider & real-time attribution layer switching.
- **FastAPI REST API**: Validated Pydantic schemas, CORS support, ONNX exporter, and OpenAPI interactive docs.

---

## 📁 Repository Structure

```
PhytoShield-AI/
├── api/                    # FastAPI routes and Pydantic schemas
├── app/                    # Main FastAPI app & Groq LLM explainer service
│   ├── main.py
│   └── services/
│       └── llm_explainer.py
├── data/                   # Data loader, splits, and class mappings
├── explainability/         # XAI algorithms (Grad-CAM, IG, SHAP, Visualizers)
│   ├── grad_cam.py
│   ├── integrated_gradients.py
│   ├── shap_explainer.py
│   └── visualizer.py
├── frontend/               # React + Vite TypeScript SaaS web app
│   ├── src/
│   │   ├── pages/ (Home.tsx, Analyze.tsx, Model.tsx)
│   │   ├── App.tsx
│   │   └── index.css
│   └── vite.config.ts
├── models/                 # Model definitions & class names mapping
│   ├── efficientnet.py
│   └── class_names.json
├── preprocessing/          # Image transforms & normalization
├── training/               # Training pipeline & learning rate schedulers
├── evaluation/             # Metrics evaluator & confusion matrix generator
├── utils/                  # Logger, ONNX export, and evaluation metrics
└── tests/                  # Unit and integration test suite
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- Groq API Key (Optional for LLM explanations, fallback knowledge base built-in)

### 1. Backend Setup

```bash
# Clone repository
git clone https://github.com/parthsharma8368/PhytoShield-AI.git
cd PhytoShield-AI

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will run at `http://localhost:8000`. Swagger API docs available at `http://localhost:8000/docs`.

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install node dependencies
npm install

# Start Vite dev server
npm run dev
```

Frontend will run at `http://localhost:5173`.

---

## 📊 Model & Benchmark Metrics

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **99.48%** |
| **Parameters** | **5.3 Million** |
| **Input Shape** | `224 × 224 × 3` |
| **Inference Latency** | `< 45ms` (MPS / GPU) |
| **Supported Classes** | **15 Plant Pathology Classes** |

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
PORT=8000
HOST=0.0.0.0
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
