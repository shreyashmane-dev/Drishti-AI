<div align="center">

# 👁️ DRISHTI AI — Clinical Retinal Vision & Biomarker Studio
### Autonomous Diabetic Retinopathy Diagnostic System • Grad-CAM Explainability • Multi-Modal Clinical Copilot

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![EfficientNet-B3](https://img.shields.io/badge/Backbone-EfficientNet--B3-0284c7.svg?style=for-the-badge)](https://arxiv.org/abs/1905.11946)
[![Dual CBAM Attention](https://img.shields.io/badge/Attention-Dual_CBAM-7c3aed.svg?style=for-the-badge)](https://arxiv.org/abs/1807.06521)
[![FastAPI](https://img.shields.io/badge/API-FastAPI_v2.0-059669.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

<!-- Big Download Action Buttons -->
<p align="center">
  <a href="https://github.com/shreyashmane-dev/Drishti-AI/raw/main/artifacts/models/drishti_best_model.pth">
    <img src="https://img.shields.io/badge/📥_Download_Model_Weights-drishti__best__model.pth_(44.2_MB)-7c3aed?style=for-the-badge&logo=pytorch&logoColor=white" alt="Download Pretrained Model Weights" />
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/shreyashmane-dev/Drishti-AI/archive/refs/heads/main.zip">
    <img src="https://img.shields.io/badge/💻_Download_Source_Code-ZIP_Archive-0284c7?style=for-the-badge&logo=github&logoColor=white" alt="Download Source Code ZIP" />
  </a>
</p>

<p align="center">
  <strong><a href="#-quickstart">Quickstart</a></strong> •
  <strong><a href="#-neural-brain--model-architecture">Neural Architecture</a></strong> •
  <strong><a href="#-key-features">Key Capabilities</a></strong> •
  <strong><a href="#-quantitative-biomarkers">Biomarkers</a></strong> •
  <strong><a href="#-clinical-copilot-personas">AI Copilot</a></strong> •
  <strong><a href="#-citation--license">License</a></strong>
</p>

</div>

---

## 🌟 Executive Summary

**Drishti AI** is a clinical-grade artificial intelligence ophthalmic platform engineered for early screening, multi-scale lesion grading, and quantitative biometric morphometry of **Diabetic Retinopathy (DR)** from digital color fundus photography.

Built on an optimized **EfficientNet-B3 + Dual Convolutional Block Attention Module (CBAM)** architecture, Drishti AI delivers human-expert level diagnostic sensitivity across the full International Clinical Diabetic Retinopathy (ICDR) scale, coupled with high-resolution **Grad-CAM visual saliency heatmaps** and an interactive **Multi-Persona Clinical Copilot**.

---

## 📥 Direct Downloads

| Resource | File | Size | Format | Direct Link |
| :--- | :--- | :--- | :--- | :--- |
| **Best Trained Model Weights** | `drishti_best_model.pth` | **44.2 MB** | PyTorch Checkpoint | [⬇️ Download Model](https://github.com/shreyashmane-dev/Drishti-AI/raw/main/artifacts/models/drishti_best_model.pth) |
| **Complete Source Repository** | `Drishti-AI-main.zip` | **~50 MB** | ZIP Archive | [⬇️ Download ZIP](https://github.com/shreyashmane-dev/Drishti-AI/archive/refs/heads/main.zip) |
| **Pretrained TorchScript Engine** | `drishti_retinopathy_efficientnet_b3_traced.pt` | **45.1 MB** | TorchScript (.pt) | `artifacts/models/` |

> 💡 *When running the local server, you can also download weights directly by clicking the **Weights (.pth)** button in the top navigation bar or via `http://127.0.0.1:8000/api/download/model`.*

---

## 🧠 Neural Brain & Model Architecture

```
[ Input Fundus ]  512 × 512 × 3 RGB  (CLAHE Preprocessing & Normalization)
       │
       ▼
[ EfficientNet-B3 Backbone ]  10,696,232 parameters (MBConv1 to MBConv6 Stages)
       │
       ▼
[ Dual CBAM Attention Block ]  295,010 parameters
       ├─ Channel Attention: Mc(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
       └─ Spatial Attention: Ms(F) = σ(f⁷ˣ⁷([AvgPool(F); MaxPool(F)]))
       │
       ▼
[ Adaptive Global Avg Pooling ]  1536-Dimensional Latent Vector
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                 ▼                                 ▼
[ Categorical Classifier ]     [ Ordinal Regressor ]           [ Grad-CAM Hook ]
  Dense(1536→256) + LayerNorm    Dense(1536→128) + ReLU          Layer: features.7
  Dropout(0.3) + Linear(5)       Dropout(0.3) + Linear(1)        Backprop Class Gradients
  ↳ 5-Class Softmax Probs        ↳ Continuous QWK Severity       ↳ 512×512 Heatmap
```

### 📊 Parameter & Hardware Telemetry

| Component | Layer Configuration | Parameter Count | % of Network |
| :--- | :--- | :--- | :--- |
| **Backbone Extractor** | `EfficientNet-B3` (Inverted Residual Blocks 1–7) | **10,696,232** | 92.34% |
| **Attention Mechanism** | `CBAM` (Dual Channel + Spatial Attention Gate) | **295,010** | 2.55% |
| **Categorical Head** | `Dense(1536→256) + LayerNorm + Linear(5)` | **395,269** | 3.41% |
| **Continuous Regressor**| `Dense(1536→128) + ReLU + Linear(1)` | **196,865** | 1.70% |
| **Total Parameters** | **DrishtiVisionV2 End-to-End** | **11,583,376** | **100.0%** |
| **Trainable Parameters**| **100% Fine-tuned on Clinical Datasets** | **11,583,376** | **100.0%** |
| **Computational Cost** | FP32 Forward Pass | **1.84 GFLOPs** | — |
| **Inference Latency** | NVIDIA RTX GPU / Apple Silicon / Intel CPU | **42 ms / 65 ms / 185 ms** | — |

---

## 🔬 Key Capabilities

### 1. Dual-Task ICDR Staging (0 to 4)
* **Grade 0 — Healthy / Normal**: No diabetic lesions; normal vascular caliber.
* **Grade 1 — Mild NPDR**: Microaneurysms isolated to capillary beds.
* **Grade 2 — Moderate NPDR**: Microaneurysms, dot/blot hemorrhages, hard exudates, cotton wool spots.
* **Grade 3 — Severe NPDR**: >20 intraretinal hemorrhages in all 4 quadrants or prominent venous beading (ETDRS 4:2:1 rule).
* **Grade 4 — Proliferative DR (PDR)**: Neovascularization of optic disc (NVD), vitreous hemorrhage risk.

### 2. Quantitative Retinal Biomarkers
* **Optic Disc Localization & Cup-to-Disc Ratio (CDR)**:
  * Automatic localization and segmentation of the optic nerve head.
  * Measures physiological CDR to screen for secondary **Glaucoma** risk (CDR > 0.50 triggers suspect alert).
* **Macular Center & CSME Risk Analysis**:
  * Assesses foveal center activation intensity for **Clinically Significant Macular Edema (CSME)**.
* **Vascular Tree Density & Arteriolar Tortuosity**:
  * Segments retinal blood vessels to compute vascular caliber % and tortuosity indices (hypertensive retinopathy indicators).
* **ETDRS 4-Quadrant Attention Spread**:
  * Measures neural focus across Superior-Temporal (ST), Inferior-Temporal (IT), Superior-Nasal (SN), and Inferior-Nasal (IN) sectors.
* **Longitudinal 1-Year Progression Modeling**:
  * Statistical projections of disease progression probability within 12 months.

### 3. Interactive Split-Screen Grad-CAM Visualizer
* Real-time interactive divider slider comparing the original digital fundus against the rectified Grad-CAM attention heatmap.
* Glow opacity controls and heat map filter tuning.

### 4. Multi-Persona Conversational Copilot
* **🩺 Patient Care Mode**: Plain-language explanations, reassuring guidance, and dietary/HbA1c recommendations.
* **👨‍⚕️ Specialist MD Mode**: ICDR/ETDRS differential analysis, lesion morphometry, anti-VEGF protocols (Aflibercept/Ranibizumab), and laser photocoagulation timing.
* **🔬 Neural Lab Mode**: Gradient distribution statistics, softmax entropy, and layer activation weights.

---

## 🚀 Quickstart

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/shreyashmane-dev/Drishti-AI.git
cd Drishti-AI

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Clinical Web Studio
python app.py
```

Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to launch the visual interface.

---

## 📁 Repository Structure

```
DrishtiAI/
├── ai/
│   ├── config.py                 # Hyperparameters, clinical labels, thresholds
│   ├── conversational_agent.py   # Multi-persona clinical LLM copilot
│   ├── inference.py              # End-to-end vision & biomarker orchestrator
│   ├── server.py                 # FastAPI production server & static routers
│   ├── datasets/
│   │   ├── preprocessing.py      # CLAHE enhancement & tensor transformations
│   │   └── unified_dataset.py    # Multi-cohort PyTorch DataLoader
│   ├── explainability/
│   │   ├── biomarkers.py         # Optic disc CDR, CSME, vascular density
│   │   └── gradcam.py            # Gradient-weighted class activation mapping
│   └── models/
│       ├── backbone.py           # EfficientNet-B3 + CBAM architecture
│       ├── advanced_backbone.py  # ConvNeXt multi-scale feature extractor
│       ├── loss.py               # Quadratic Weighted Kappa & Focal loss
│       └── quality_checker.py    # Sharpness & illumination heuristic validator
├── artifacts/
│   └── models/
│       └── drishti_best_model.pth # Pretrained PyTorch model weights (44.2 MB)
├── frontend/
│   ├── index.html                # Pristine clinical web studio & HUD
│   ├── style.css                 # Glassmorphic responsive styling
│   └── app.js                    # Visualizer, slider, neural brain canvas
├── data/                         # Sample clinical scans for demo
├── requirements.txt              # PyTorch, FastAPI, OpenCV, Timm, etc.
└── app.py                        # Root launcher script
```

---

## ⚖️ Clinical Disclaimer

> **⚠️ Medical Notice:** Drishti AI is developed as an assistive artificial intelligence diagnostic and research tool. It is designed to aid optometrists, ophthalmologists, and primary care physicians in triage and screening workflows. It does not replace definitive medical judgment or in-person slit-lamp biomicroscopy by an eye care specialist.

---

## 📄 License & Attribution

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Developed with ❤️ by **[Shreyash Mane](https://github.com/shreyashmane-dev)**. If you find this project helpful for research or clinical development, please consider giving it a ⭐ on GitHub!
