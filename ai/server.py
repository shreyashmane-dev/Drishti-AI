"""
DRISHTI AI - Full Production Web API & GUI Server.
Powers the sleek medical white GUI with deep vision diagnostics, Grad-CAM visualization,
and ChatGPT-style interactive conversational intelligence.
"""

import base64
import io
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import webbrowser

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from ai.config import DATA_DIR
from ai.conversational_agent import chat_agent
from ai.inference import DrishtiInferenceEngine

app = FastAPI(title="DrishtiAI Clinical Vision & Dialogue System", version="2.0.0")

# Lazy-loaded inference engine
engine: Optional[DrishtiInferenceEngine] = None


def get_inference_engine() -> DrishtiInferenceEngine:
    global engine
    if engine is None:
        print("🧠 Initializing Drishti Deep Vision Backbone (DrishtiVisionV2 + CBAM Attention)...")
        engine = DrishtiInferenceEngine()
    return engine


# Clinical sample presets
SAMPLE_PRESETS = [
    {
        "id": "sample_g0",
        "name": "Healthy Retina (Grade 0)",
        "stage": "No Diabetic Retinopathy",
        "file": "data/aptos2019/real_images/002c21358ce6.png",
    },
    {
        "id": "sample_g1",
        "name": "Mild NPDR (Grade 1)",
        "stage": "Early Microaneurysms",
        "file": "data/aptos2019/real_images/0024cdab0c1e.png",
    },
    {
        "id": "sample_g2",
        "name": "Moderate NPDR (Grade 2)",
        "stage": "Hemorrhages & Exudates",
        "file": "data/aptos2019/real_images/000c1434d8d7.png",
    },
    {
        "id": "sample_g3",
        "name": "Severe NPDR (Grade 3)",
        "stage": "Extensive Quadrant Bleeds",
        "file": "data/aptos2019/real_images/0104b032c141.png",
    },
    {
        "id": "sample_g4",
        "name": "Proliferative DR (Grade 4)",
        "stage": "Neovascularization Risk",
        "file": "data/aptos2019/real_images/001639a390f0.png",
    },
]


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    sample_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    context_analysis: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None
    persona: Optional[str] = "patient"


@app.get("/api/samples")
async def get_samples():
    """Returns list of curated clinical fundus samples for one-click testing."""
    samples_out = []
    for s in SAMPLE_PRESETS:
        p = PROJECT_ROOT / s["file"]
        if p.exists():
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            samples_out.append({**s, "thumbnail": f"data:image/png;base64,{b64}"})
    return {"samples": samples_out}


@app.post("/api/analyze")
async def analyze_image(req: AnalyzeRequest):
    """
    Executes end-to-end deep learning vision analysis, Grad-CAM generation,
    and returns initial conversational synthesis.
    """
    img_bytes = None
    orig_b64 = None

    if req.sample_id:
        target = next((s for s in SAMPLE_PRESETS if s["id"] == req.sample_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="Sample preset not found")
        img_path = PROJECT_ROOT / target["file"]
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Sample image file missing")
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        orig_b64 = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

    elif req.image_base64:
        # Strip data URL prefix if present
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(raw_b64)
            orig_b64 = req.image_base64 if req.image_base64.startswith("data:") else f"data:image/jpeg;base64,{raw_b64}"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Base64 image payload: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide either image_base64 or sample_id")

    # Run inference engine
    eng = get_inference_engine()
    result = eng.analyze(img_bytes, return_gradcam_overlay=True)

    # Attach original image data for UI display
    result["original_image_base64"] = orig_b64

    # Generate initial responsive ChatGPT-style clinical report
    initial_report = chat_agent.generate_initial_report(result)
    result["conversational_report"] = initial_report

    return result


@app.post("/api/chat")
async def chat_interaction(req: ChatRequest):
    """
    Generates dynamic, non-templated AI text responses with context memory.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    reply = chat_agent.generate_chat_response(
        query=req.message,
        context_analysis=req.context_analysis,
        chat_history=req.history,
        persona=req.persona or "patient",
    )
    return {"reply": reply}


@app.get("/api/download/model")
async def download_model():
    """Provides direct download of best trained Drishti AI model weights (.pth)."""
    model_path = PROJECT_ROOT / "artifacts" / "models" / "drishti_best_model.pth"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model weights file not found")
    return FileResponse(
        str(model_path),
        filename="drishti_best_model.pth",
        media_type="application/octet-stream"
    )


# Mount static frontend and data
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR_PATH = PROJECT_ROOT / "data"
ARTIFACTS_DIR_PATH = PROJECT_ROOT / "artifacts"

if DATA_DIR_PATH.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR_PATH)), name="data")

if ARTIFACTS_DIR_PATH.exists():
    app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR_PATH)), name="artifacts")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/favicon.ico")
    async def favicon():
        from fastapi import Response
        return Response(status_code=204)


def start_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    url = f"http://{host}:{port}"
    print(f"\n========================================================")
    print(f"✨ DRISHTI AI Clinical Web Studio is launching!")
    print(f"🔗 Local Web Interface: {url}")
    print(f"========================================================\n")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
