"""
Gemma 4 + SAM 3.1 FastAPI Backend
====================================
Serves segmentation and analysis endpoints for the SwiftUI macOS app.
Both models run locally on Apple Silicon via MLX.
"""

import io
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .models import ModelManager

# Global model manager — loaded on startup
manager = ModelManager()
_load_thread = None


def _load_models_background():
    """Load models in a background thread so the server starts immediately."""
    manager.load_all()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start model loading in background on startup."""
    global _load_thread
    _load_thread = threading.Thread(target=_load_models_background, daemon=True)
    _load_thread.start()
    yield


app = FastAPI(
    title="Gemma 4 + SAM 3.1 Segmentation API",
    description="Local vision-guided segmentation on Apple Silicon via MLX",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow local SwiftUI app to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_upload_image(file: UploadFile) -> Image.Image:
    """Read an uploaded file into a PIL Image."""
    contents = file.file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Check model loading status."""
    return manager.status()


@app.post("/api/segment")
async def segment(
    image: UploadFile = File(...),
    prompt: str = Form(...),
):
    """
    Segment objects in the image using SAM 3.1.

    Returns masks as base64-encoded PNG images, bounding boxes, and scores.
    """
    if not manager.sam.is_loaded:
        return {"error": "SAM 3.1 model not loaded yet", "status": manager.status()}

    img = _read_upload_image(image)
    result = manager.sam.segment(img, prompt)
    return result


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(None),
):
    """
    Analyze the image using Gemma 4.

    If prompt is provided, Gemma translates it into a segmentation instruction.
    If no prompt, Gemma auto-analyzes and suggests what to segment.
    Returns the model's text response and any extracted action JSON.
    """
    if not manager.gemma.is_loaded:
        return {"error": "Gemma 4 model not loaded yet", "status": manager.status()}

    img = _read_upload_image(image)

    if prompt:
        result = manager.gemma.analyze_with_scene(img, prompt)
    else:
        result = manager.gemma.auto_analyze(img)

    return {
        "text": result["text"],
        "action": result["action"],
    }


@app.post("/api/analyze-and-segment")
async def analyze_and_segment(
    image: UploadFile = File(...),
    prompt: str = Form(None),
):
    """
    One-shot: Gemma 4 analyzes the image and produces a SAM prompt,
    then SAM 3.1 executes the segmentation.

    If prompt is provided, Gemma translates it into a SAM-friendly prompt.
    If no prompt, Gemma decides what to segment.
    """
    if not manager.gemma.is_loaded or not manager.sam.is_loaded:
        return {"error": "Models not loaded yet", "status": manager.status()}

    img = _read_upload_image(image)

    if prompt:
        result = manager.gemma.analyze_with_scene(img, prompt)
    else:
        result = manager.gemma.auto_analyze(img)

    action = result["action"]
    gemma_text = result["text"]

    if action and action.get("action") == "segment":
        seg_prompt = action["prompt"]
    elif prompt:
        # Fallback: use the user's prompt directly with SAM
        seg_prompt = prompt
    else:
        return {
            "text": gemma_text,
            "action": action,
            "segmentation": None,
            "error": "Gemma 4 did not produce a valid segmentation prompt",
        }

    # Run SAM 3.1 segmentation
    seg_result = manager.sam.segment(img, seg_prompt)

    return {
        "text": gemma_text,
        "action": action,
        "segmentation": seg_result,
    }


@app.get("/api/health")
async def health():
    """Simple health check."""
    return {"ok": True}
