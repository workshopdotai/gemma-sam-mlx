"""
Gemma 4 + SAM 3.1 Model Wrappers
===================================
Model classes for FastAPI backend.
Adds serialization (masks → base64 PNG) for HTTP transfer.
"""

import base64
import io
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ─── System prompt for Gemma 4 ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a vision assistant that helps segment objects in images. "
    "You receive a description of an image scene and a user request. "
    "Your job is to output a JSON object telling the segmentation model what to find.\n\n"
    "Rules:\n"
    "- Use simple plural nouns for the prompt (e.g. \"cars\", \"people\", \"buildings\").\n"
    "- If the user asks for multiple things, combine them with \"and\" (e.g. \"cars and people\").\n"
    "- Output ONLY the JSON object, nothing else."
)

# ─── Gemma 4: Vision-Language Reasoning ─────────────────────────────────────

class Gemma4Agent:
    """Uses Gemma 4 (via mlx-vlm) to analyze images and decide what to segment."""

    def __init__(self, model_id: str = "mlx-community/gemma-4-26b-a4b-it-4bit"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        """Load the model and processor."""
        if self._loaded:
            return
        print(f"[🧠] Loading Gemma 4: {self.model_id} ...")
        from mlx_vlm import load as vl_load
        self.model, self.processor = vl_load(self.model_id)
        self._loaded = True
        print(f"[✅] Gemma 4 loaded.")

    def _format_prompt(self, system: str, user: str) -> str:
        """Format a prompt using the processor's chat template.
        
        Uses apply_chat_template for proper Gemma 4 formatting including
        system messages, turn markers, and generation prompts.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def analyze(self, image: Image.Image, prompt: str) -> str:
        """Send an image + prompt to Gemma 4 and get a response.
        
        The prompt should already be formatted with chat template if needed.
        Use _format_prompt() for structured prompts.
        """
        from mlx_vlm import generate as vl_generate

        if not self._loaded:
            self.load()

        output = vl_generate(
            self.model,
            self.processor,
            prompt=prompt,
            image=[image],
            max_tokens=256,
            verbose=False,
            repetition_penalty=1.5,
            repetition_context_size=64,
        )
        text = output.text if hasattr(output, 'text') else str(output)
        text = self._clean_repetition(text)
        return text.strip()

    def analyze_with_scene(self, image: Image.Image, user_prompt: str) -> dict:
        """Analyze the image with a user prompt, producing a SAM segmentation instruction.
        
        Uses a two-step approach:
        1. First, describe the scene (what's in the image)
        2. Then, translate the user's request into a SAM-friendly prompt
        
        Returns dict with: 'text' (full response), 'action' (extracted JSON action)
        """
        # Step 1: Describe the scene and generate the segmentation prompt in one call
        full_prompt = self._format_prompt(
            SYSTEM_PROMPT,
            f"User request: \"{user_prompt}\"\n\n"
            f'Look at the image and output JSON: {{\"action\": \"segment\", \"prompt\": \"<objects to find>\"}}'
        )
        
        response = self.analyze(image, full_prompt)
        action = self.extract_action(response)
        
        return {
            "text": response,
            "action": action,
        }

    def auto_analyze(self, image: Image.Image) -> dict:
        """Analyze the image without a user prompt — let Gemma decide what to segment."""
        full_prompt = self._format_prompt(
            SYSTEM_PROMPT,
            "Look at the image. Decide what objects would be most interesting to segment, "
            'then output JSON: {"action": "segment", "prompt": "<objects>"}'
        )
        
        response = self.analyze(image, full_prompt)
        action = self.extract_action(response)
        
        return {
            "text": response,
            "action": action,
        }

    @staticmethod
    def _clean_repetition(text: str) -> str:
        """Detect and truncate repetitive token loops and strip Gemma 4 special tokens."""
        import re as _re
        
        # Strip Gemma 4 thought channel output: <|channel>thought\n...<channel|>
        text = _re.sub(r'<\|channel\>thought\n.*?<channel\|>', '', text, flags=_re.DOTALL)
        
        # Strip any remaining special tokens
        text = _re.sub(r'<\|[^>]+\>', '', text)
        text = _re.sub(r'<[^>]+\|>', '', text)
        
        # Pattern: a short token repeated 3+ times consecutively
        if _re.search(r'(.{1,20}?)\1{2,}', text):
            match = _re.search(r'(.{1,20}?)\1{2,}', text)
            if match:
                text = text[:match.start()]
        return text

    def auto_analyze(self, image: Image.Image) -> str:
        """Analyze the image and decide what to segment."""
        prompt = (
            "Look at this image. Describe what you see, then decide what to segment. "
            "Output ONLY a JSON object with your segmentation request:\n"
            '{"action": "segment", "prompt": "<what to segment>"}\n'
            "Do not output anything after the JSON."
        )
        return self.analyze(image, prompt)

    def interpret_results(self, image: Image.Image, previous_prompt: str, mask_count: int) -> str:
        """Ask Gemma 4 to interpret segmentation results and potentially refine the query."""
        prompt = self._format_prompt(
            SYSTEM_PROMPT,
            f"You segmented '{previous_prompt}' and found {mask_count} objects. "
            f"Look at the image. Want to refine?\n"
            f'Output JSON: {{"action": "segment", "prompt": "<refined>"}} or {{"action": "done", "summary": "<brief summary>"}}\n'
            f"Do not output anything after the JSON."
        )
        return self.analyze(image, prompt)

    @staticmethod
    def extract_action(text: str) -> Optional[dict]:
        """Extract a JSON action from the model's text output."""
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if not json_blocks:
            json_blocks = re.findall(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)

        for block in json_blocks:
            try:
                parsed = json.loads(block)
                if "action" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if "action" in parsed:
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        return None


# ─── SAM 3.1: Segmentation Execution ──────────────────────────────────────

class SAM31Segmenter:
    """
    Uses SAM 3.1 (via MLX) to segment objects in images based on text prompts.
    """

    def __init__(self, model_id: str = "mlx-community/sam3.1-bf16"):
        self.model_id = model_id
        self.predictor = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        """Load the SAM 3.1 model."""
        if self._loaded:
            return
        print(f"[🎯] Loading SAM 3.1: {self.model_id} ...")
        from mlx_vlm.utils import load_model, get_model_path
        from mlx_vlm.models.sam3.generate import Sam3Predictor
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor

        model_path = get_model_path(self.model_id)
        model = load_model(model_path)
        processor = Sam31Processor.from_pretrained(str(model_path))
        self.predictor = Sam3Predictor(model, processor, score_threshold=0.3)
        self._loaded = True
        print(f"[✅] SAM 3.1 loaded.")

    def segment(self, image: Image.Image, text_prompt: str) -> dict:
        """
        Segment objects in the image based on a text prompt.

        Returns dict with:
          - count: number of objects found
          - boxes: bounding boxes as list of [x1, y1, x2, y2]
          - masks: base64-encoded PNG masks (for HTTP transfer)
          - scores: confidence scores as list of floats
          - text_prompt: the prompt used
        """
        if not self._loaded:
            self.load()

        result = self.predictor.predict(image, text_prompt=text_prompt)

        count = len(result.scores)
        print(f"[🎯] SAM 3.1 found {count} objects for '{text_prompt}'")
        for i in range(min(count, 5)):
            x1, y1, x2, y2 = result.boxes[i]
            print(f"     [{result.scores[i]:.2f}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        if count > 5:
            print(f"     ... and {count - 5} more")

        W, H = image.size

        # Convert masks to base64 PNG for efficient HTTP transfer
        mask_b64_list = []
        for mask in result.masks:
            # mask is numpy array (H, W), convert to binary PNG
            binary = (mask > 0).astype(np.uint8) * 255
            mask_img = Image.fromarray(binary, mode='L')
            buf = io.BytesIO()
            mask_img.save(buf, format='PNG', optimize=True)
            mask_b64_list.append(base64.b64encode(buf.getvalue()).decode('ascii'))

        # Convert boxes to list of lists for JSON serialization
        boxes_list = []
        for box in result.boxes:
            boxes_list.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])

        scores_list = [float(s) for s in result.scores]

        return {
            "count": count,
            "boxes": boxes_list,
            "masks": mask_b64_list,
            "scores": scores_list,
            "text_prompt": text_prompt,
            "image_width": W,
            "image_height": H,
        }


# ─── Model Manager ──────────────────────────────────────────────────────────

class ModelManager:
    """Manages loading and access to both models."""

    def __init__(self, gemma_model_id: str = None, sam_model_id: str = None):
        self.gemma = Gemma4Agent(model_id=gemma_model_id or "mlx-community/gemma-4-26b-a4b-it-4bit")
        self.sam = SAM31Segmenter(model_id=sam_model_id or "mlx-community/sam3.1-bf16")

    def load_all(self):
        """Load both models."""
        self.gemma.load()
        self.sam.load()

    def status(self) -> dict:
        """Return loading status of both models."""
        return {
            "gemma_loaded": self.gemma.is_loaded,
            "sam_loaded": self.sam.is_loaded,
            "gemma_model": self.gemma.model_id,
            "sam_model": self.sam.model_id,
            "status": "ready" if (self.gemma.is_loaded and self.sam.is_loaded) else "loading",
        }