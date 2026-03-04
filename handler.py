"""
RunPod Serverless Handler for ColQwen2.5 visual embeddings.

Serves multi-vector embeddings (128-dim per patch/token) for document
page images and text queries. Designed for RunPod serverless deployment.

Operations:
  - embed_query:  text → [num_tokens, 128]
  - embed_image:  base64 PNG → [num_patches, 128]
  - embed_images: list of base64 PNGs → list of [num_patches, 128]
"""

import base64
import io
import logging
import os

import runpod
import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "vidore/colqwen2.5-v0.1"
API_KEY = os.environ.get("EMBEDDING_API_KEY", "")

# ---------------------------------------------------------------------------
# Load model at module level — persists across warm invocations
# ---------------------------------------------------------------------------
logger.info(f"Loading {MODEL_ID}...")
model = ColQwen2_5.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
).eval()
processor = ColQwen2_5_Processor.from_pretrained(MODEL_ID)
logger.info("Model loaded successfully")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_api_key(job_input: dict) -> None:
    """Validate custom API key (defense-in-depth on top of RunPod auth)."""
    if not API_KEY:
        return
    provided = job_input.get("api_key", "")
    if provided != API_KEY:
        raise ValueError("Invalid API key")


def _b64_to_image(b64_str: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _embed_query(job_input: dict) -> dict:
    query = job_input.get("query", "")
    if not query:
        raise ValueError("'query' field is required for embed_query")

    with torch.no_grad():
        batch = processor.process_queries([query])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        embeddings = model(**batch)

    # [1, num_tokens, 128] → [num_tokens, 128]
    result = embeddings[0].cpu().float().tolist()
    return {"embeddings": result}


def _embed_image(job_input: dict) -> dict:
    image_b64 = job_input.get("image_b64", "")
    if not image_b64:
        raise ValueError("'image_b64' field is required for embed_image")

    image = _b64_to_image(image_b64)

    with torch.no_grad():
        batch = processor.process_images([image])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        embeddings = model(**batch)

    # [1, num_patches, 128] → [num_patches, 128]
    result = embeddings[0].cpu().float().tolist()
    return {"embeddings": result}


def _embed_images(job_input: dict) -> dict:
    images_b64 = job_input.get("images_b64", [])
    if not images_b64:
        raise ValueError("'images_b64' list is required for embed_images")

    images = [_b64_to_image(b64) for b64 in images_b64]

    with torch.no_grad():
        batch = processor.process_images(images)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        embeddings = model(**batch)

    # [N, num_patches, 128]
    result = [emb.cpu().float().tolist() for emb in embeddings]
    return {"embeddings": result}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
OPERATIONS = {
    "embed_query": _embed_query,
    "embed_image": _embed_image,
    "embed_images": _embed_images,
}


def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    try:
        _validate_api_key(job_input)

        operation = job_input.get("operation", "")
        if operation not in OPERATIONS:
            return {"error": f"Unknown operation '{operation}'. Valid: {list(OPERATIONS.keys())}"}

        return OPERATIONS[operation](job_input)

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception(f"Handler error: {e}")
        return {"error": f"Internal error: {str(e)}"}


runpod.serverless.start({"handler": handler})
