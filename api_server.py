from __future__ import annotations

import argparse
import base64
import uuid
from io import BytesIO

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image

from config import get_settings
from image_store import save_image_and_metadata
from model_manager import ModelManager
from progress_state import reset as progress_reset
from progress_state import snapshot as progress_snapshot
from progress_state import update as progress_update
from schemas import (
    ActiveTaskResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ServerInfoResponse,
)


VERSION = "0.1.0"
WORKER_ID = uuid.uuid4().hex[:6]
SETTINGS = get_settings()
MODEL_MANAGER = ModelManager(SETTINGS)

app = FastAPI(
    title="Nymphs2D2 API",
    version=VERSION,
    description="Local Nymphs Stable Diffusion-style backend scaffold.",
)


def _coerce_dimension(value: int, *, maximum: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    if value > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return max(64, value - (value % 8))


def _decode_base64_image(raw: str) -> Image.Image:
    payload = raw.split(",", 1)[1] if "," in raw else raw
    try:
        image = Image.open(BytesIO(base64.b64decode(payload)))
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def _normalize_request(payload: GenerateRequest) -> GenerateRequest:
    width = _coerce_dimension(payload.width, maximum=SETTINGS.max_width, label="width")
    height = _coerce_dimension(payload.height, maximum=SETTINGS.max_height, label="height")
    steps = payload.steps or SETTINGS.default_steps
    guidance_scale = payload.guidance_scale if payload.guidance_scale is not None else SETTINGS.default_guidance_scale
    strength = payload.strength if payload.strength is not None else SETTINGS.default_strength

    if steps <= 0:
        raise ValueError("steps must be greater than zero.")
    if payload.mode == "img2img" and not payload.image:
        raise ValueError("img2img mode requires an input image.")
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    return payload.model_copy(
        update={
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "negative_prompt": payload.negative_prompt or SETTINGS.default_negative_prompt,
        }
    )


def _generate(payload: GenerateRequest) -> GenerateResponse:
    progress_update(
        status="processing",
        stage="loading_model",
        detail="Loading or reusing model",
        model_id=payload.model_id or SETTINGS.default_model_id,
        progress_current=0,
        progress_total=3,
        progress_percent=0.0,
    )

    if payload.mode == "txt2img":
        progress_update(
            status="processing",
            stage="generating_image",
            detail="Running txt2img",
            progress_current=1,
            progress_total=3,
            progress_percent=33.0,
        )
        image, model_id = MODEL_MANAGER.generate_text_to_image(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            guidance_scale=payload.guidance_scale,
            seed=payload.seed,
            model_id=payload.model_id,
        )
    else:
        init_image = _decode_base64_image(payload.image or "")
        progress_update(
            status="processing",
            stage="generating_image",
            detail="Running img2img",
            progress_current=1,
            progress_total=3,
            progress_percent=33.0,
        )
        image, model_id = MODEL_MANAGER.generate_image_to_image(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            image=init_image,
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            guidance_scale=payload.guidance_scale,
            strength=payload.strength,
            seed=payload.seed,
            model_id=payload.model_id,
        )

    progress_update(
        status="processing",
        stage="saving_output",
        detail="Saving output image",
        model_id=model_id,
        progress_current=2,
        progress_total=3,
        progress_percent=66.0,
    )

    metadata = {
        "backend": "Nymphs2D2",
        "version": VERSION,
        "worker_id": WORKER_ID,
        "mode": payload.mode,
        "model_id": model_id,
        "prompt": payload.prompt,
        "negative_prompt": payload.negative_prompt,
        "width": payload.width,
        "height": payload.height,
        "steps": payload.steps,
        "guidance_scale": payload.guidance_scale,
        "seed": payload.seed,
        "strength": payload.strength,
    }
    output_path, metadata_path = save_image_and_metadata(
        image,
        SETTINGS.output_dir,
        mode=payload.mode,
        prompt=payload.prompt,
        metadata=metadata,
    )

    progress_update(
        status="idle",
        stage="complete",
        detail="Generation complete",
        model_id=model_id,
        progress_current=3,
        progress_total=3,
        progress_percent=100.0,
        last_output_path=str(output_path),
    )
    progress_reset()
    progress_update(last_output_path=str(output_path), model_id=model_id)

    return GenerateResponse(
        status="ok",
        worker_id=WORKER_ID,
        mode=payload.mode,
        model_id=model_id,
        output_path=str(output_path),
        metadata_path=str(metadata_path),
    )


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    return JSONResponse({"status": "healthy", "worker_id": WORKER_ID}, status_code=200)


@app.get("/server_info", response_model=ServerInfoResponse, tags=["status"])
async def server_info():
    return ServerInfoResponse(
        backend="Nymphs2D2",
        version=VERSION,
        worker_id=WORKER_ID,
        configured_model_id=SETTINGS.default_model_id,
        loaded_model_id=MODEL_MANAGER.loaded_model_id,
        device=SETTINGS.device,
        dtype=SETTINGS.dtype,
        output_dir=str(SETTINGS.output_dir),
        supported_modes=["txt2img", "img2img"],
        extra={
            "max_width": SETTINGS.max_width,
            "max_height": SETTINGS.max_height,
        },
    )


@app.get("/active_task", response_model=ActiveTaskResponse, tags=["status"])
async def active_task():
    return ActiveTaskResponse(**progress_snapshot())


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate(payload: GenerateRequest):
    try:
        normalized = _normalize_request(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return await run_in_threadpool(_generate, normalized)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        progress_update(status="error", stage="failed", detail=str(exc))
        raise HTTPException(status_code=500, detail="Image generation failed.") from exc


def main():
    parser = argparse.ArgumentParser(description="Run the Nymphs2D2 API server.")
    parser.add_argument("--host", default=SETTINGS.host)
    parser.add_argument("--port", type=int, default=SETTINGS.port)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
