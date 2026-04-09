from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_RUNTIME = "standard"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _normalize_model_id(model_id: str | None) -> str:
    return (model_id or DEFAULT_MODEL_ID).strip()


def _normalize_runtime(runtime: str | None) -> str:
    normalized = (runtime or DEFAULT_RUNTIME).strip().lower()
    if normalized == "nunchaku":
        return "nunchaku"
    return "standard"


def _is_zimage_model(model_id: str | None) -> bool:
    return "z-image" in _normalize_model_id(model_id).lower()


def _is_zimage_turbo_model(model_id: str | None) -> bool:
    return _normalize_model_id(model_id).lower().endswith("/z-image-turbo")


def _default_dtype_for_model(model_id: str | None) -> str:
    if _is_zimage_model(model_id):
        return "bfloat16"
    return "float16"


def _default_variant_for_model(model_id: str | None) -> str | None:
    if _is_zimage_model(model_id):
        return None
    if _normalize_model_id(model_id).lower() == "playgroundai/playground-v2.5-1024px-aesthetic":
        return "fp16"
    return None


def _default_steps_for_model(model_id: str | None) -> int:
    if _is_zimage_turbo_model(model_id):
        return 9
    return 30


def _default_guidance_for_model(model_id: str | None) -> float:
    if _is_zimage_turbo_model(model_id):
        return 0.0
    return 3.0


def _default_strength_for_model(model_id: str | None) -> float:
    if _is_zimage_model(model_id):
        return 0.6
    return 0.45


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    output_dir: Path
    host: str
    port: int
    default_model_id: str
    runtime: str
    default_negative_prompt: str
    device: str
    dtype: str
    variant: str | None
    nunchaku_rank: int
    nunchaku_precision: str
    nunchaku_model_repo: str
    use_safetensors: bool
    hf_cache_dir: Path | None
    hf_token: str | None
    max_width: int
    max_height: int
    default_steps: int
    default_guidance_scale: float
    default_strength: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parent
    output_dir = Path(os.getenv("NYMPHS2D2_OUTPUT_DIR", root_dir / "outputs")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_cache_raw = os.getenv("NYMPHS3D_HF_CACHE_DIR")
    hf_cache_dir = Path(hf_cache_raw).expanduser() if hf_cache_raw else None
    default_model_id = os.getenv("NYMPHS2D2_MODEL_ID", DEFAULT_MODEL_ID)
    runtime = _normalize_runtime(os.getenv("NYMPHS2D2_RUNTIME", DEFAULT_RUNTIME))

    return Settings(
        root_dir=root_dir,
        output_dir=output_dir,
        host=os.getenv("NYMPHS2D2_HOST", "0.0.0.0"),
        port=int(os.getenv("NYMPHS2D2_PORT", "8090")),
        default_model_id=default_model_id,
        runtime=runtime,
        default_negative_prompt=os.getenv("NYMPHS2D2_DEFAULT_NEGATIVE_PROMPT", ""),
        device=os.getenv("NYMPHS2D2_DEVICE", _default_device()),
        dtype=os.getenv("NYMPHS2D2_DTYPE", _default_dtype_for_model(default_model_id)),
        variant=os.getenv("NYMPHS2D2_MODEL_VARIANT") or _default_variant_for_model(default_model_id),
        nunchaku_rank=int(os.getenv("NYMPHS2D2_NUNCHAKU_RANK", "32")),
        nunchaku_precision=(os.getenv("NYMPHS2D2_NUNCHAKU_PRECISION", "auto") or "auto").strip().lower(),
        nunchaku_model_repo=os.getenv("NYMPHS2D2_NUNCHAKU_MODEL_REPO", "nunchaku-ai/nunchaku-z-image-turbo"),
        use_safetensors=_env_bool("NYMPHS2D2_USE_SAFETENSORS", True),
        hf_cache_dir=hf_cache_dir,
        hf_token=os.getenv("NYMPHS3D_HF_TOKEN") or None,
        max_width=int(os.getenv("NYMPHS2D2_MAX_WIDTH", "1536")),
        max_height=int(os.getenv("NYMPHS2D2_MAX_HEIGHT", "1536")),
        default_steps=int(os.getenv("NYMPHS2D2_DEFAULT_STEPS", str(_default_steps_for_model(default_model_id)))),
        default_guidance_scale=float(
            os.getenv("NYMPHS2D2_DEFAULT_GUIDANCE_SCALE", str(_default_guidance_for_model(default_model_id)))
        ),
        default_strength=float(os.getenv("NYMPHS2D2_DEFAULT_STRENGTH", str(_default_strength_for_model(default_model_id)))),
    )
