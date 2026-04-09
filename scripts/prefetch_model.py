from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import get_settings


SDXL_FP16_CORE_PATTERNS = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/*",
    "tokenizer_2/*",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]

ZIMAGE_CORE_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
]

PROFILE_PATTERNS = {
    "sdxl-fp16-core": SDXL_FP16_CORE_PATTERNS,
    "playground-sdxl-fp16": SDXL_FP16_CORE_PATTERNS,
    "zimage-core": ZIMAGE_CORE_PATTERNS,
}


def _default_profile(model_id: str, variant: str | None) -> str:
    normalized_model_id = model_id.lower()
    normalized_variant = (variant or "").lower()

    if normalized_model_id == "playgroundai/playground-v2.5-1024px-aesthetic":
        return "playground-sdxl-fp16"

    if "tongyi-mai/z-image" in normalized_model_id:
        return "zimage-core"

    if normalized_variant in {"fp16", "float16"} and "xl" in normalized_model_id:
        return "sdxl-fp16-core"

    return "full"


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    default_model_id = settings.default_model_id
    default_variant = settings.variant
    default_profile = _default_profile(default_model_id, default_variant)

    parser = argparse.ArgumentParser(
        description="Prefetch a Hugging Face diffusion model into the shared cache.",
    )
    parser.add_argument(
        "--model-id",
        default=default_model_id,
        help="Hugging Face model id to prefetch.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit revision.",
    )
    parser.add_argument(
        "--variant",
        default=default_variant,
        help="Optional model variant such as fp16.",
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "full", *sorted(PROFILE_PATTERNS.keys())],
        default="auto",
        help=(
            "Pattern set for prefetching. "
            f"Default resolves to '{default_profile}' for the current config."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
        help="Explicit Hugging Face cache dir. Defaults to NYMPHS3D_HF_CACHE_DIR when set.",
    )
    parser.add_argument(
        "--token",
        default=settings.hf_token,
        help="Optional Hugging Face token. Defaults to NYMPHS3D_HF_TOKEN if set.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent download workers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be fetched without downloading them.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only already-cached files and fail if the snapshot is incomplete.",
    )
    parser.add_argument(
        "--allow-xet",
        action="store_true",
        help="Do not force HF_HUB_DISABLE_XET=1 for this prefetch run.",
    )
    return parser.parse_args()


def _resolve_profile(model_id: str, variant: str | None, requested_profile: str) -> str:
    if requested_profile != "auto":
        return requested_profile
    return _default_profile(model_id, variant)


def _prepare_environment(*, allow_xet: bool) -> None:
    if not allow_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _format_patterns(patterns: list[str] | None) -> str:
    if not patterns:
        return "full snapshot"
    return ", ".join(patterns)


def main() -> int:
    args = _parse_args()
    profile = _resolve_profile(args.model_id, args.variant, args.profile)
    allow_patterns = PROFILE_PATTERNS.get(profile)

    _prepare_environment(allow_xet=args.allow_xet)

    import huggingface_hub
    from huggingface_hub import snapshot_download

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    supports_dry_run = "dry_run" in inspect.signature(snapshot_download).parameters

    print(f"model_id={args.model_id}")
    print(f"revision={args.revision or 'main'}")
    print(f"profile={profile}")
    print(f"patterns={_format_patterns(allow_patterns)}")
    print(f"cache_dir={cache_dir or 'default HF cache'}")
    print(f"local_files_only={args.local_files_only}")
    print(f"dry_run={args.dry_run}")
    print(f"HF_HUB_DISABLE_XET={os.getenv('HF_HUB_DISABLE_XET', '0')}")
    print(f"huggingface_hub={huggingface_hub.__version__}")
    sys.stdout.flush()

    if args.dry_run and not supports_dry_run:
        print(
            "dry_run_requested_but_unsupported=true "
            f"(huggingface_hub {huggingface_hub.__version__})"
        )
        return 2

    snapshot_kwargs = {
        "repo_id": args.model_id,
        "revision": args.revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "token": args.token,
        "local_files_only": args.local_files_only,
        "allow_patterns": allow_patterns,
        "max_workers": args.max_workers,
    }

    if supports_dry_run:
        snapshot_kwargs["dry_run"] = args.dry_run

    result = snapshot_download(**snapshot_kwargs)

    if args.dry_run:
        print(f"dry_run_files={len(result)}")
        for entry in result:
            print(f"- {getattr(entry, 'file_name', str(entry))}")
    else:
        print(f"snapshot_path={result}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
