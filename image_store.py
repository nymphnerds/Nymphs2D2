from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _slugify(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned[:48] or fallback


def save_image_and_metadata(image, output_dir: Path, *, mode: str, prompt: str, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = _slugify(prompt, mode)
    suffix = uuid.uuid4().hex[:8]
    image_path = output_dir / f"{timestamp}-{mode}-{slug}-{suffix}.png"
    metadata_path = image_path.with_suffix(".json")

    image.save(image_path, format="PNG")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return image_path, metadata_path
