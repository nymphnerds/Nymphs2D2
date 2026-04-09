# Nymphs2D2

`Nymphs2D2` is the planned local 2D image-generation backend for the Nymphs product family.

This repo is intentionally small in `v1`:

- one local FastAPI service
- one configurable Hugging Face model id
- `txt2img` and `img2img`
- output files saved to disk
- API shape that fits the current Nymphs launcher and installer patterns

It is not trying to solve the full long-term vision yet:

- no multiview orchestration yet
- no prompt expansion yet
- no background-removal pipeline yet
- no ControlNet / IP-Adapter yet
- no Blender-specific logic in the backend itself

## Runtime Layout

Expected in-distro layout:

- repo: `~/Nymphs2D2`
- venv: `~/Nymphs2D2/.venv`
- outputs: `~/Nymphs2D2/outputs`
- Hugging Face cache: shared through `~/.cache/huggingface/hub`

## Environment Variables

Supported environment variables:

- `NYMPHS2D2_MODEL_ID`
- `NYMPHS2D2_DEVICE`
- `NYMPHS2D2_DTYPE`
- `NYMPHS2D2_OUTPUT_DIR`
- `NYMPHS2D2_MODEL_VARIANT`
- `NYMPHS2D2_DEFAULT_NEGATIVE_PROMPT`
- `NYMPHS2D2_DEFAULT_STEPS`
- `NYMPHS2D2_DEFAULT_GUIDANCE_SCALE`
- `NYMPHS2D2_DEFAULT_STRENGTH`
- `NYMPHS2D2_MAX_WIDTH`
- `NYMPHS2D2_MAX_HEIGHT`
- `NYMPHS3D_HF_CACHE_DIR`
- `NYMPHS3D_HF_TOKEN`

Useful starting values:

- `NYMPHS2D2_MODEL_ID=Tongyi-MAI/Z-Image-Turbo`
- `NYMPHS2D2_DEVICE=cuda`
- `NYMPHS2D2_DTYPE=bfloat16`
- `NYMPHS2D2_MODEL_VARIANT=` (blank)
- `HF_HUB_DISABLE_XET=1`

## Installation

Install a CUDA-compatible PyTorch build first, then install the repo requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
# install the correct torch build for the target machine first
pip install -r requirements.lock.txt
```

The repo intentionally does not pin `torch` directly because the correct wheel depends on the target CUDA/runtime environment.

Important note:

- the current remake branch uses the latest `diffusers` source build because `Z-Image` support is not present in the older `0.35.1` build that was previously installed in this repo

## Run

```bash
source .venv/bin/activate
python api_server.py --host 0.0.0.0 --port 8090
```

## Prefetch a Model

For larger models, prefetch the snapshot into the shared Hugging Face cache before first use:

```bash
source .venv/bin/activate
python scripts/prefetch_model.py
```

With the current default config, that resolves to `Tongyi-MAI/Z-Image-Turbo` and uses a filtered Z-Image profile. The script forces `HF_HUB_DISABLE_XET=1` unless `--allow-xet` is passed, because that matched the most reliable real-world download behavior on this machine.

Useful examples:

```bash
python scripts/prefetch_model.py --dry-run
python scripts/prefetch_model.py --profile full
python scripts/prefetch_model.py --model-id stabilityai/stable-diffusion-xl-base-1.0 --variant fp16
python scripts/prefetch_model.py --local-files-only
```

## Endpoints

- `GET /health`
- `GET /server_info`
- `GET /active_task`
- `POST /generate`

`POST /generate` supports:

- `mode="txt2img"`
- `mode="img2img"`

Example request:

```json
{
  "mode": "txt2img",
  "prompt": "painted fantasy goblin adventurer, full body, neutral pose, plain background",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "guidance_scale": 0.0,
  "seed": 12345
}
```

Example response:

```json
{
  "status": "ok",
  "worker_id": "abc123",
  "mode": "txt2img",
  "model_id": "Tongyi-MAI/Z-Image-Turbo",
  "output_path": "/home/nymphs3d/Nymphs2D2/outputs/20260408-230000-txt2img-painted-fantasy-1a2b3c4d.png",
  "metadata_path": "/home/nymphs3d/Nymphs2D2/outputs/20260408-230000-txt2img-painted-fantasy-1a2b3c4d.json"
}
```

## Near-Term Intended Use

This backend is meant to become the upstream image-generation service for:

- prompt-to-image generation for `2mv`
- later prompt-to-multiview generation
- later texture-idea generation for Blender-side retexture

The backend is intentionally generic enough that different compatible models can be tested later by changing `NYMPHS2D2_MODEL_ID` instead of rewriting the service.
