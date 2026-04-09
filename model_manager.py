from __future__ import annotations

import gc
from threading import RLock

import torch

from config import Settings


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._txt2img = None
        self._img2img = None
        self._loaded_model_id = None
        self._dtype = self._resolve_torch_dtype(settings.dtype)
        if settings.device == "cpu" and self._dtype != torch.float32:
            self._dtype = torch.float32
        self._loaded_model_family = None

    @property
    def loaded_model_id(self) -> str | None:
        return self._loaded_model_id

    def _model_family(self, model_id: str | None) -> str:
        normalized = (model_id or self.settings.default_model_id or "").strip().lower()
        if "z-image" in normalized:
            return "zimage"
        return "generic"

    def _resolve_torch_dtype(self, dtype_name: str):
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(dtype_name.lower(), torch.float16)

    def _pipeline_kwargs(self, model_id: str | None) -> dict:
        model_family = self._model_family(model_id)
        kwargs = {
            "torch_dtype": self._dtype,
        }
        if model_family == "zimage":
            kwargs["low_cpu_mem_usage"] = False
        elif self.settings.variant:
            kwargs["variant"] = self.settings.variant
        if self.settings.hf_cache_dir:
            kwargs["cache_dir"] = str(self.settings.hf_cache_dir)
        if self.settings.hf_token:
            kwargs["token"] = self.settings.hf_token
        if self.settings.use_safetensors:
            kwargs["use_safetensors"] = True
        return kwargs

    def _prepare_pipeline(self, pipeline):
        if self.settings.device:
            pipeline = pipeline.to(self.settings.device)
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        return pipeline

    def _unload_pipelines(self):
        self._txt2img = None
        self._img2img = None
        self._loaded_model_id = None
        self._loaded_model_family = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_txt2img_pipeline(self, model_id: str):
        if self._model_family(model_id) == "zimage":
            try:
                from diffusers import ZImagePipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Current diffusers build does not include Z-Image support. "
                    "Install a newer diffusers build before loading Tongyi-MAI/Z-Image models."
                ) from exc
            return ZImagePipeline.from_pretrained(model_id, **self._pipeline_kwargs(model_id))

        from diffusers import AutoPipelineForText2Image

        return AutoPipelineForText2Image.from_pretrained(model_id, **self._pipeline_kwargs(model_id))

    def ensure_model(self, requested_model_id: str | None = None) -> str:
        model_id = requested_model_id or self.settings.default_model_id
        with self._lock:
            if self._txt2img is not None and self._loaded_model_id == model_id:
                return model_id

            self._unload_pipelines()
            self._txt2img = self._load_txt2img_pipeline(model_id)
            self._txt2img = self._prepare_pipeline(self._txt2img)
            self._loaded_model_id = model_id
            self._loaded_model_family = self._model_family(model_id)
            return model_id

    def _ensure_img2img(self):
        if self._img2img is not None:
            return self._img2img

        if self._loaded_model_family == "zimage":
            # Z-Image img2img uses a separate pipeline class. Drop the txt2img
            # pipeline first so we do not keep two full 6B-class pipelines on
            # the GPU at once during iterative edit workflows.
            self._txt2img = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                from diffusers import ZImageImg2ImgPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Current diffusers build does not include Z-Image img2img support. "
                    "Install a newer diffusers build before using Tongyi-MAI/Z-Image models."
                ) from exc
            self._img2img = ZImageImg2ImgPipeline.from_pretrained(
                self._loaded_model_id,
                **self._pipeline_kwargs(self._loaded_model_id),
            )
            self._img2img = self._prepare_pipeline(self._img2img)
            return self._img2img

        from diffusers import AutoPipelineForImage2Image

        try:
            self._img2img = AutoPipelineForImage2Image.from_pipe(self._txt2img)
            self._img2img = self._prepare_pipeline(self._img2img)
        except AttributeError:
            self._img2img = AutoPipelineForImage2Image.from_pretrained(
                self._loaded_model_id,
                **self._pipeline_kwargs(self._loaded_model_id),
            )
            self._img2img = self._prepare_pipeline(self._img2img)
        return self._img2img

    def _build_generator(self, seed: int | None):
        if seed is None:
            return None

        device = self.settings.device if self.settings.device != "mps" else "cpu"
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def generate_text_to_image(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        model_id: str | None,
    ):
        with self._lock:
            active_model_id = self.ensure_model(model_id)
            generator = self._build_generator(seed)
            result = self._txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            return result.images[0], active_model_id

    def generate_image_to_image(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        strength: float,
        seed: int | None,
        model_id: str | None,
    ):
        with self._lock:
            active_model_id = self.ensure_model(model_id)
            pipeline = self._ensure_img2img()
            generator = self._build_generator(seed)
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            )
            return result.images[0], active_model_id
