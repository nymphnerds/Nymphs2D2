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
        self._loaded_runtime = None
        self._loaded_runtime_extra = {}

    @property
    def loaded_model_id(self) -> str | None:
        return self._loaded_model_id

    @property
    def loaded_runtime(self) -> str | None:
        return self._loaded_runtime

    @property
    def loaded_runtime_extra(self) -> dict:
        return dict(self._loaded_runtime_extra)

    def _model_family(self, model_id: str | None) -> str:
        normalized = (model_id or self.settings.default_model_id or "").strip().lower()
        if "z-image" in normalized:
            return "zimage"
        return "generic"

    def _is_zimage_turbo_model(self, model_id: str | None) -> bool:
        normalized = (model_id or self.settings.default_model_id or "").strip().lower()
        return normalized.endswith("/z-image-turbo")

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

    def _resolve_runtime(self, model_id: str | None) -> str:
        runtime = (self.settings.runtime or "standard").strip().lower()
        if runtime != "nunchaku":
            return "standard"
        if self._model_family(model_id) != "zimage" or not self._is_zimage_turbo_model(model_id):
            raise RuntimeError("Nunchaku runtime currently supports Tongyi-MAI/Z-Image-Turbo only.")
        return "nunchaku"

    def supports_img2img(self, requested_model_id: str | None = None) -> bool:
        return self._resolve_runtime(requested_model_id or self.settings.default_model_id) != "nunchaku"

    def supported_modes(self, requested_model_id: str | None = None) -> list[str]:
        if self.supports_img2img(requested_model_id):
            return ["txt2img", "img2img"]
        return ["txt2img"]

    def _pipeline_kwargs(self, model_id: str | None, runtime: str) -> dict:
        model_family = self._model_family(model_id)
        kwargs = {
            "torch_dtype": self._dtype,
        }
        if model_family == "zimage":
            kwargs["low_cpu_mem_usage"] = False
        elif runtime != "nunchaku" and self.settings.variant:
            kwargs["variant"] = self.settings.variant
        if self.settings.hf_cache_dir:
            kwargs["cache_dir"] = str(self.settings.hf_cache_dir)
        if self.settings.hf_token:
            kwargs["token"] = self.settings.hf_token
        if self.settings.use_safetensors:
            kwargs["use_safetensors"] = True
        return kwargs

    def _prepare_pipeline(self, pipeline, runtime: str):
        if runtime == "nunchaku":
            if self.settings.device != "cpu" and hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload()
            elif self.settings.device:
                pipeline = pipeline.to(self.settings.device)
            return pipeline

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
        self._loaded_runtime = None
        self._loaded_runtime_extra = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_nunchaku_dtype(self):
        if self.settings.device == "cpu":
            return torch.float32
        if self._dtype == torch.float32:
            return torch.float32
        try:
            from nunchaku.utils import is_turing

            if is_turing(self.settings.device):
                return torch.float16
        except Exception:
            pass
        return self._dtype

    def _nunchaku_rank_path(self) -> tuple[str, str]:
        from nunchaku.utils import get_precision

        precision = self.settings.nunchaku_precision or "auto"
        if precision == "auto":
            precision = get_precision(precision="auto", device=self.settings.device)
        rank_path = (
            f"{self.settings.nunchaku_model_repo}/"
            f"svdq-{precision}_r{self.settings.nunchaku_rank}-z-image-turbo.safetensors"
        )
        return rank_path, precision

    def _load_txt2img_pipeline(self, model_id: str, runtime: str):
        if runtime == "nunchaku":
            try:
                from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
                from nunchaku import NunchakuZImageTransformer2DModel
            except ImportError as exc:
                raise RuntimeError("Nunchaku runtime dependencies are not installed in this environment.") from exc

            rank_path, precision = self._nunchaku_rank_path()
            dtype = self._resolve_nunchaku_dtype()
            transformer = NunchakuZImageTransformer2DModel.from_pretrained(rank_path, torch_dtype=dtype)
            self._loaded_runtime_extra = {
                "runtime": "nunchaku",
                "nunchaku_rank": self.settings.nunchaku_rank,
                "nunchaku_precision": precision,
                "nunchaku_rank_path": rank_path,
                "runtime_dtype": str(dtype).replace("torch.", ""),
            }
            return ZImagePipeline.from_pretrained(
                model_id,
                transformer=transformer,
                **self._pipeline_kwargs(model_id, runtime),
            )

        if self._model_family(model_id) == "zimage":
            try:
                from diffusers import ZImagePipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Current diffusers build does not include Z-Image support. "
                    "Install a newer diffusers build before loading Tongyi-MAI/Z-Image models."
                ) from exc
            return ZImagePipeline.from_pretrained(model_id, **self._pipeline_kwargs(model_id, runtime))

        from diffusers import AutoPipelineForText2Image

        return AutoPipelineForText2Image.from_pretrained(model_id, **self._pipeline_kwargs(model_id, runtime))

    def ensure_model(self, requested_model_id: str | None = None) -> str:
        model_id = requested_model_id or self.settings.default_model_id
        with self._lock:
            if self._txt2img is not None and self._loaded_model_id == model_id:
                return model_id

            runtime = self._resolve_runtime(model_id)
            self._unload_pipelines()
            self._txt2img = self._load_txt2img_pipeline(model_id, runtime)
            self._txt2img = self._prepare_pipeline(self._txt2img, runtime)
            self._loaded_model_id = model_id
            self._loaded_model_family = self._model_family(model_id)
            self._loaded_runtime = runtime
            return model_id

    def _ensure_img2img(self):
        if self._img2img is not None:
            return self._img2img

        if self._loaded_runtime == "nunchaku":
            raise RuntimeError("Nunchaku runtime currently supports txt2img only in Nymphs2D2.")

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
                **self._pipeline_kwargs(self._loaded_model_id, self._loaded_runtime or "standard"),
            )
            self._img2img = self._prepare_pipeline(self._img2img, self._loaded_runtime or "standard")
            return self._img2img

        from diffusers import AutoPipelineForImage2Image

        try:
            self._img2img = AutoPipelineForImage2Image.from_pipe(self._txt2img)
            self._img2img = self._prepare_pipeline(self._img2img, self._loaded_runtime or "standard")
        except AttributeError:
            self._img2img = AutoPipelineForImage2Image.from_pretrained(
                self._loaded_model_id,
                **self._pipeline_kwargs(self._loaded_model_id, self._loaded_runtime or "standard"),
            )
            self._img2img = self._prepare_pipeline(self._img2img, self._loaded_runtime or "standard")
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
            kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if self._loaded_runtime != "nunchaku":
                kwargs["negative_prompt"] = negative_prompt
            result = self._txt2img(**kwargs)
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
