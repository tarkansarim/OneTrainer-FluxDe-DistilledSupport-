import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import torch
from PIL import Image

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from modules.dataLoader.pipelineModules.DetailCropGenerator import DetailCropGenerator, DetailEntry
from modules.util import ollama_manager

try:
    import ollama as _OLLAMA_MODULE  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _OLLAMA_MODULE = None


_DEFAULT_CAPTION_MAX_TOKENS = 256


class CropCaptionGenerator(PipelineModule, RandomAccessPipelineModule):
    """
    Generates captions for detail crops on-the-fly using a local Ollama endpoint.

    Captions are produced deterministically per crop and cached based on the crop metadata and prompt templates.
    """

    def __init__(
            self,
            image_name: str,
            concept_name: str,
            prompt_name: str,
            image_path_name: str,
            additional_passthrough: Optional[Iterable[str]] = None,
            parallel_workers: int = 0,
    ):
        self.image_name = image_name
        self.concept_name = concept_name
        self.prompt_name = prompt_name
        self.image_path_name = image_path_name
        self.additional_passthrough = tuple(sorted(set(additional_passthrough or [])))
        self.parallel_workers = parallel_workers
        super().__init__()

        self._memory_cache: Dict[str, str] = {}
        self._session = requests.Session()
        self._models_pulled: set[str] = set()
        self._current_variation: Optional[int] = None
        self._metrics = self._new_metrics()
        self._progress_prefix: str = ""
        self._progress_total: int = 0
        self._progress_count: int = 0
        self._progress_last_percent: int = -1
        self._progress_finalized: bool = True
        self._warned_missing_save_dir: set[str] = set()
        self._has_shutdown_ollama: bool = False
        self._length_calculation_mode: bool = False
        self._pregeneration_complete: bool = False

    def clear_item_cache(self):
        super().clear_item_cache()

    def length(self) -> int:
        return self._get_previous_length(self.prompt_name)

    def get_inputs(self) -> List[str]:
        return [
            self.image_name,
            self.concept_name,
            self.prompt_name,
            self.image_path_name,
            'detail_crop_type',
            'detail_crop_scale',
            'detail_crop_index',
            'detail_crop_coords',
            'detail_crop_source_resolution',
            'detail_crop_prompt_context',
            *self.additional_passthrough,
        ]

    def get_outputs(self) -> List[str]:
        # We pass through all inputs and add the caption output
        caption_out_name = f"{self.prompt_name}_caption" if hasattr(self, 'prompt_name') else "prompt_caption"
        return list(self.get_inputs()) + [caption_out_name]

    def start(self, variation: int):
        self._flush_metrics()
        self._current_variation = variation
        self._metrics = self._new_metrics()
        self._reset_progress(f"[Detail Captions] Epoch {variation}", self.length())
        self._has_shutdown_ollama = False
        super().clear_item_cache()

        # Determine distributed context
        try:
            from modules.util import multi_gpu_util as _multi  # type: ignore
            world_size = int(getattr(_multi, "world_size")())
            rank = int(getattr(_multi, "rank")())
        except Exception:
            world_size, rank = 1, 0
        is_master = (rank == 0)

        # Pre-generate all captions upfront for better UX and progress tracking
        try:
            total = self.length()
            if total > 0:
                concept0 = self._get_previous_item(variation, self.concept_name, 0)
                detail_cfg = DetailCropGenerator._extract_detail_config(concept0)
                if detail_cfg.get('enable_captioning', False) and detail_cfg.get('caption_probability', 0.0) > 0.0:
                    # Distributed Captioning Implementation
                    # Each rank starts its own Ollama server and processes its own share of the crops.
                    # This allows fully utilizing all GPUs for caption generation.
                    
                    # Determine my share of the workload
                    my_shard_indices = [i for i in range(total) if i % world_size == rank]
                    
                    if not my_shard_indices:
                        # No work for this rank, mark complete and return
                        self._pregeneration_complete = True
                        return

                    prob_pct = int(detail_cfg.get('caption_probability', 0.0) * 100)
                    print(f"[Detail Captions] Rank {rank}/{world_size} generating captions for epoch {variation} (~{prob_pct}% of {len(my_shard_indices)} crops)...")
                    sys.stdout.flush()

                    # Start per-rank Ollama instance
                    # Use a unique port per rank to avoid collisions (default 11434, 11435, etc)
                    my_port = 11434 + rank
                    # Use only THIS rank's GPU index if we are in a distributed setting
                    # Assuming 'rank' maps 1:1 to visible device index in standard DDP
                    my_device_index = str(rank) 
                    
                    # If we are running locally single-gpu, rank is 0, device is whatever was passed or default.
                    # But if world_size > 1, we enforce strict binding.
                    
                    # Note: We must ensure we don't conflict with the global 'ollama_manager' state 
                    # if multiple threads were sharing it, but here we are in separate processes (DDP).
                    # Each process has its own 'ollama_manager' module instance.
                    
                    # REMOVED suppress(Exception) to debug why Rank 1 fails to start
                    # with suppress(Exception):
                    ollama_manager.prepare(
                        train_device="cuda",
                        device_indexes=my_device_index,
                        multi_gpu=False, # Treat as single-GPU instance for this specific server
                        port=my_port
                    )

                    # Ensure model exists locally on this rank's server
                    my_endpoint = f"http://localhost:{my_port}"
                    # Override the config endpoint for this session to force using our private server
                    # We cheat a bit by mutating the dict for this method scope, or passing it explicitly
                    
                    # We need to update detail_cfg to point to our private port for _request_caption calls
                    detail_cfg['caption_endpoint'] = my_endpoint

                    with suppress(Exception):
                        self._ensure_model_available(
                            detail_cfg.get('caption_model') or "qwen2.5vl:3b",
                            my_endpoint,
                            float(detail_cfg.get('caption_timeout', 120.0) or 120.0),
                        )

                    # Pre-generate captions for MY shard items
                    caption_out_name = f"{self.prompt_name}_caption" if hasattr(self, 'prompt_name') else "prompt_caption"
                    
                    # Helper function for generating a single caption
                    def _generate_one(idx):
                        try:
                            # get_item calls _request_caption, which uses detail_cfg['caption_endpoint']
                            self.get_item(variation, idx, caption_out_name)
                        except Exception as e:
                            print(f"[Detail Captions] Rank {rank} error pre-generating caption for item {idx}: {e}")
                            sys.stdout.flush()

                    # Even within a rank, we can use threads if we want concurrency against our private server
                    if self.parallel_workers > 1:
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        print(f"[Detail Captions] Rank {rank} using {self.parallel_workers} parallel workers")
                        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                            futures = [executor.submit(_generate_one, i) for i in my_shard_indices]
                            for future in as_completed(futures):
                                pass
                    else:
                        for index in my_shard_indices:
                            _generate_one(index)

                    print(f"[Detail Captions] Rank {rank} pre-generation complete. Metrics: {self._metrics}")
                    sys.stdout.flush()
                    self._pregeneration_complete = True
        except Exception as e:
            print(f"[Detail Captions] Error during pre-generation: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            self._pregeneration_complete = True  # Set flag even on error to prevent infinite retries

    def get_item(self, variation: int, index: int, requested_name: str = None) -> Dict[str, Any]:
        # Compute caption output name (may be called before __init__ completes)
        caption_out_name = f"{self.prompt_name}_caption" if hasattr(self, 'prompt_name') else "prompt_caption"

        # Handle passthrough requests (data we don't modify)
        if requested_name != caption_out_name and requested_name in self.get_inputs():
            # Just return the upstream data for passthrough fields
            return {requested_name: self._get_previous_item(variation, requested_name, index)}

        # For caption requests, do the full processing
        required_names = [
            self.image_name,
            self.concept_name,
            self.prompt_name,
            self.image_path_name,
            'detail_crop_type',
            'detail_crop_scale',
            'detail_crop_index',
            'detail_crop_coords',
            'detail_crop_source_resolution',
            'detail_crop_prompt_context',
        ]
        optional_names = [
            name for name in self.additional_passthrough
            if name not in required_names
        ]

        data: Dict[str, Any] = {}
        for name in required_names:
            data[name] = self._get_previous_item(variation, name, index)

        for name in optional_names:
            try:
                data[name] = self._get_previous_item(variation, name, index)
            except Exception:
                data[name] = None

        concept = data[self.concept_name]
        detail_cfg = DetailCropGenerator._extract_detail_config(concept)

        if (not detail_cfg.get('enable_captioning', False)
                or detail_cfg.get('caption_probability', 0.0) <= 0.0):
            if data.get('detail_crop_type') != 'full':
                self._record_metric('disabled')
            # No caption produced; preserve original prompt and expose empty caption field
            data[caption_out_name] = ""
            return self._finalize_output(data, detail_cfg)

        crop_type = data.get('detail_crop_type')
        if crop_type == 'full':
            # Do not caption full-image crops by design
            data[caption_out_name] = ""
            return self._finalize_output(data, detail_cfg)

        image_tensor: Optional[torch.Tensor] = data.get(self.image_name)
        if image_tensor is None:
            data[caption_out_name] = ""
            return self._finalize_output(data, detail_cfg)

        # During length calculation, skip actual captioning to avoid blocking
        if self._length_calculation_mode:
            data[caption_out_name] = ""
            return self._finalize_output(data, detail_cfg)

        caption_probability = max(0.0, min(1.0, detail_cfg.get('caption_probability', 0.0)))
        selection_value = self._deterministic_selection_value(data, detail_cfg)
        if selection_value >= caption_probability:
            self._record_metric('skipped_probability')
            # Set both prompt and caption to empty string for uncaptioned crops
            data[self.prompt_name] = ""
            data[caption_out_name] = ""
            return self._finalize_output(data, detail_cfg)

        cache_key = self._build_cache_key(data, detail_cfg)

        regenerate_each_epoch = detail_cfg.get('regenerate_each_epoch', False)
        caption = self._memory_cache.get(cache_key)
        
        # If pre-generation is complete, only use cached/disk captions, don't generate new ones
        if self._pregeneration_complete and caption is None:
            caption = self._load_caption_file(data, detail_cfg)
            if caption is not None:
                caption = self._sanitize_caption(caption)
                self._memory_cache[cache_key] = caption
                self._record_metric('reused')
            else:
                # Pre-generation already ran, don't generate during training
                self._record_metric('skipped_probability')
                data[self.prompt_name] = ""
                data[caption_out_name] = ""
                return self._finalize_output(data, detail_cfg)
            data[self.prompt_name] = caption
            data[caption_out_name] = caption
            return self._finalize_output(data, detail_cfg)

        if not regenerate_each_epoch and variation > 0:
            if caption is None:
                caption = self._load_caption_file(data, detail_cfg)
                if caption is not None:
                    caption = self._sanitize_caption(caption)
                    self._memory_cache[cache_key] = caption
                    self._record_metric('reused')
                else:
                    caption = self._request_caption(image_tensor, data, detail_cfg)
                    self._memory_cache[cache_key] = caption
                    self._write_caption_file(variation, data, detail_cfg, caption)
                    self._record_metric('captioned')
            else:
                caption = self._sanitize_caption(caption)
                self._memory_cache[cache_key] = caption
                self._record_metric('reused')
            data[self.prompt_name] = caption
            data[caption_out_name] = caption
            return self._finalize_output(data, detail_cfg)

        if caption is None:
            caption = self._load_caption_file(data, detail_cfg)
            if caption is not None:
                caption = self._sanitize_caption(caption)
                self._memory_cache[cache_key] = caption
                self._record_metric('reused')
        if caption is None:
            caption = self._request_caption(image_tensor, data, detail_cfg)
            self._memory_cache[cache_key] = caption
            self._write_caption_file(variation, data, detail_cfg, caption)
            self._record_metric('captioned')
        else:
            self._record_metric('reused')

        data[self.prompt_name] = caption
        data[caption_out_name] = caption
        return self._finalize_output(data, detail_cfg)

    @staticmethod
    def _deterministic_selection_value(data: Dict[str, Any], detail_cfg: dict) -> float:
        key_components = [
            str(data.get('detail_crop_type')),
            str(data.get('detail_crop_scale')),
            str(data.get('detail_crop_index')),
            str(data.get('detail_crop_coords')),
            str(data.get('detail_crop_source_resolution')),
            str(data.get('detail_crop_prompt_context')),
            str(detail_cfg.get('caption_model')),
            str(detail_cfg.get('caption_system_prompt')),
            str(detail_cfg.get('caption_user_prompt')),
        ]
        digest = hashlib.sha256("|".join(key_components).encode('utf-8')).digest()
        return int.from_bytes(digest[:8], byteorder='big') / float(1 << 64)

    def _build_cache_key(self, data: Dict[str, Any], detail_cfg: dict) -> str:
        key_data = {
            "prompt_context": data.get('detail_crop_prompt_context', '') or '',
            "image_path": data.get(self.image_path_name),
            "type": data.get('detail_crop_type'),
            "scale": data.get('detail_crop_scale'),
            "index": data.get('detail_crop_index'),
            "coords": data.get('detail_crop_coords'),
            "source_resolution": data.get('detail_crop_source_resolution'),
            "model": detail_cfg.get('caption_model'),
            "system": detail_cfg.get('caption_system_prompt'),
            "user": detail_cfg.get('caption_user_prompt'),
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode('utf-8')).hexdigest()

    def _write_caption_file(
            self,
            variation: int,
            data: Dict[str, Any],
            detail_cfg: dict,
            caption: str,
    ) -> None:
        caption_path = self._caption_file_path(variation, data, detail_cfg, create_dirs=True)
        if not caption_path:
            # Saving to disk is disabled or no valid directory was resolved; rely on in-memory cache only.
            return

        try:
            print(f"[Detail Captions Debug] Writing caption to: {caption_path}")
            with open(caption_path, "w", encoding="utf-8") as fh:
                fh.write(caption)
        except Exception as exc:
            print(f"[Detail Captions] Failed to write caption file at {caption_path}: {exc}")

    def _load_caption_file(
            self,
            data: Dict[str, Any],
            detail_cfg: dict,
    ) -> Optional[str]:
        caption_path = self._caption_file_path(0, data, detail_cfg, create_dirs=False)
        if not caption_path or not os.path.isfile(caption_path):
            return None

        try:
            with open(caption_path, "r", encoding="utf-8") as fh:
                contents = fh.read().strip()
                if not contents:
                    print(f"[Detail Captions] Caption file empty at {caption_path}")
                    return None
                return contents
        except Exception as exc:
            print(f"[Detail Captions] Failed to read caption file at {caption_path}: {exc}")
            return None

    @staticmethod
    def _encode_image(tensor: torch.Tensor) -> str:
        tensor_cpu = tensor.detach().to(device='cpu', dtype=torch.float32)
        tensor_cpu = torch.clamp(tensor_cpu, 0.0, 1.0)
        array = tensor_cpu.mul(255).to(dtype=torch.uint8).permute(1, 2, 0).contiguous().numpy()
        image = Image.fromarray(array)

        max_side = 768
        if max(image.size) > max_side:
            resample_attr = getattr(Image, "Resampling", Image)
            image.thumbnail((max_side, max_side), resample_attr.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _request_caption(self, tensor: torch.Tensor, data: Dict[str, Any], detail_cfg: dict) -> str:
        endpoint = detail_cfg.get('caption_endpoint') or "http://localhost:11434"
        timeout = float(detail_cfg.get('caption_timeout', 120.0) or 120.0)
        max_retries = max(0, int(detail_cfg.get('caption_max_retries', 4) or 4))
        backoff = 5.0

        if self._should_use_ollama_python_client(endpoint):
            return self._request_caption_via_python_module(
                tensor,
                data,
                detail_cfg,
                timeout=timeout,
                max_retries=max_retries,
                endpoint=endpoint,
            )

        model = detail_cfg.get('caption_model') or "qwen2.5vl:3b"

        system_prompt = detail_cfg.get('caption_system_prompt') or ""
        user_prompt_template = detail_cfg.get('caption_user_prompt') or ""

        prompt_context = data.get('detail_crop_prompt_context') or ""
        formatted_prompt = self._format_user_prompt(
            user_prompt_template,
            prompt_context,
            data,
            data.get(self.image_path_name),
        )

        payload = {
            "model": model,
            "system": system_prompt,
            "prompt": formatted_prompt,
            "images": [self._encode_image(tensor)],
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        attempt = 0
        last_error: Optional[Exception] = None
        while attempt <= max_retries:
            try:
                response = self._session.post(
                    f"{endpoint.rstrip('/')}/api/generate",
                    json=payload,
                    timeout=timeout,
                )
                if response.status_code != 200:
                    if response.status_code == 404 and "not found" in response.text.lower():
                        if detail_cfg.get('caption_auto_pull', True):
                            self._ensure_model_available(model, endpoint, timeout)
                            attempt += 1
                            continue
                        raise RuntimeError(
                            f"Ollama model '{model}' not found at {endpoint}. "
                            f"Install it with 'ollama pull {model}' or update the caption model setting."
                        )
                    raise RuntimeError(f"Ollama responded with status {response.status_code}: {response.text}")

                data = response.json()
                raw_caption = str(data.get("response", "")).strip()

                if not raw_caption:
                    raise RuntimeError("Received empty caption from Ollama.")
                return self._sanitize_caption(raw_caption)
            except Exception as exc:
                if isinstance(exc, requests.exceptions.Timeout):
                    with suppress(Exception):
                        ollama_manager.ensure_running(force_restart=True)
                elif self._should_attempt_restart(exc):
                    with suppress(Exception):
                        ollama_manager.ensure_running()
                last_error = exc
                attempt += 1
                timeout = min(timeout + backoff, timeout * 2)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

        raise RuntimeError(f"Failed to generate caption after {max_retries + 1} attempts: {last_error}")

    def _ensure_model_available(self, model: str, endpoint: str, timeout: float) -> None:
        if model in self._models_pulled:
            return

        # Only support local endpoints for auto pull
        if not endpoint.startswith("http://localhost") and not endpoint.startswith("http://127.0.0.1"):
            raise RuntimeError(
                f"Ollama model '{model}' not found at {endpoint} and auto-pull only works for local servers. "
                f"Install the model manually or adjust the caption model setting."
            )

        pull_command = ["ollama", "pull", model]
        try:
            subprocess.run(
                pull_command,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(timeout * 2, 60.0),
            )
            self._models_pulled.add(model)
            return
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not find the 'ollama' executable on PATH. Install Ollama desktop/CLI or adjust your PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out while pulling Ollama model '{model}'. Try pulling it manually with 'ollama pull {model}'."
            ) from exc
        except subprocess.CalledProcessError as exc:
            pass  # fall through to fallback

        # fallback: query the manifest and pick the smallest matching model
        try:
            list_proc = subprocess.run(
                ["ollama", "list"],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            available_names = self._parse_ollama_list(list_proc.stdout)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to auto-pull Ollama model '{model}'. Command output:\n{exc.stdout if isinstance(exc, subprocess.CalledProcessError) else ''}\n{exc.stderr if isinstance(exc, subprocess.CalledProcessError) else ''}"
            ) from exc

        manifest_model = self._select_manifest_model(model, available_names)
        if manifest_model is None:
            raise RuntimeError(
                f"Failed to auto-pull Ollama model '{model}'. Ensure the name is correct or install it manually."
            )

        try:
            subprocess.run(
                ["ollama", "pull", manifest_model],
                check=True,
                capture_output=True,
                text=True,
                timeout=max(timeout * 2, 60.0),
            )
            self._models_pulled.add(manifest_model)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to auto-pull Ollama model '{manifest_model}'. "
                f"Command output:\n{exc.stdout}\n{exc.stderr}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out while pulling Ollama model '{manifest_model}'. Try pulling it manually with 'ollama pull {manifest_model}'."
            ) from exc

    @staticmethod
    def _parse_ollama_list(output: str) -> List[str]:
        models = []
        for line in output.splitlines():
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models

    @staticmethod
    def _select_manifest_model(requested: str, available: List[str]) -> Optional[str]:
        def _normalize(name: str) -> str:
            return ''.join(ch for ch in name.lower() if ch.isalnum())

        norm_requested = _normalize(requested)
        candidates: List[Tuple[str, str]] = []
        for name in available:
            norm_name = _normalize(name)
            if norm_name == norm_requested:
                return name
            if norm_name.startswith(norm_requested) or norm_requested.startswith(norm_name):
                candidates.append((norm_name, name))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (len(item[0]), item[1]))
        return candidates[0][1]

    @staticmethod
    def _format_user_prompt(template: str, context: str, data: Dict[str, Any], image_path: Optional[str]) -> str:
        formatted_context = context if context else "No additional context."
        safe_context = formatted_context.replace("\r", " ").replace("\n", " ").strip()

        substitutions = _DefaultDict({
            "context": safe_context,
            "crop_type": data.get('detail_crop_type'),
            "scale": data.get('detail_crop_scale'),
            "index": data.get('detail_crop_index'),
            "coords": data.get('detail_crop_coords'),
            "source_resolution": data.get('detail_crop_source_resolution'),
            "image_path": image_path,
        })
        return template.format_map(substitutions)

    @staticmethod
    def _should_attempt_restart(error: Exception) -> bool:
        if isinstance(error, requests.exceptions.ConnectionError):
            return True
        if isinstance(error, requests.exceptions.Timeout):
            return True
        if isinstance(error, requests.exceptions.RequestException):
            return isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))
        message = str(error).lower()
        if "failed to establish a new connection" in message:
            return True
        if "connection refused" in message:
            return True
        return False

    def _caption_file_path(
            self,
            variation: int,
            data: Dict[str, Any],
            detail_cfg: dict,
            *,
            create_dirs: bool,
    ) -> Optional[str]:
        # Use disk when:
        # - user explicitly enabled save_to_disk, or
        # - multi-GPU training is active (to share captions across ranks).
        use_disk = bool(detail_cfg.get('save_to_disk'))
        try:
            from modules.util import multi_gpu_util as _multi  # type: ignore
            if int(getattr(_multi, "world_size")()) > 1:
                use_disk = True
        except Exception:
            pass

        if not use_disk:
            return None

        base_dir = detail_cfg.get('save_directory')
        if not base_dir:
            # Fallback to OneTrainer cache directory by default
            # TrainConfig default cache_dir is "workspace-cache/run"
            base_dir = os.path.join(os.getcwd(), "workspace-cache", "run")
        # Place captions under a stable sub-directory inside cache
        base_dir = os.path.join(base_dir, "detail_captions")

        if not detail_cfg.get('regenerate_each_epoch', False):
            variation = 0

        variant_type = str(data.get('detail_crop_type', 'detail') or 'detail')
        scale_value = data.get('detail_crop_scale')
        if scale_value in (None, 1, 1.0):
            scale_label = "1"
        else:
            try:
                scale_label = f"{float(scale_value):g}".replace('.', 'p')
            except (TypeError, ValueError):
                scale_label = str(scale_value)

        epoch_dir = os.path.join(base_dir, f"epoch-{variation}")
        variant_dir = os.path.join(epoch_dir, f"{variant_type}_scale-{scale_label}")
        if create_dirs:
            os.makedirs(variant_dir, exist_ok=True)

        original_path = data.get(self.image_path_name) or ""
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        try:
            tile_index = int(data.get('detail_crop_index', 0))
        except (TypeError, ValueError):
            tile_index = 0

        filename = f"{base_name}_detail_{variant_type}_scale-{scale_label}_tile-{tile_index:03d}.txt"
        return os.path.join(variant_dir, filename)

    def _record_metric(self, kind: str) -> None:
        self._metrics['total'] += 1
        if kind in self._metrics:
            self._metrics[kind] += 1

    def _new_metrics(self) -> Dict[str, int]:
        return {
            'total': 0,
            'captioned': 0,
            'reused': 0,
            'skipped_probability': 0,
            'disabled': 0,
            'missing': 0,
        }

    def _flush_metrics(self, final: bool = False) -> None:
        if self._progress_total > 0 and not self._progress_finalized:
            self._print_progress(final=True)
            self._progress_finalized = True
        if self._current_variation is None:
            return
        if self._metrics['total'] == 0:
            return

        prefix = "[Detail Captions]"
        variation_label = f"variation {self._current_variation}" if not final else "final variation"
        print(
            f"{prefix} {variation_label}: total={self._metrics['total']} "
            f"captioned={self._metrics['captioned']} reused={self._metrics['reused']} "
            f"skipped_probability={self._metrics['skipped_probability']} "
            f"disabled={self._metrics['disabled']} missing={self._metrics['missing']}"
        )

    def __del__(self):
        with suppress(Exception):
            self._flush_metrics(final=True)

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------
    def _reset_progress(self, prefix: str, total: int):
        self._progress_prefix = prefix
        self._progress_total = max(0, total)
        self._progress_count = 0
        self._progress_last_percent = -1
        self._progress_finalized = self._progress_total == 0
        if self._progress_total > 0:
            self._print_progress()

    def _advance_progress(self, detail_cfg: Optional[dict]):
        if self._progress_total <= 0 or self._progress_finalized:
            return
        self._progress_count = min(self._progress_count + 1, self._progress_total)
        final = self._progress_count >= self._progress_total
        self._print_progress(final=final)
        if final:
            self._progress_finalized = True
            self._maybe_shutdown_ollama(detail_cfg)

    def _print_progress(self, final: bool = False):
        if self._progress_total <= 0:
            return
        percent = int((self._progress_count * 100) / self._progress_total) if self._progress_total else 100
        if percent != self._progress_last_percent or final:
            sys.stdout.write(
                f"\r{self._progress_prefix}: {self._progress_count}/{self._progress_total} ({percent}%)"
            )
            sys.stdout.flush()
            self._progress_last_percent = percent
        if final:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _finalize_output(self, data: Dict[str, Any], detail_cfg: dict) -> Dict[str, Any]:
        self._advance_progress(detail_cfg)
        return data

    # ------------------------------------------------------------------
    # Ollama python client helpers
    # ------------------------------------------------------------------
    def _should_use_ollama_python_client(self, endpoint: str) -> bool:
        if _OLLAMA_MODULE is None:
            return False
        endpoint_normalized = (endpoint or "").strip().lower()
        if not endpoint_normalized:
            return True
        return endpoint_normalized in {"http://localhost:11434", "http://127.0.0.1:11434"}

    def _request_caption_via_python_module(
            self,
            tensor: torch.Tensor,
            data: Dict[str, Any],
            detail_cfg: dict,
            *,
            timeout: float,
            max_retries: int,
            endpoint: str,
    ) -> str:
        if _OLLAMA_MODULE is None:
            raise RuntimeError("Ollama Python module is not available.")

        if endpoint:
            os.environ["OLLAMA_HOST"] = endpoint.rstrip("/")

        model = detail_cfg.get('caption_model') or "qwen2.5vl:3b"
        system_prompt = detail_cfg.get('caption_system_prompt') or ""
        formatted_prompt = self._format_user_prompt(
            detail_cfg.get('caption_user_prompt') or "",
            data.get('detail_crop_prompt_context') or "",
            data,
            data.get(self.image_path_name),
        )
        image_b64 = self._encode_image(tensor)

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message: Dict[str, Any] = {"role": "user", "content": formatted_prompt}
        if image_b64:
            user_message["images"] = [image_b64]
        messages.append(user_message)

        is_thinking_model = "qwen3" in model.lower()
        max_tokens = detail_cfg.get('caption_max_tokens', _DEFAULT_CAPTION_MAX_TOKENS) or _DEFAULT_CAPTION_MAX_TOKENS
        if is_thinking_model:
            max_tokens = max(max_tokens, _DEFAULT_CAPTION_MAX_TOKENS * 2)

        options: Dict[str, Any] = {
            "temperature": 0.0,
            "num_ctx": 4096,
            "num_predict": int(max_tokens),
        }

        attempt = 0
        last_error: Optional[Exception] = None
        backoff = 2.0
        while attempt <= max_retries:
            try:
                ollama_manager.ensure_running()
                response = self._call_ollama_chat_with_timeout(model, messages, options, timeout)
                raw_caption = self._parse_ollama_response(response, is_thinking_model)
                if not raw_caption:
                    raise RuntimeError("Received empty caption from Ollama.")
                return self._sanitize_caption(raw_caption)
            except TimeoutError as exc:
                last_error = exc
                with suppress(Exception):
                    ollama_manager.ensure_running(force_restart=True)
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "model" in message and "not found" in message:
                    raise
                with suppress(Exception):
                    ollama_manager.ensure_running(force_restart=True)
            attempt += 1
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

        raise RuntimeError(f"Failed to generate caption after {max_retries + 1} attempts: {last_error}")

    def _call_ollama_chat_with_timeout(
            self,
            model: str,
            messages: List[Dict[str, Any]],
            options: Dict[str, Any],
            timeout: float,
    ):
        if _OLLAMA_MODULE is None:
            raise RuntimeError("Ollama Python module is not available.")

        def _invoke():
            return _OLLAMA_MODULE.chat(
                model=model,
                messages=messages,
                options=options,
                stream=False,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeout as exc:
                future.cancel()
                raise TimeoutError(f"Ollama chat timed out after {timeout} seconds") from exc

    def _parse_ollama_response(self, response: Any, is_thinking_model: bool) -> str:
        content: Any = ""
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            message = getattr(response, "message", None)
            if message is not None:
                content = getattr(message, "content", "")

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text") or item.get("content") or "")
                elif isinstance(item, str):
                    parts.append(item)
            content = "".join(parts)

        if not isinstance(content, str):
            content = str(content or "")

        content = content.strip()
        if is_thinking_model and content:
            content = self._strip_thinking_prefix(content)
        return content.strip()

    def _strip_thinking_prefix(self, content: str) -> str:
        import re

        thinking_patterns = [
            r"^(got it[,\s]+let['s]*\s+(tackle|analyze|check|see|examine))",
            r"^(let['s]*\s+(analyze|check|tackle|see|examine))",
            r"^(first[,\s]+)",
            r"^(now[,\s]+)",
            r"^(ok[,\s]+)",
            r"(\bgot it[,\s]+)",
            r"(let['s]*\s+(analyze|check|tackle|see|examine)[\s\w]+\.)",
            r"(wait[,\s]+)",
            r"(let['s]*\s+see)",
        ]

        cleaned = content
        for pattern in thinking_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

        cleaned = cleaned.strip()
        return cleaned or content

    def _sanitize_caption(self, caption: str) -> str:
        if not caption:
            return ""
        cleaned = caption.strip()
        if "unanswerable" in cleaned.lower():
            return ""
        return cleaned

    def _maybe_shutdown_ollama(self, detail_cfg: Optional[dict]) -> None:
        if self._has_shutdown_ollama:
            return
        if not detail_cfg:
            return
        if detail_cfg.get('regenerate_each_epoch', False):
            return
        if not detail_cfg.get('enable_captioning', False):
            return
        with suppress(Exception):
            ollama_manager.cleanup()
        self._has_shutdown_ollama = True


class _DefaultDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"

