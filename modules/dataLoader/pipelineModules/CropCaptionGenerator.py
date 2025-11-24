import base64
import datetime
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import torch
from PIL import Image

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from modules.dataLoader.pipelineModules.DetailCropGenerator import DetailCropGenerator
from modules.util import ollama_manager

try:
    import ollama as _OLLAMA_MODULE  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _OLLAMA_MODULE = None


_DEFAULT_CAPTION_MAX_TOKENS = 256


@dataclass(frozen=True)
class CaptionGPUConfig:
    enabled: bool
    device_indices: List[int]
    base_port: int = 12134


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
            caption_gpu_config: Optional[CaptionGPUConfig] = None,
    ):
        self.image_name = image_name
        self.concept_name = concept_name
        self.prompt_name = prompt_name
        self.image_path_name = image_path_name
        self.additional_passthrough = tuple(sorted(set(additional_passthrough or [])))
        self.parallel_workers = parallel_workers
        self._caption_gpu_config = caption_gpu_config
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

    @staticmethod
    def _run_cli_command(
            args: List[str],
            *,
            env: Optional[dict] = None,
            timeout: Optional[float] = None,
            check: bool = False,
    ) -> subprocess.CompletedProcess:
        """
        Execute a helper CLI command (ollama/sc/taskkill) with UTF-8 decoding so Windows consoles don't choke.
        """
        return subprocess.run(
            args,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
            check=check,
        )

    def clear_item_cache(self):
        super().clear_item_cache()

    def length(self) -> int:
        """
        Approximate the number of detail crops that need captions.
        Prefer detail crop specific signals if they exist, because the raw prompt length
        only reflects the base dataset (e.g., 7 images) instead of the expanded crop set.
        """
        detail_length_fields = (
            "detail_crop_index",
            "detail_crop_type",
            "detail_crop_coords",
        )

        for field in detail_length_fields:
            with suppress(Exception):
                detail_length = self._get_previous_length(field)
                if detail_length > 0:
                    return detail_length

        # Fall back to prompt length so non-detail mode still works
        with suppress(Exception):
            return self._get_previous_length(self.prompt_name)

        return 0

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
        
        # Wait for DetailCropGenerator to finish generating crops before querying length
        # In multi-GPU, ensure all ranks have crops populated
        rank, world_size, distributed_ready = self._distributed_status()
        if distributed_ready:
            import torch
            torch.distributed.barrier()
        
        total = self.length()
        print(f"[Detail Captions] start() called for epoch {variation}, total crops: {total}")
        sys.stdout.flush()
        self._reset_progress(f"[Detail Captions] Epoch {variation}", total)
        self._has_shutdown_ollama = False
        super().clear_item_cache()
        self._pregeneration_complete = False

        if total <= 0:
            print(f"[Detail Captions] Returning early: total={total} (no crops to caption)")
            sys.stdout.flush()
            self._pregeneration_complete = True
            return

        try:
            concept0 = self._get_previous_item(variation, self.concept_name, 0)
            detail_cfg = DetailCropGenerator._extract_detail_config(concept0)
        except Exception:
            detail_cfg = {}

        if (not detail_cfg.get('enable_captioning', False)
                or not detail_cfg.get('enabled', False)  # Skip if detail crops are disabled entirely
                or detail_cfg.get('caption_probability', 0.0) <= 0.0):
            self._pregeneration_complete = True
            return

        # Early validation: external captioner requires save_to_disk=True
        if not detail_cfg.get('save_to_disk', False):
            error_msg = (
                "Detail crop captioning requires 'save_to_disk=True' in the detail crop configuration "
                "when using the external captioner. Please enable 'Save Crops to Disk' in your detail crop settings."
            )
            print(f"[Detail Captions] ERROR: {error_msg}")
            sys.stdout.flush()
            raise RuntimeError(error_msg)

        barrier_pending = distributed_ready
        try:
            manifest_info = None
            if rank == 0:
                manifest_info = self._write_caption_manifest(variation, total)
                job_count = manifest_info.get("job_count", 0)
                if job_count > 0:
                    print(f"[Detail Captions] Launching external caption job for {job_count} crops (manifest: {manifest_info['manifest_path']})")
                    sys.stdout.flush()
                    self._run_external_caption_job(manifest_info)
                    # Ensure caption files are written and flushed before barrier
                    print(f"[Detail Captions] External caption job completed, ensuring file synchronization...")
                    sys.stdout.flush()
                    # Small delay to ensure file system sync (especially on Windows)
                    time.sleep(0.5)
                else:
                    print(f"[Detail Captions] No crops selected for captioning at epoch {variation}; skipping external job.")
                    sys.stdout.flush()
            if distributed_ready:
                self._distributed_barrier(rank, label="CAPTION_JOB_DONE")
                barrier_pending = False
                # Small delay to ensure file system is ready on all ranks
                time.sleep(0.2)
            self._pregeneration_complete = True
        except Exception as exc:
            if barrier_pending:
                try:
                    self._distributed_barrier(rank, label="CAPTION_JOB_DONE")
                except Exception:
                    pass
            print(f"[Detail Captions] FATAL ERROR during external caption job: {exc}")
            traceback.print_exc()
            sys.stdout.flush()
            raise

    def _write_caption_manifest(self, variation: int, total: int) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        for idx in range(total):
            try:
                entry = self._build_manifest_entry(variation, idx)
            except Exception as exc:
                raise RuntimeError(f"Failed to build caption manifest entry at index {idx}: {exc}") from exc
            if not entry:
                continue
            entries.append(entry)

        if not entries:
            return {"manifest_path": "", "log_path": "", "job_count": 0}

        cache_root = os.path.join(os.getcwd(), "workspace-cache", "detail_crops_cache")
        manifest_dir = os.path.join(cache_root, "detail_captions", "manifests")
        os.makedirs(manifest_dir, exist_ok=True)
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        manifest_name = f"caption-manifest-epoch{variation}-{timestamp}.json"
        manifest_path = os.path.join(manifest_dir, manifest_name)
        log_path = os.path.join(manifest_dir, f"{Path(manifest_name).stem}.log")

        # Remove helper-only fields before writing
        for entry in entries:
            entry.pop("save_directory", None)

        manifest = {
            "version": 1,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "epoch": variation,
            "job_count": len(entries),
            "config": {
                "gpu_indices": self._resolve_caption_gpu_indices(),
                "base_port": self._caption_gpu_config.base_port if (self._caption_gpu_config and self._caption_gpu_config.base_port) else 12134,
                "show_console": self._debug_show_console(),
            },
            "jobs": entries,
        }

        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        return {"manifest_path": manifest_path, "log_path": log_path, "job_count": len(entries)}

    def _build_manifest_entry(self, variation: int, dataset_index: int) -> Optional[Dict[str, Any]]:
        concept = self._get_previous_item(variation, self.concept_name, dataset_index)
        detail_cfg = DetailCropGenerator._extract_detail_config(concept)
        if not detail_cfg.get('enable_captioning', False):
            return None

        caption_probability = max(0.0, min(1.0, detail_cfg.get('caption_probability', 0.0)))
        if caption_probability <= 0.0:
            return None

        if not detail_cfg.get('save_to_disk', False):
            raise RuntimeError(
                f"Detail crop captioning requires 'save_to_disk=True' in the detail crop configuration "
                f"when using the external captioner (concept at dataset index {dataset_index}). "
                f"Please enable 'Save Crops to Disk' in your detail crop settings."
            )
        save_dir = os.path.join(os.getcwd(), "workspace-cache", "detail_crops_cache")
        save_dir = os.path.abspath(save_dir)

        data: Dict[str, Any] = {
            self.concept_name: concept,
            self.image_path_name: self._get_previous_item(variation, self.image_path_name, dataset_index),
            'detail_crop_type': self._get_previous_item(variation, 'detail_crop_type', dataset_index),
            'detail_crop_scale': self._get_previous_item(variation, 'detail_crop_scale', dataset_index),
            'detail_crop_index': self._get_previous_item(variation, 'detail_crop_index', dataset_index),
            'detail_crop_coords': self._get_previous_item(variation, 'detail_crop_coords', dataset_index),
            'detail_crop_source_resolution': self._get_previous_item(variation, 'detail_crop_source_resolution', dataset_index),
            'detail_crop_prompt_context': self._get_previous_item(variation, 'detail_crop_prompt_context', dataset_index),
        }

        crop_type = data['detail_crop_type']
        if crop_type == 'full':
            return None

        prompt_context = data.get('detail_crop_prompt_context') or ""
        scale_value = float(data.get('detail_crop_scale') or 1.0)
        tile_index = int(data.get('detail_crop_index') or 0)
        image_identifier = data[self.image_path_name]

        selection_value = self._deterministic_selection_value(data, detail_cfg)
        if selection_value >= caption_probability:
            return None

        export_path = self._resolve_exported_crop_path(
            detail_cfg,
            variation,
            image_identifier,
            crop_type,
            scale_value,
            tile_index,
        )

        if not os.path.isfile(export_path):
            raise RuntimeError(f"Expected detail crop image missing at {export_path}")

        caption_path = self._caption_file_path(variation, data, detail_cfg, create_dirs=True)
        if not caption_path:
            raise RuntimeError("Failed to resolve caption output path; ensure detail crop save_directory is configured.")

        formatted_prompt = self._format_user_prompt(
            detail_cfg.get('caption_user_prompt') or "",
            prompt_context,
            data,
            self._strip_detail_identifier(image_identifier),
        )

        entry = {
            "dataset_index": dataset_index,
            "image_path": export_path,
            "caption_path": caption_path,
            "prompt": formatted_prompt,
            "prompt_context": prompt_context,
            "crop_type": crop_type,
            "scale": scale_value,
            "tile_index": tile_index,
            "model": detail_cfg.get('caption_model') or "qwen2.5vl:3b",
            "system_prompt": detail_cfg.get('caption_system_prompt') or "",
            "timeout": float(detail_cfg.get('caption_timeout', 120.0) or 120.0),
            "max_retries": int(detail_cfg.get('caption_max_retries', 4) or 4),
            "max_tokens": int(detail_cfg.get('caption_max_tokens', _DEFAULT_CAPTION_MAX_TOKENS) or _DEFAULT_CAPTION_MAX_TOKENS),
            "selection_value": selection_value,
            "concept_label": DetailCropGenerator._concept_label(concept),
        }
        return entry

    @staticmethod
    def _strip_detail_identifier(identifier: str) -> str:
        if not identifier:
            return ""
        marker = "#detail"
        if marker in identifier:
            return identifier.split(marker, 1)[0]
        return identifier

    def _resolve_exported_crop_path(
            self,
            detail_cfg: dict,
            variation: int,
            image_identifier: str,
            crop_type: str,
            scale_value: float,
            tile_index: int,
    ) -> str:
        save_dir = detail_cfg.get('save_directory')
        if not save_dir:
            raise RuntimeError("Detail crop save_directory is required to locate exported crops.")
        save_dir = os.path.abspath(save_dir)
        epoch_value = variation if detail_cfg.get('regenerate_each_epoch', False) else 0
        try:
            numeric_scale = float(scale_value)
        except (TypeError, ValueError):
            numeric_scale = 1.0
        if numeric_scale == 1.0:
            scale_label = "1"
        else:
            scale_label = f"{numeric_scale:g}".replace('.', 'p')
        variant_dir = os.path.join(save_dir, "detail_crops", f"epoch-{epoch_value}", f"{crop_type}_scale-{scale_label}")
        base_name = os.path.splitext(os.path.basename(self._strip_detail_identifier(image_identifier)))[0]
        file_name = f"{base_name}_detail_{crop_type}_scale-{scale_label}_tile-{int(tile_index):03d}.png"
        return os.path.abspath(os.path.join(variant_dir, file_name))

    def _resolve_caption_gpu_indices(self) -> List[int]:
        if self._caption_gpu_config and self._caption_gpu_config.enabled and self._caption_gpu_config.device_indices:
            return [int(idx) for idx in self._caption_gpu_config.device_indices]
        return [0]

    def _run_external_caption_job(self, manifest_info: Dict[str, Any]) -> None:
        manifest_path = manifest_info.get("manifest_path")
        if not manifest_path:
            return
        
        script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_ollama_caption_job.py"
        if not script_path.exists():
            raise RuntimeError(f"Unable to locate external caption runner at {script_path}")
        log_path = manifest_info.get("log_path") or str(script_path.with_suffix(".log"))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cmd = [
            sys.executable,
            str(script_path),
            "--manifest",
            manifest_path,
        ]
        print(f"[Detail Captions] Running external caption job: {' '.join(cmd)}", flush=True)
        print(f"[Detail Captions] Log file: {log_path}", flush=True)
        print(f"[Detail Captions] Script path: {script_path}", flush=True)
        print(f"[Detail Captions] Script exists: {script_path.exists()}", flush=True)
        print(f"[Detail Captions] Working directory: {script_path.parents[1]}", flush=True)
        print(f"[Detail Captions] Python executable: {sys.executable}", flush=True)
        
        with open(log_path, "w", encoding="utf-8") as log_file:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(script_path.parents[1]),
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            # Check if process started successfully
            if process.poll() is not None:
                # Process already terminated
                return_code = process.returncode
                print(f"[Detail Captions] Process terminated immediately with exit code {return_code}", flush=True)
            else:
                assert process.stdout is not None
                output_lines = []
                for line in process.stdout:
                    output_lines.append(line)
                    log_file.write(line)
                    log_file.flush()
                    print(line.rstrip("\n"))
                    sys.stdout.flush()
                return_code = process.wait()
                # If we got no output and process failed, it might have crashed before writing
                if return_code != 0 and not output_lines:
                    print(f"[Detail Captions] WARNING: Process failed with exit code {return_code} but produced no output", flush=True)
        if return_code != 0:
            # Read the log file to include error details
            log_content = ""
            log_file_exists = False
            try:
                if os.path.exists(log_path):
                    log_file_exists = True
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        log_lines = f.readlines()
                        # Include last 50 lines of log for context
                        if log_lines:
                            log_content = "\n".join(log_lines[-50:])
                        else:
                            log_content = "(log file exists but is empty)"
            except Exception as e:
                log_content = f"(error reading log file: {e})"
            
            # Check if process was killed by signal (negative exit code on Unix)
            signal_info = ""
            if return_code < 0:
                signal_num = -return_code
                signal_info = f" (killed by signal {signal_num}"
                if signal_num == 15:
                    signal_info += " - SIGTERM, likely killed by system or OOM killer"
                elif signal_num == 9:
                    signal_info += " - SIGKILL, forcefully terminated"
                signal_info += ")"
            
            error_msg = (
                f"External caption job failed with exit code {return_code}{signal_info}.\n"
                f"Command: {' '.join(shlex.quote(arg) for arg in cmd)}\n"
                f"Working directory: {script_path.parents[1]}\n"
                f"Script path: {script_path}\n"
                f"Script exists: {script_path.exists()}\n"
                f"Python executable: {sys.executable}\n"
                f"Log file: {log_path}\n"
                f"Log file exists: {log_file_exists}\n"
            )
            if log_content:
                error_msg += f"\nLast 50 lines of log:\n{log_content}"
            else:
                error_msg += "\n(No log content available - process may have been killed before writing output)"
            raise RuntimeError(error_msg)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> Dict[str, Any]:
        # Compute caption output name (may be called before __init__ completes)
        caption_out_name = f"{self.prompt_name}_caption" if hasattr(self, 'prompt_name') else "prompt_caption"

        # Handle passthrough requests (data we don't modify)
        if requested_name != caption_out_name and requested_name in self.get_inputs():
            # Just return the upstream data for passthrough fields
            result = self._get_previous_item(variation, requested_name, index)
            return {requested_name: result}

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
        
        for name_idx, name in enumerate(required_names):
            try:
                data[name] = self._get_previous_item(variation, name, index)
            except Exception as exc:
                raise

        for name in optional_names:
            try:
                data[name] = self._get_previous_item(variation, name, index)
            except Exception:
                data[name] = None
        
        concept = data[self.concept_name]
        detail_cfg = DetailCropGenerator._extract_detail_config(concept)
        endpoint_override = None

        enable_captioning = detail_cfg.get('enable_captioning', False)
        caption_probability = detail_cfg.get('caption_probability', 0.0)

        if (not enable_captioning or caption_probability <= 0.0):
            crop_type_check = data.get('detail_crop_type')
            if crop_type_check != 'full':
                self._record_metric('disabled')
            # No caption produced; preserve original prompt and expose empty caption field
            data[caption_out_name] = ""
            # Note: Blank captions are returned on-the-fly during training, no need to write files
            return self._finalize_output(data, detail_cfg)

        crop_type = data.get('detail_crop_type')
        
        if crop_type == 'full':
            # Do not caption full-image crops by design
            data[caption_out_name] = ""
            # Note: Blank captions are returned on-the-fly during training, no need to write files
            return self._finalize_output(data, detail_cfg)

        image_tensor: Optional[torch.Tensor] = data.get(self.image_name)
        
        if image_tensor is None:
            data[caption_out_name] = ""
            # Note: Blank captions are returned on-the-fly during training, no need to write files
            return self._finalize_output(data, detail_cfg)

        # During length calculation, skip actual captioning to avoid blocking
        if self._length_calculation_mode:
            data[caption_out_name] = ""
            # Note: Blank captions are returned on-the-fly during training, no need to write files
            return self._finalize_output(data, detail_cfg)
        
        caption_probability = max(0.0, min(1.0, detail_cfg.get('caption_probability', 0.0)))
        selection_value = self._deterministic_selection_value(data, detail_cfg)
        
        if selection_value >= caption_probability:
            self._record_metric('skipped_probability')
            # Set both prompt and caption to empty string for uncaptioned crops
            data[self.prompt_name] = ""
            data[caption_out_name] = ""
            # Note: Blank captions are returned on-the-fly during training, no need to write files
            return self._finalize_output(data, detail_cfg)

        cache_key = self._build_cache_key(data, detail_cfg)
        regenerate_each_epoch = detail_cfg.get('regenerate_each_epoch', False)
        caption = self._memory_cache.get(cache_key)
        
        # If pre-generation is complete, only use cached/disk captions, don't generate new ones
        if self._pregeneration_complete and caption is None:
            # Load caption file with retry logic (handles race conditions)
            try:
                caption = self._load_caption_file(data, detail_cfg)
            except Exception as exc:
                # If loading fails, log but don't block - return empty caption
                caption = None
            
            if caption is not None and caption != "":
                caption = self._sanitize_caption(caption)
                self._memory_cache[cache_key] = caption
                self._record_metric('reused')
            else:
                # Pre-generation already ran, don't generate during training
                # If caption file is missing or empty, return empty caption (blank caption on the fly for skipped crops)
                caption = ""  # Blank caption on the fly for skipped crops
                self._memory_cache[cache_key] = caption  # Cache blank caption to avoid repeated file checks
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
                    caption = self._request_caption(image_tensor, data, detail_cfg, endpoint_override=endpoint_override)
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
            caption = self._request_caption(image_tensor, data, detail_cfg, endpoint_override=endpoint_override)
            self._memory_cache[cache_key] = caption
            self._write_caption_file(variation, data, detail_cfg, caption)
            self._record_metric('captioned')
        else:
            self._record_metric('reused')

        data[self.prompt_name] = caption
        data[caption_out_name] = caption
        return self._finalize_output(data, detail_cfg)


    def _distributed_barrier(self, rank: int, label: str) -> None:
        try:
            from modules.util import multi_gpu_util as _multi  # type: ignore
            world_size = int(getattr(_multi, "world_size")())
            if world_size <= 1:
                return
            if not torch.distributed.is_initialized():
                raise RuntimeError(f"Rank {rank}: torch.distributed not initialized for barrier '{label}'")
            print(f"[Detail Captions] Rank {rank} waiting at {label} barrier...")
            sys.stdout.flush()
            torch.distributed.barrier()
            print(f"[Detail Captions] Rank {rank} barrier {label} released")
            sys.stdout.flush()
        except Exception as exc:
            print(f"[Detail Captions] Rank {rank} {label} barrier exception: {exc}")
            traceback.print_exc()
            sys.stdout.flush()
            raise

    def _distributed_status(self) -> Tuple[int, int, bool]:
        rank = 0
        world_size = 1
        try:
            from modules.util import multi_gpu_util as _multi  # type: ignore
            world_size = int(getattr(_multi, "world_size")())
            rank = int(getattr(_multi, "rank")())
        except Exception:
            pass
        distributed_ready = False
        try:
            distributed_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        except Exception:
            distributed_ready = False
        return rank, world_size, distributed_ready and world_size > 1

    def _using_managed_servers(self) -> bool:
        # External caption jobs own the Ollama lifecycle, so the in-process helper never manages servers.
        return False

    def _log_caption_heartbeat(self, rank: int, world_size: int, processed: int, total: int) -> None:
        if total <= 0:
            return
        percent = int((processed / total) * 100)
        print(
            f"[Detail Captions] Rank {rank}/{world_size} progress: "
            f"{processed}/{total} ({percent}%)"
        )
        sys.stdout.flush()

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
            # Show which rank is writing for debugging parallel execution
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
        if not caption_path:
            return None
        
        # During training (pregeneration complete), missing files mean skipped crops - return blank caption immediately
        # Only retry if pregeneration is not complete (files might still be written)
        max_retries = 3 if not self._pregeneration_complete else 1
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                # Check if file exists
                file_exists = os.path.isfile(caption_path)
                
                if not file_exists:
                    # File doesn't exist - during training this means skipped crop, return blank caption immediately
                    return ""  # Return blank caption immediately for skipped crops (generated on the fly)

                # File exists, try to read it
                try:
                    with open(caption_path, "r", encoding="utf-8") as fh:
                        contents = fh.read().strip()
                        if not contents:
                            # Empty file means blank caption (skipped crop)
                            return ""  # Return empty string for blank captions
                        return contents
                except (IOError, OSError) as exc:
                    # File might be locked or still being written, retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        # After max retries, return blank caption instead of None
                        return ""  # Return blank caption instead of None
                        
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    print(f"[Detail Captions] Unexpected error loading caption file at {caption_path}: {exc}")
                    traceback.print_exc()
                    sys.stdout.flush()
                    return None
        
        return None

    @staticmethod
    def _encode_image(tensor: torch.Tensor) -> str:
        tensor_cpu = tensor.detach().to(device='cpu', dtype=torch.float32)
        tensor_cpu = torch.clamp(tensor_cpu, 0.0, 1.0)
        array = tensor_cpu.mul(255).to(dtype=torch.uint8).permute(1, 2, 0).contiguous().numpy()
        image = Image.fromarray(array)

        max_side = 512
        if max(image.size) > max_side:
            resample_attr = getattr(Image, "Resampling", Image)
            image.thumbnail((max_side, max_side), resample_attr.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _request_caption(
            self,
            tensor: torch.Tensor,
            data: Dict[str, Any],
            detail_cfg: dict,
            *,
            endpoint_override: Optional[str] = None,
    ) -> str:
        endpoint = endpoint_override or detail_cfg.get('caption_endpoint') or "http://localhost:11434"
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
                    if not self._using_managed_servers():
                        ollama_manager.ensure_running(force_restart=True)
                elif self._should_attempt_restart(exc):
                    if not self._using_managed_servers():
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

        self._ensure_model_available_local(model, endpoint, timeout)

    @staticmethod
    def _debug_show_console() -> bool:
        return True

    def _ensure_model_available_local(self, model: str, endpoint: str, timeout: float) -> None:
        if model in self._models_pulled:
            return

        if not endpoint.startswith("http://localhost") and not endpoint.startswith("http://127.0.0.1"):
            raise RuntimeError(
                f"Ollama model '{model}' not found at {endpoint} and auto-pull only works for local servers. "
                f"Install the model manually or adjust the caption model setting."
            )

        def _pull(target_model: str) -> None:
            self._run_cli_command(
                ["ollama", "pull", target_model],
                timeout=max(timeout * 2, 60.0),
                check=True,
            )

        try:
            _pull(model)
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
        except subprocess.CalledProcessError:
            pass  # fall through to fallback

        # fallback: query the manifest and pick the smallest matching model
        try:
            list_proc = self._run_cli_command(
                ["ollama", "list"],
                timeout=timeout,
                check=True,
            )
            available_names = self._parse_ollama_list(list_proc.stdout or "")
        except Exception as exc:
            stdout = getattr(exc, "stdout", "")
            stderr = getattr(exc, "stderr", "")
            raise RuntimeError(
                f"Failed to auto-pull Ollama model '{model}'. Command output:\n{stdout}\n{stderr}"
            ) from exc

        manifest_model = self._select_manifest_model(model, available_names)
        if manifest_model is None:
            raise RuntimeError(
                f"Failed to auto-pull Ollama model '{model}'. Ensure the name is correct or install it manually."
            )

        try:
            _pull(manifest_model)
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
        
        # Only print progress if enabled, to reduce noise when acting as passthrough
        # Note: _caption_gpu_config can be None when multi-GPU captioning is not configured
        if self._caption_gpu_config is not None and not self._caption_gpu_config.enabled:
             # We should check the concept config but we don't have it here easily without fetching upstream?
             # But wait, _finalize_output passes detail_cfg.
             pass

    def _finalize_output(self, data: Dict[str, Any], detail_cfg: dict) -> Dict[str, Any]:
        if detail_cfg.get('enabled', False):
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
        
        # Allow python client for any local port (127.0.0.1 or localhost)
        # The client handles connection details better than raw requests on Windows
        return (endpoint_normalized.startswith("http://localhost:") or 
                endpoint_normalized.startswith("http://127.0.0.1:"))

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
                if not self._using_managed_servers():
                    ollama_manager.ensure_running()
                # Pass endpoint to _call_ollama_chat_with_timeout to use specific client host
                response = self._call_ollama_chat_with_timeout(model, messages, options, timeout, endpoint=endpoint)
                raw_caption = self._parse_ollama_response(response, is_thinking_model)
                if not raw_caption:
                    raise RuntimeError("Received empty caption from Ollama.")
                return self._sanitize_caption(raw_caption)
            except TimeoutError as exc:
                last_error = exc
                if not self._using_managed_servers():
                    ollama_manager.ensure_running(force_restart=True)
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "model" in message and "not found" in message:
                    raise
                if not self._using_managed_servers():
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
            endpoint: str = None,
    ):
        if _OLLAMA_MODULE is None:
            raise RuntimeError("Ollama Python module is not available.")

        def _invoke():
            # Use Client to support custom host/port safely
            # ollama.Client expects host in format "127.0.0.1:port" (no http:// prefix)
            if endpoint:
                host = endpoint.replace("http://", "").replace("https://", "")
                client = _OLLAMA_MODULE.Client(host=host)
            else:
                client = _OLLAMA_MODULE
            return client.chat(
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
        try:
            rank, world_size, _ = self._distributed_status()
            rank_label = f"[Rank {rank}/{world_size}]" if world_size > 1 else ""
            print(f"[Detail Captions Debug]{rank_label} _maybe_shutdown_ollama: START, has_shutdown={self._has_shutdown_ollama}")
            sys.stdout.flush()
        except Exception:
            pass
        
        if self._has_shutdown_ollama:
            return
        
        if not detail_cfg:
            return
        
        if detail_cfg.get('regenerate_each_epoch', False):
            return
        
        if not detail_cfg.get('enable_captioning', False):
            return
        
        try:
            ollama_manager.cleanup()
        finally:
            self._has_shutdown_ollama = True


class _DefaultDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"

