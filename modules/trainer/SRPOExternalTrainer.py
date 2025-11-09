from __future__ import annotations

import contextlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import psutil

from modules.trainer.BaseTrainer import BaseTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.srpo_dataset import ensure_srpo_assets, prepare_srpo_dataset
from third_party.srpo import SRPOBootstrapError, ensure_srpo_repo


class SRPOExternalTrainer(BaseTrainer):
    """Thin wrapper around Tencent-Hunyuan's SRPO training stack.

    The SRPO training code lives out-of-tree; this trainer prepares the
    repository (via :mod:`third_party.srpo`) and delegates execution to the
    ``scripts/srpo_launcher.py`` helper, keeping the integration isolated from
    the generic trainer pipeline.
    """

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)
        self._project_root = Path(__file__).resolve().parents[2]
        self._srpo_repo_path: Path | None = None
        self._process: subprocess.Popen | None = None
        self._srpo_dataset_path: Path | None = None
        self._srpo_prompts_path: Path | None = None
        self._srpo_auto_flags: list[tuple[str, str | None]] = []
        self._srpo_working_dir: Path | None = None

    # ------------------------------------------------------------------
    # BaseTrainer API
    # ------------------------------------------------------------------
    def start(self) -> None:  # noqa: D401 - base class docs suffice
        self.callbacks.on_update_status("Preparing SRPO repository")

        repo_ref = self.config.srpo_repo_ref or None
        try:
            result = ensure_srpo_repo(
                self._project_root,
                ref=repo_ref,
                force_refresh=self.config.srpo_force_refresh,
            )
        except SRPOBootstrapError as exc:
            raise RuntimeError(f"Failed to bootstrap SRPO repository: {exc}") from exc

        self._srpo_repo_path = result.repo_path

        if self.config.multi_gpu:
            raise RuntimeError(
                "SRPO external trainer manages distributed execution internally; "
                "please disable OneTrainer multi-GPU mode."
            )

    def train(self) -> None:  # noqa: D401 - base class docs suffice
        if self._srpo_repo_path is None:
            raise RuntimeError("SRPO repository not prepared; did start() succeed?")

        working_dir = self._resolve_working_dir()
        self._srpo_working_dir = working_dir
        launcher = self._project_root / "scripts" / "srpo_launcher.py"
        if not launcher.exists():
            raise RuntimeError(f"SRPO launcher script missing: {launcher}")

        ensure_srpo_assets(self.config, working_dir, self.callbacks)

        try:
            dataset_path = prepare_srpo_dataset(self.config, self.callbacks)
        except RuntimeError as exc:
            raise RuntimeError(f"Unable to prepare SRPO dataset: {exc}") from exc

        self._srpo_dataset_path = dataset_path
        self._srpo_prompts_path = dataset_path.parent / "prompts.txt"

        args = self._build_launcher_args(launcher, working_dir)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        self.callbacks.on_update_status("Launching SRPO training")

        process = subprocess.Popen(args, cwd=self._project_root, env=env)
        self._process = process

        try:
            self._monitor_process(process)
        finally:
            self._process = None

    def end(self) -> None:  # noqa: D401 - base class docs suffice
        if self._process and self._process.poll() is None:
            self._process.terminate()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_working_dir(self) -> Path:
        raw_path = (self.config.srpo_working_dir or "").strip()
        if not raw_path:
            raise RuntimeError(
                "SRPO working directory is undefined. Set 'srpo_working_dir' in the training config "
                "to point at the SRPO data workspace (expected to contain the data/, output/, etc. folders)."
            )

        working_dir = Path(raw_path).expanduser().resolve()
        if not working_dir.exists():
            raise RuntimeError(f"SRPO working directory does not exist: {working_dir}")

        return working_dir

    def _build_launcher_args(self, launcher: Path, working_dir: Path) -> List[str]:
        args: List[str] = [
            sys.executable,
            str(launcher),
            "--workspace",
            str(self._project_root),
            "--script",
            self.config.srpo_training_script,
            "--working-dir",
            str(working_dir),
        ]

        repo_ref = self.config.srpo_repo_ref.strip()
        if repo_ref:
            args.extend(["--repo-ref", repo_ref])
        if self.config.srpo_force_refresh:
            args.append("--force-refresh")

        self._srpo_auto_flags = self._build_srpo_auto_flags(working_dir)
        args.extend(self._build_env_args())

        args_file = (self.config.srpo_args_file or "").strip()
        if args_file:
            args.extend(["--args-file", str(Path(args_file).expanduser().resolve())])

        extra_args = (self.config.srpo_extra_args or "").strip()
        if extra_args:
            args.append("--")
            args.extend(shlex.split(extra_args))

        return args

    def _build_env_args(self) -> List[str]:
        env_entries: Dict[str, str] = {
            "SRPO_WORKSPACE_DIR": str(Path(self.config.workspace_dir).expanduser().resolve()),
            "SRPO_CACHE_DIR": str(Path(self.config.cache_dir).expanduser().resolve()),
            "SRPO_PYTHON": sys.executable,
        }
        if self._srpo_working_dir:
            env_entries["SRPO_WORKING_DIR"] = str(self._srpo_working_dir.resolve())

        model_names = self.config.model_names()
        if model_names.base_model:
            env_entries["SRPO_BASE_MODEL"] = model_names.base_model
        if model_names.vae_model:
            env_entries["SRPO_VAE_MODEL"] = model_names.vae_model
        if model_names.lora:
            env_entries["SRPO_LORA"] = model_names.lora
        if self._srpo_repo_path:
            env_entries["SRPO_REPO_PATH"] = str(self._srpo_repo_path.resolve())

        if self.config.concept_file_name:
            env_entries["SRPO_CONCEPT_FILE"] = str(Path(self.config.concept_file_name).expanduser().resolve())
        if self._srpo_dataset_path:
            env_entries["SRPO_DATA_JSON"] = str(self._srpo_dataset_path.resolve())
        if self._srpo_prompts_path and self._srpo_prompts_path.exists():
            env_entries["SRPO_PROMPTS_TXT"] = str(self._srpo_prompts_path.resolve())
        if self._srpo_auto_flags:
            serialized = []
            for flag, value in self._srpo_auto_flags:
                if value is None:
                    serialized.append([flag])
                elif isinstance(value, (list, tuple)):
                    serialized.append([flag, *[str(v) for v in value]])
                else:
                    serialized.append([flag, str(value)])
            env_entries["SRPO_AUTO_ARGS"] = json.dumps(serialized)
        if self._srpo_repo_path:
            env_entries["SRPO_REPO_PATH"] = str(self._srpo_repo_path.resolve())

        extra_env = (self.config.srpo_extra_env or "").strip()
        if extra_env:
            normalized = extra_env.replace("\r", "\n")
            normalized = normalized.replace(";", "\n")
            normalized = normalized.replace(",", "\n")
            for line in normalized.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_entries[key.strip()] = value.strip()
                else:
                    env_entries[line] = env_entries.get(line, "")

        env_args: List[str] = []
        for key, value in env_entries.items():
            if value is None:
                continue
            env_args.extend(["--env", f"{key}={value}"])

        return env_args

    def _relative_to_working_dir(self, path: Path, working_dir: Path) -> str:
        try:
            return str(path.resolve().relative_to(working_dir.resolve()).as_posix())
        except ValueError:
            return str(path.resolve().as_posix())

    def _build_srpo_auto_flags(self, working_dir: Path) -> list[tuple[str, str | Sequence[str] | None]]:
        config = self.config
        flags: list[tuple[str, str | Sequence[str] | None]] = []

        def add(flag: str, value: str | int | float | None) -> None:
            if value is None:
                return
            string_value: str
            if isinstance(value, float) and math.isfinite(value) and value.is_integer():
                string_value = str(int(value))
            else:
                string_value = str(value)
            if string_value == "":
                return
            flags.append((flag, string_value))

        def add_multi(flag: str, values: Sequence[str | int | float] | None) -> None:
            if not values:
                return
            str_values: list[str] = []
            for v in values:
                if isinstance(v, float) and math.isfinite(v) and v.is_integer():
                    str_values.append(str(int(v)))
                else:
                    stringified = str(v)
                    if stringified != "":
                        str_values.append(stringified)
            if not str_values:
                return
            flags.append((flag, str_values))

        def add_relative(flag: str, path_value: str) -> None:
            if not path_value:
                return
            path_obj = Path(path_value)
            if path_obj.exists():
                add(flag, self._relative_to_working_dir(path_obj, working_dir))
            else:
                add(flag, path_value)

        def add_bool(flag: str, enabled: bool) -> None:
            if enabled:
                flags.append((flag, None))

        add("--seed", config.srpo_seed)
        add("--train_batch_size", config.srpo_train_batch_size)
        add("--gradient_accumulation_steps", config.srpo_gradient_accumulation_steps)
        add("--learning_rate", config.srpo_learning_rate)
        add("--lr_warmup_steps", config.srpo_lr_warmup_steps)
        add("--weight_decay", config.srpo_weight_decay)
        add("--lr_scheduler", config.srpo_lr_scheduler)
        add("--lr_num_cycles", config.srpo_lr_num_cycles)
        add("--lr_power", config.srpo_lr_power)
        add("--master_weight_type", config.srpo_master_weight_type)
        add("--max_train_steps", config.srpo_max_train_steps)
        add("--sampling_steps", config.srpo_sampling_steps)
        add("--train_guidence", config.srpo_train_guidence)
        add("--vis_guidence", config.srpo_vis_guidence)
        add("--eta", config.srpo_eta)
        add("--max_grad_norm", config.srpo_max_grad_norm)
        add("--ema_decay", config.srpo_ema_decay)
        add("--ema_start_step", config.srpo_ema_start_step)
        add("--cfg", config.srpo_cfg)
        add("--loss_coef", config.srpo_loss_coef)
        add("--num_latent_t", config.srpo_num_latent_t)
        add("--sp_size", config.srpo_sp_size)
        add("--train_sp_batch_size", config.srpo_train_sp_batch_size)
        add("--dataloader_num_workers", config.srpo_dataloader_num_workers)
        add("--timestep_length", config.srpo_timestep_length)
        add("--groundtruth_ratio", config.srpo_groundtruth_ratio)
        add_multi("--discount_inv", self._parse_pair(config.srpo_discount_inv, "srpo_discount_inv"))
        add_multi("--discount_pos", self._parse_pair(config.srpo_discount_pos, "srpo_discount_pos"))
        add_multi("--train_timestep", self._parse_pair(config.srpo_train_timestep, "srpo_train_timestep"))
        add("--shift", config.srpo_shift)
        add("--h", config.srpo_h)
        add("--w", config.srpo_w)
        add("--t", config.srpo_t)
        add("--vis_sampling_step", config.srpo_vis_sampling_step)
        add("--vis_size", config.srpo_vis_size)
        add("--vis_tile_size", config.srpo_vis_tile_size)
        add("--image_p", config.srpo_image_prefix)
        add("--sampler_seed", config.srpo_sampler_seed)
        add("--mixed_precision", config.srpo_mixed_precision)
        add("--reward_model", config.srpo_reward_model)
        add("--checkpointing_steps", config.srpo_checkpointing_steps)
        add("--selective_checkpointing", config.srpo_selective_checkpointing)

        add_bool("--gradient_checkpointing", config.srpo_gradient_checkpointing)
        add_bool("--allow_tf32", config.srpo_allow_tf32)
        add_bool("--ignore_last", config.srpo_ignore_last)
        add_bool("--precondition_outputs", config.srpo_precondition_outputs)
        add_bool("--use_cpu_offload", config.srpo_use_cpu_offload)

        add_relative("--output_dir", config.srpo_output_dir)
        add_relative("--cache_dir", config.srpo_cache_dir)
        add_relative("--logging_dir", config.srpo_logging_dir)
        if config.srpo_resume_from_checkpoint:
            add_relative("--resume_from_checkpoint", config.srpo_resume_from_checkpoint)

        model_names = config.model_names()
        if model_names.base_model:
            add_relative("--pretrained_model_name_or_path", model_names.base_model)
        if model_names.vae_model:
            add_relative("--vae_model_path", model_names.vae_model)

        return flags

    def _parse_pair(self, value: str, field_name: str) -> list[str]:
        if not value:
            return []
        parts = value.replace(",", " ").split()
        if len(parts) < 2:
            raise RuntimeError(f"Expected two values for {field_name}, received '{value}'. Please provide two space-separated numbers.")
        return parts[:2]

    def _monitor_process(self, process: subprocess.Popen) -> None:
        try:
            while True:
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        raise RuntimeError(f"SRPO training exited with code {return_code}")
                    self.callbacks.on_update_status("SRPO training finished")
                    break

                if self.commands.get_stop_command():
                    self.callbacks.on_update_status("Stopping SRPO training (requested)")
                    self._terminate_process_tree(process.pid)
                    raise RuntimeError("SRPO training interrupted by user")

                time.sleep(1)
        except KeyboardInterrupt:
            process.terminate()
            raise

    def _terminate_process_tree(self, pid: int) -> None:
        try:
            parent = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

        children = parent.children(recursive=True)
        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child.terminate()

        gone, alive = psutil.wait_procs(children, timeout=10)
        for child in alive:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child.kill()

        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            parent.terminate()
            try:
                parent.wait(timeout=10)
            except psutil.TimeoutExpired:
                parent.kill()

