from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List

import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download, snapshot_download
try:
    from huggingface_hub.utils import HfHubHTTPError  # type: ignore[import]
except ImportError:
    try:
        from huggingface_hub.utils._errors import HfHubHTTPError  # type: ignore[import]
    except ImportError:
        class HfHubHTTPError(Exception):
            """Fallback HTTP error when huggingface_hub lacks HfHubHTTPError."""
            pass

from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util import path_util


def _resolve_path(base: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_concepts(config: TrainConfig, workspace_root: Path, project_root: Path) -> list[ConceptConfig]:
    concepts: list[ConceptConfig] | None = config.concepts
    if concepts is None:
        concept_file = _resolve_path(workspace_root, config.concept_file_name)
        if not concept_file.exists():
            concept_file = _resolve_path(project_root, config.concept_file_name)
        if not concept_file.exists():
            concept_file = Path(config.concept_file_name).expanduser().resolve()
        if not concept_file.exists():
            raise RuntimeError(f"Concept file not found: {concept_file}")
        with open(concept_file, "r", encoding="utf-8") as fh:
            concept_dicts = json.load(fh)
        concepts = [ConceptConfig.default_values().from_dict(c) for c in concept_dicts]
    return concepts


def _read_prompt_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh.read().splitlines()]
        return [line for line in lines if line]
    except OSError:
        return []


def _collect_prompts(config: TrainConfig, workspace_root: Path) -> list[str]:
    image_extensions = path_util.supported_image_extensions()
    prompts: list[str] = []

    project_root = Path(__file__).resolve().parents[2]

    for concept in _load_concepts(config, workspace_root, project_root):
        if not concept.enabled:
            continue
        if ConceptType(concept.type) != ConceptType.STANDARD:
            continue

        base_path = _resolve_path(workspace_root, concept.path)
        if not base_path.exists():
            continue

        include_sub = concept.include_subdirectories
        files: Iterable[Path]
        if include_sub:
            files = base_path.rglob("*")
        else:
            files = base_path.glob("*")

        concept_prompts: list[str] = []
        if concept.text.prompt_source == "concept" and concept.text.prompt_path:
            concept_prompt_path = _resolve_path(base_path, concept.text.prompt_path)
            if not concept_prompt_path.is_absolute():
                concept_prompt_path = _resolve_path(workspace_root, concept.text.prompt_path)
            concept_prompts = _read_prompt_file(concept_prompt_path)

        for entry in files:
            if entry.suffix.lower() not in image_extensions:
                continue

            prompt_source = concept.text.prompt_source
            variations = max(1, concept.text_variations)
            selected_prompts: list[str] = []

            if prompt_source == "sample":
                sample_prompt = entry.with_suffix(".txt")
                selected_prompts = _read_prompt_file(sample_prompt)[:variations]
            elif prompt_source == "concept":
                selected_prompts = concept_prompts[:variations]
            elif prompt_source == "filename":
                selected_prompts = [entry.stem]
            else:
                selected_prompts = []

            for prompt in selected_prompts:
                text = prompt.strip()
                if text:
                    prompts.append(text)

    if not prompts:
        raise RuntimeError("No prompts could be collected from the active concepts. "
                           "Ensure your concept files include prompts or captions.")
    return prompts


def _relative_to_working_dir(path: Path, working_dir: Path) -> str:
    try:
        return str(path.relative_to(working_dir).as_posix())
    except ValueError:
        return str(path.resolve().as_posix())


def prepare_srpo_dataset(config: TrainConfig, callbacks: TrainCallbacks | None = None) -> Path:
    workspace_root = Path(config.workspace_dir).resolve()
    working_dir = Path(config.srpo_working_dir or "").expanduser()

    if not working_dir:
        raise RuntimeError("SRPO working directory is not set in the training configuration.")

    working_dir = working_dir.resolve()
    if not working_dir.exists():
        raise RuntimeError(f"SRPO working directory does not exist: {working_dir}")

    if callbacks:
        callbacks.on_update_status("Collecting prompts for SRPO dataset")

    prompts = _collect_prompts(config, workspace_root)

    dataset_root = working_dir / "data" / "rl_embeddings"
    prompt_embed_dir = dataset_root / "prompt_embed"
    pooled_dir = dataset_root / "pooled_prompt_embeds"
    text_ids_dir = dataset_root / "text_ids"
    dataset_root.mkdir(parents=True, exist_ok=True)
    prompt_embed_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir.mkdir(parents=True, exist_ok=True)
    text_ids_dir.mkdir(parents=True, exist_ok=True)

    prompts_txt = dataset_root / "prompts.txt"
    with open(prompts_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(prompts))

    unique_prompts = list(OrderedDict.fromkeys(prompts).keys())

    if callbacks:
        callbacks.on_update_status(f"Encoding {len(unique_prompts)} unique prompts for SRPO dataset")

    base_model_name = config.model_names().base_model
    if base_model_name:
        model_path = Path(base_model_name)
        if model_path.exists():
            model_location: str | Path = model_path
        else:
            model_location = base_model_name
    else:
        default_flux_dir = working_dir / "data" / "flux"
        if not default_flux_dir.exists():
            raise RuntimeError(
                "SRPO base model directory not found. Either set a base model in the training configuration "
                "or download the FLUX weights to '<SRPO working dir>/data/flux'."
            )
        model_location = default_flux_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = FluxPipeline.from_pretrained(
        model_location,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    prompt_cache: dict[str, str] = {}
    entries: list[dict] = []
    batch_size = 8
    counter = 0

    with torch.inference_mode():
        total_batches = (len(unique_prompts) + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, len(unique_prompts), batch_size), start=1):
            batch = unique_prompts[start:start + batch_size]
            if callbacks:
                callbacks.on_update_status(f"Encoding SRPO prompts batch {batch_idx}/{total_batches}")
            prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
                prompt=batch,
                prompt_2=batch,
            )

            prompt_embeds = prompt_embeds.cpu()
            pooled_embeds = pooled_embeds.cpu()
            text_ids = text_ids.cpu()

            for idx, prompt in enumerate(batch):
                file_id = f"{counter:08d}"
                counter += 1

                torch.save(prompt_embeds[idx], prompt_embed_dir / f"{file_id}.pt")
                torch.save(pooled_embeds[idx], pooled_dir / f"{file_id}.pt")
                torch.save(text_ids[idx], text_ids_dir / f"{file_id}.pt")

                prompt_cache[prompt] = file_id

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for prompt in prompts:
        file_id = prompt_cache[prompt]
        entries.append(
            {
                "prompt_embed_path": f"{file_id}.pt",
                "pooled_prompt_embeds_path": f"{file_id}.pt",
                "text_ids": f"{file_id}.pt",
                "caption": prompt,
            }
        )

    dataset_json = dataset_root / "videos2caption2.json"
    with open(dataset_json, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)

    if callbacks:
        callbacks.on_update_status(f"SRPO dataset ready with {len(entries)} samples")

    return dataset_json


def ensure_srpo_assets(
    config: TrainConfig,
    working_dir: Path,
    callbacks: TrainCallbacks | None = None,
) -> None:
    def status(message: str) -> None:
        if callbacks:
            callbacks.on_update_status(message)

    reward_model = (config.srpo_reward_model or "HPS").strip().upper()

    # Always ensure CLIP processor weights exist (required for HPS and PickScore workflows).
    clip_dir = working_dir / "data" / "clip"
    clip_dir_parent = clip_dir.parent
    clip_dir_parent.mkdir(parents=True, exist_ok=True)

    if not clip_dir.exists() or not any(clip_dir.iterdir()):
        status("Downloading CLIP weights for SRPO (laion/CLIP-ViT-H-14-laion2B-s32B-b79K)")
        try:
            snapshot_download(
                repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                local_dir=clip_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except HfHubHTTPError as exc:
            raise RuntimeError(
                "Failed to download CLIP weights required by SRPO. "
                "Ensure you have Hugging Face credentials configured."
            ) from exc

    # Ensure HPS checkpoint when using the HPS reward model.
    if reward_model == "HPS":
        hps_dir = working_dir / "hps_ckpt"
        hps_dir.mkdir(parents=True, exist_ok=True)
        hps_ckpt = hps_dir / "HPS_v2.1_compressed.pt"
        if not hps_ckpt.exists():
            status("Downloading HPS-v2.1 reward checkpoint")
            try:
                hf_hub_download(
                    repo_id="xswu/HPSv2",
                    filename="HPS_v2.1_compressed.pt",
                    local_dir=hps_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            except HfHubHTTPError as exc:
                raise RuntimeError(
                    "Failed to download HPS_v2.1_compressed.pt. "
                    "Please ensure Hugging Face access (xswu/HPSv2)."
                ) from exc

    # PickScore-specific weights.
    if reward_model == "PICKSCORE":
        ps_dir = working_dir / "data" / "ps"
        ps_dir.parent.mkdir(parents=True, exist_ok=True)
        if not ps_dir.exists() or not any(ps_dir.iterdir()):
            status("Downloading PickScore weights (yuvalkirstain/PickScore_v1)")
            try:
                snapshot_download(
                    repo_id="yuvalkirstain/PickScore_v1",
                    local_dir=ps_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            except HfHubHTTPError as exc:
                raise RuntimeError(
                    "Failed to download PickScore weights (yuvalkirstain/PickScore_v1). "
                    "Ensure you have access to the repository."
                ) from exc


