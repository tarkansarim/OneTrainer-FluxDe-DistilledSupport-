import argparse
import os
import sys
from contextlib import suppress

import torch

# MGDS core
from mgds.MGDS import MGDS, TrainDataLoader
from mgds.PipelineModule import PipelineState

# MGDS pipeline modules
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.OutputPipelineModule import OutputPipelineModule

# Project-specific pipeline modules
from modules.dataLoader.pipelineModules.DetailCropGenerator import DetailCropGenerator
from modules.dataLoader.pipelineModules.CropCaptionGenerator import CropCaptionGenerator
from modules.util import ollama_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick test for DetailCropGenerator and optional Ollama captions.")
    parser.add_argument("--concept_dir", required=True, help="Directory with input images (e.g., H:/.../img)")
    parser.add_argument("--save_dir", required=True, help="Directory to write generated crops/captions")
    parser.add_argument("--with_captions", action="store_true", help="Also request captions from Ollama")
    parser.add_argument("--caption_endpoint", default="http://127.0.0.1:11434", help="Ollama endpoint")
    parser.add_argument("--caption_model", default="qwen2.5vl:3b", help="Ollama model to use")
    parser.add_argument("--caption_timeout", type=float, default=20.0, help="Per-caption timeout (seconds)")
    parser.add_argument("--caption_max_retries", type=int, default=1, help="Retries for captioning")
    parser.add_argument("--batch_size", type=int, default=1, help="Local batch size for MGDS")
    parser.add_argument("--max_items", type=int, default=50, help="Process at most this many crops (items)")
    return parser.parse_args()


def build_concepts(concept_dir: str, save_dir: str, with_captions: bool, endpoint: str,
                   model: str, timeout: float, retries: int) -> list[dict]:
    detail_cfg = {
        "enabled": True,
        "tile_resolution": 512,
        "overlap": 128,
        "include_context_tiles": True,
        "scales": [1.0, 1.5],
        "save_to_disk": True,
        "save_directory": os.path.abspath(save_dir),
        "enable_captioning": bool(with_captions),
        "caption_probability": 1.0 if with_captions else 0.0,
        "caption_endpoint": endpoint,
        "caption_model": model,
        "caption_timeout": float(timeout),
        "caption_max_retries": int(retries),
        "caption_auto_pull": True,
        # Optional prompt templates (safe defaults)
        "caption_system_prompt": "",
        "caption_user_prompt": "Describe the content of the crop succinctly.",
    }

    concept = {
        "enabled": True,
        "path": os.path.abspath(concept_dir),
        "include_subdirectories": True,
        "seed": 0,
        "image": {
            "detail_crops": detail_cfg,
        },
        # Prompt optional; left empty to focus on image
        "text": {
            "prompt_path": "",
        },
    }
    return [concept]


def build_pipeline(batch_size: int, with_captions: bool, save_dir: str):
    # File collection -> image load
    collect_paths = CollectPaths(
        concept_in_name="concept",
        path_in_name="path",
        include_subdirectories_in_name="concept.include_subdirectories",
        enabled_in_name="enabled",
        path_out_name="image_path",
        concept_out_name="concept",
        extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"],
        include_postfix=None,
        exclude_postfix=[],
    )

    load_image = LoadImage(
        path_in_name="image_path",
        image_out_name="image",
        range_min=0,
        range_max=1,
        supported_extensions={".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
    )

    # Detail crops
    detail = DetailCropGenerator(
        image_name="image",
        concept_name="concept",
        image_path_name="image_path",
        additional_image_like_names=None,
        export_root=os.path.abspath(save_dir),
        passthrough_names=["prompt"],
    )

    modules = [collect_paths, load_image, detail]

    # Optional: captions (only triggered if we request caption output downstream)
    if with_captions:
        captions = CropCaptionGenerator(
            image_name="image",
            concept_name="concept",
            prompt_name="prompt",
            image_path_name="image_path",
            additional_passthrough=[],
        )
        modules.append(captions)

    # Output drives iteration; request fields we want materialized
    outputs = ["image_path", "image", "concept"]
    if with_captions:
        outputs.append("prompt_caption")  # triggers actual caption generation

    out = OutputPipelineModule(names=outputs)
    modules.append(out)

    return modules


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Device selection (match training default behavior)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Mirror training behavior: manage Ollama lifecycle when captions are requested
    prepared_ollama = False
    if args.with_captions:
        try:
            ollama_manager.prepare(train_device="cuda", device_indexes="0", multi_gpu=False)
            prepared_ollama = True
        except Exception as exc:
            print(f"[Test] Warning: failed to prepare Ollama automatically: {exc}")

    concepts = build_concepts(
        args.concept_dir,
        args.save_dir,
        args.with_captions,
        args.caption_endpoint,
        args.caption_model,
        args.caption_timeout,
        args.caption_max_retries,
    )

    settings = {
        "target_resolution": "1024x1024",
        "target_frames": 1,
    }

    definition = build_pipeline(args.batch_size, args.with_captions, args.save_dir)

    ds = MGDS(
        device=device,
        concepts=concepts,
        settings=settings,
        definition=definition,
        batch_size=args.batch_size,
        state=PipelineState(4),
        initial_epoch=0,
        initial_epoch_sample=0,
    )

    # One pass
    ds.start_next_epoch()
    approx_len = ds.approximate_length()
    print(f"[Test] approximate_length (batches): {approx_len}")

    processed = 0
    with suppress(StopIteration):
        for item in ds:
            image_path = item.get("image_path", "")
            print(f"[Test] processed: {image_path}")
            processed += 1
            if processed >= args.max_items:
                break

    print(f"[Test] Done. Items processed: {processed}.")
    print(f"[Test] Crops should be under: {os.path.abspath(args.save_dir)}")
    if args.with_captions:
        print("[Test] Captions were requested via CropCaptionGenerator (prompt_caption).")

    if prepared_ollama:
        with suppress(Exception):
            ollama_manager.cleanup()


if __name__ == "__main__":
    main()


