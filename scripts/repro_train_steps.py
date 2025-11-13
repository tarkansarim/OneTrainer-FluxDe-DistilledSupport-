import argparse
import json
import os
import sys
import time
from contextlib import suppress

import torch

# Project imports (assumes PYTHONPATH includes project root and venv mgds src)
from modules.util.config.TrainConfig import TrainConfig
from modules.util import create
from modules.util import path_util
from modules.util.ollama_manager import prepare as ollama_prepare, cleanup as ollama_cleanup
from modules.util.torch_util import torch_gc
import modules.util.multi_gpu_util as multi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce UI training steps with fine-grained prints.")
    p.add_argument("--config", required=True, help="Absolute path to TrainConfig JSON")
    p.add_argument("--skip-model", action="store_true", help="Skip model load/setup (data loader only)")
    p.add_argument("--batches", type=int, default=1, help="How many batches to fetch to verify iteration")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load config from file
    print(f"[Repro] Loading config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    config = TrainConfig.default_values().from_dict(raw)

    # Canonical workspace dirs
    path_util.canonical_join(config.workspace_dir)  # ensure no crash on join

    # 2) Prepare Ollama like UI
    print("[Repro] Preparing Ollama (UI behavior)")
    with suppress(Exception):
        ollama_prepare(
            config.train_device,
            getattr(config, "device_indexes", ""),
            bool(getattr(config, "multi_gpu", False)),
        )

    model = None
    model_loader = None
    model_setup = None
    data_loader = None

    try:
        # 3) Create model loader/setup and load model (optional)
        if not args.skip_model:
            print("[Repro] Creating model loader/setup")
            model_loader = create.create_model_loader(config.model_type, config.training_method)
            model_setup = create.create_model_setup(
                config.model_type,
                torch.device(config.train_device),
                torch.device(config.temp_device),
                config.training_method,
                config.debug_mode,
            )

            print("[Repro] Loading model")
            model = model_loader.load(
                model_type=config.model_type,
                model_names=config.model_names(),
                weight_dtypes=config.weight_dtypes(),
            )
            model.train_config = config

            print("[Repro] Running model setup")
            model_setup.setup_optimizations(model, config)
            model_setup.setup_train_device(model, config)
            model_setup.setup_model(model, config)

            print("[Repro] Moving model to temp device for eval")
            with suppress(Exception):
                model.to(torch.device(config.temp_device))
            with suppress(Exception):
                model.eval()

        torch_gc()

        # 4) Create data loader (same as BaseTrainer.start())
        print("[Repro] Creating data loader")
        # Minimal model object is accepted by loaders that don't need it
        dl_model = model if model is not None else object()
        data_loader = create.create_data_loader(
            torch.device(config.train_device),
            torch.device(config.temp_device),
            dl_model,
            config.model_type,
            config.training_method,
            config,
            getattr(dl_model, "train_progress", None),
            False,
        )

        dataset = data_loader.get_data_set()

        # 5) Approximate length (like GenericTrainer)
        approx_len = dataset.approximate_length()
        print(f"[Repro] approximate_length (batches): {approx_len}")

        # 6) Ensure model is on train device for caching (mimics before_cache_fun)
        if model is not None:
            print("[Repro] Moving model components to train device for caching")
            with suppress(Exception):
                model.to(torch.device(config.temp_device))
            with suppress(Exception):
                model.vae_to(torch.device(config.train_device))
            with suppress(Exception):
                model.text_encoder_1_to(torch.device(config.train_device))
            with suppress(Exception):
                model.text_encoder_2_to(torch.device(config.train_device))
            with suppress(Exception):
                model.eval()
        
        # 7) Start first epoch and try to fetch batches
        print("[Repro] Starting epoch (dataset.start_next_epoch)")
        dataset.start_next_epoch()
        epoch_len = dataset.approximate_length()
        print(f"[Repro] epoch approximate_length (batches): {epoch_len}")

        print("[Repro] Building batch iterator")
        batches = data_loader.get_data_loader()

        # 7) Iterate N batches with timings
        want = max(1, int(args.batches))
        got = 0
        t0 = time.time()
        print(f"[Repro] Fetching {want} batch(es) ...")
        for batch in batches:
            got += 1
            print(f"[Repro] Got batch {got} (keys: {list(batch.keys())})")
            if got >= want:
                break
        t1 = time.time()
        print(f"[Repro] Done. Fetched {got}/{want} batches in {t1 - t0:.2f}s")

    except Exception as exc:
        print("[Repro] ERROR occurred:")
        import traceback
        traceback.print_exc()
    finally:
        # 8) Cleanup
        print("[Repro] Cleanup")
        with suppress(Exception):
            ollama_cleanup()
        with suppress(Exception):
            torch_gc()


if __name__ == "__main__":
    main()


