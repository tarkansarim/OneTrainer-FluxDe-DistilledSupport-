#!/usr/bin/env python3

import os
import sys
import json

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Test just the MGDS creation directly
try:
    from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.TrainProgress import TrainProgress

    print("Testing MGDS creation directly...")

    # Load config from the workspace
    config_path = "workspace/animereal2/config/2025-11-10_18-52-45.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = TrainConfig.default_values()
    config = config.from_dict(config_dict)

    train_progress = TrainProgress()
    train_progress.epoch = 120
    train_progress.epoch_sample = 0

    print("Creating MGDS dataset...")

    # Create the definition like FluxBaseDataLoader does
    from modules.dataLoader.FluxBaseDataLoader import FluxBaseDataLoader
    flux_loader = FluxBaseDataLoader.__new__(FluxBaseDataLoader)  # Create without __init__

    # Copy methods
    flux_loader._enumerate_input_modules = FluxBaseDataLoader._enumerate_input_modules
    flux_loader._load_input_modules = FluxBaseDataLoader._load_input_modules
    flux_loader._detail_crop_modules = FluxBaseDataLoader._detail_crop_modules
    flux_loader._mask_augmentation_modules = FluxBaseDataLoader._mask_augmentation_modules
    flux_loader._aspect_bucketing_in = FluxBaseDataLoader._aspect_bucketing_in
    flux_loader._crop_modules = FluxBaseDataLoader._crop_modules
    flux_loader._augmentation_modules = FluxBaseDataLoader._augmentation_modules
    flux_loader._inpainting_modules = FluxBaseDataLoader._inpainting_modules
    flux_loader._preparation_modules = FluxBaseDataLoader._preparation_modules
    flux_loader._cache_modules = FluxBaseDataLoader._cache_modules
    flux_loader._output_modules = FluxBaseDataLoader._output_modules
    flux_loader._debug_modules = FluxBaseDataLoader._debug_modules

    # Create the definition
    enumerate_input = flux_loader._enumerate_input_modules(config)
    load_input = flux_loader._load_input_modules(config, 'float32')  # Mock dtype
    detail_crops = flux_loader._detail_crop_modules(config)
    mask_augmentation = flux_loader._mask_augmentation_modules(config)
    aspect_bucketing_in = flux_loader._aspect_bucketing_in(config, 64)
    crop_modules = flux_loader._crop_modules(config)
    augmentation_modules = flux_loader._augmentation_modules(config)
    inpainting_modules = flux_loader._inpainting_modules(config)
    preparation_modules = flux_loader._preparation_modules(config, None)  # Mock model
    cache_modules = flux_loader._cache_modules(config, None)  # Mock model
    output_modules = flux_loader._output_modules(config, None)  # Mock model

    definition = [
        enumerate_input,
        load_input,
        detail_crops,
        mask_augmentation,
        aspect_bucketing_in,
        crop_modules,
        augmentation_modules,
        inpainting_modules,
        preparation_modules,
        cache_modules,
        output_modules,
    ]

    print("Creating MGDS with definition...")
    ds = DataLoaderMgdsMixin._create_mgds(
        flux_loader,
        config,
        definition,
        train_progress,
        is_validation=False,
    )

    print(f"Calculating approximate length...")
    length = ds.approximate_length()

    print(f"Dataset approximate length: {length}")

    if length > 0:
        print("SUCCESS: Length calculation works!")
    else:
        print("FAILED: Length is still 0")

    print("Test completed!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
