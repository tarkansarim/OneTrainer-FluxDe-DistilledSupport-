# OneTrainer Plus

OneTrainer is a one-stop solution for all your Diffusion training needs.

> [!NOTE]
> **This fork adds complete support for Flux Dev Dedistilled models**, **detail crop generation with Ollama-based captioning**, **fixes GPU selection in single-GPU mode**, **unlocks T5 token limit to 512**, **fixes Flux Dev sampling guidance**, and **adds block-wise learning rate control**. Includes traditional CFG during sampling, detail crop tiling with AI-generated captions, negative prompts, crash prevention, proper device index handling, extended prompt support, proper sample quality, and individual LR control for each Flux transformer block. All changes are automatic and backward-compatible. See [Detail Crop Captioning](#detail-crop-captioning), [Flux Dev Dedistilled Support](#flux-dev-dedistilled-support) and [Additional Improvements](#additional-improvements-in-this-fork) sections below for details.

<a href="https://discord.gg/KwgcQd5scF"><img src="https://discord.com/api/guilds/1102003518203756564/widget.png" alt="OneTrainer Discord"/></a><br>

## Features

-   **Supported models**: Qwen Image, FLUX.1 (including Flux Dev Dedistilled), Chroma, Stable Diffusion 1.5, 2.0, 2.1, 3.0, 3.5, SDXL, Würstchen-v2, Stable Cascade,
    PixArt-Alpha, PixArt-Sigma, Sana, Hunyuan Video and inpainting models
-   **Model formats**: diffusers and ckpt models
-   **Training methods**: Full fine-tuning, LoRA, embeddings
-   **Masked Training**: Let the training focus on just certain parts of the samples
-   **Automatic backups**: Fully back up your training progress regularly during training. This includes all information to seamlessly continue training
-   **Image augmentation**: Apply random transforms such as rotation, brightness, contrast or saturation to each image sample to quickly create a more diverse dataset
-   **TensorBoard**: A simple TensorBoard integration to track the training progress
-   **Multiple prompts per image**: Train the model on multiple different prompts per image sample
-   **Noise Scheduler Rescaling**: From the paper
    [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)
-   **EMA**: Train your own EMA model. Optionally keep EMA weights in CPU memory to reduce VRAM usage
-   **Aspect Ratio Bucketing**: Automatically train on multiple aspect ratios at a time. Just select the target resolutions, buckets are created automatically
-   **Multi-Resolution Training**: Train multiple resolutions at the same time
-   **Dataset Tooling**: Automatically caption your dataset using BLIP, BLIP2 and WD-1.4, or create masks for masked training using ClipSeg or Rembg
-   **Model Tooling**: Convert between different model formats from a simple UI
-   **Sampling UI**: Sample the model during training without switching to a different application

![OneTrainerGUI.gif](resources/images/OneTrainerGUI.gif)

> [!NOTE]
> Explore our 📚 wiki for essential tips and tutorials after installing. Start [here!](https://github.com/Nerogar/OneTrainer/wiki).
> For command-line usage, see the [CLI Mode section](#cli-mode).


## Installation

> [!IMPORTANT]
> Installing OneTrainer requires Python >=3.10 and <3.13.
> You can download Python at https://www.python.org/downloads/windows/.
> Then follow the below steps.

#### Automatic installation

1. Clone the repository `git clone https://github.com/Nerogar/OneTrainer.git`
2. Run:
    - Windows: Double click or execute `install.bat`
    - Linux and Mac: Execute `install.sh`

    #### Manual installation

    1. Clone the repository `git clone https://github.com/Nerogar/OneTrainer.git`
    2. Navigate into the cloned directory `cd OneTrainer`
    3. Set up a virtual environment `python -m venv venv`
    4. Activate the new venv:
        - Windows: `venv/scripts/activate`
        - Linux and Mac: Depends on your shell, activate the venv accordingly
    5. Install the requirements `pip install -r requirements.txt`

> [!Tip]
> Some Linux distributions are missing required packages for instance: On Ubuntu you must install `libGL`:
>
> ```bash
> sudo apt-get update
> sudo apt-get install libgl1
> ```
>
> Additionally it's been reported Alpine, Arch and Xubuntuu Linux may be missing `tkinter`. Install it via `apk add py3-tk` for Alpine and `sudo pacman -S tk` for Arch.

## Updating

#### Automatic update

-   Run `update.bat` or `update.sh`

#### Manual update

1. Cd to folder containing the repo `cd OneTrainer`
2. Pull changes `git pull`
3. Activate the venv `venv/scripts/activate`
4. Re-install all requirements `pip install -r requirements.txt --force-reinstall`

## Detail Crop Captioning

This fork includes a powerful **Detail Crop Generation and Captioning** system that automatically tiles high-resolution images and generates context-aware captions using local Ollama vision-language models.

### What's Included

- ✅ **Deterministic Tiling**: Automatically splits high-res images into detail crops at multiple scales
- ✅ **Context Tiles**: Optional context tiles at base resolution for better training context
- ✅ **Blank Detection**: Automatically skips crops with low detail (blank/uniform areas)
- ✅ **Ollama Integration**: Generate captions using local vision-language models (e.g., qwen2.5vl:3b)
- ✅ **Upfront Generation**: All captions pre-generated before training starts for consistent experience
- ✅ **Disk Export**: Optionally save crops and captions to disk for inspection
- ✅ **Smart Prompts**: Crops without generated captions use empty strings (avoids misleading parent captions)
- ✅ **Proper Cache Clearing**: Automatically clears old crops/captions when "Clear cache before training" is enabled
- ✅ **Multi-Concept Support**: Each concept can have independent detail crop settings

### Key Features

**For Training:**
- Train on high-resolution details without VRAM limitations
- Multiple scale factors (e.g., 2x, 4x) for multi-scale detail learning
- Probabilistic captioning (e.g., 30% get captions, 70% use empty prompts)
- Per-concept configuration (enable detail crops for specific concepts only)

**For Captioning:**
- Local Ollama server integration (no API costs)
- Automatic model pulling if model isn't available
- Configurable system and user prompts for caption style
- Context-aware captioning (uses parent image caption as context)
- Retry logic and timeout handling for reliability

### Usage

1. **Enable Detail Crops** for a concept in the "detail crops" tab
2. **Configure Scales**: Set comma-separated scales (e.g., `2, 4` for 2x and 4x)
3. **Enable Captioning**: Turn on "Enable Captioning" and set probability (e.g., `0.3` for 30%)
4. **Configure Ollama**: Set model name (e.g., `qwen2.5vl:3b`), endpoint, and prompts
5. **Optional Disk Save**: Enable "Save Crops to Disk" and set directory to export crops/captions
6. **Train**: Start training - crops and captions are generated automatically

### Technical Details

**Pipeline Modules:**
- `DetailCropGenerator`: Generates deterministic tiled crops from high-res images
- `CropCaptionGenerator`: Generates captions via Ollama for detail crops
- `ollama_manager`: Manages Ollama server lifecycle during training

**Caption Generation:**
- Pre-generated upfront during `start_next_epoch()` before training begins
- Uses deterministic selection based on crop hash for consistent probability across epochs
- Saves to `{save_directory}/epoch-{N}/{crop_type}_scale-{scale}/` structure
- Empty prompts for uncaptioned crops prevent misleading training signals

### Files Added/Modified

- `modules/dataLoader/pipelineModules/DetailCropGenerator.py` - Detail crop generation
- `modules/dataLoader/pipelineModules/CropCaptionGenerator.py` - Ollama captioning
- `modules/util/ollama_manager.py` - Ollama server management
- `modules/dataLoader/FluxBaseDataLoader.py` - Caption output integration
- `modules/trainer/GenericTrainer.py` - Proper cache clearing with concept file loading

## Flux Dev Dedistilled Support

This fork includes comprehensive support for **Flux Dev Dedistilled** models, which are variants of Flux Dev with guidance distillation removed, requiring traditional Classifier-Free Guidance (CFG) during inference.

### What's Included

-   ✅ **Crash Prevention**: Fixed `AttributeError` when training with dedistilled models
-   ✅ **Traditional CFG**: Automatic two-pass inference for dedistilled models during sample generation
-   ✅ **Negative Prompt Support**: Full support for negative prompts in sampling
-   ✅ **Proper Guidance Scaling**: Guidance scale parameters correctly passed from training config to sampling
-   ✅ **Auto-Detection**: Automatically detects dedistilled vs regular Flux Dev via `guidance_embeds` config parameter
-   ✅ **Zero Impact on Regular Flux**: All fixes are conditional - regular Flux Dev models work exactly as before

### Key Features

**For Training:**
- Train Flux Dev Dedistilled LoRAs without crashes
- Same training parameters and workflow as regular Flux Dev
- Automatic model type detection (no manual configuration needed)

**For Sampling:**
- Traditional CFG with unconditional + conditional passes
- Negative prompt support during sample generation
- Correct positional encoding handling (matching official implementation)
- Better sample quality during training with proper CFG

### Technical Details

The implementation follows the official `pipeline_flux_de_distill.py` specification and matches behavior from other trainers like Kohya's sd-scripts. Key changes include:

1. **Guidance None-Safety**: Added conditional checks for guidance parameters in both training and sampling
2. **CFG Detection**: Automatic detection via `transformer.config.guidance_embeds == False`
3. **Two-Pass Inference**: Proper implementation of unconditional/conditional approach with correct CFG formula
4. **Parameter Passing**: Fixed guidance_scale propagation from training config to sample generation

### Files Modified

-   `modules/modelSampler/FluxSampler.py` - Added CFG support and crash prevention
-   `modules/modelSetup/BaseFluxSetup.py` - Training crash prevention
-   `modules/util/config/SampleConfig.py` - Guidance scale parameter copying

### Documentation

For complete technical documentation, see [FLUX_DEDISTILLED_FIXES.md](FLUX_DEDISTILLED_FIXES.md)

### Usage

Training Flux Dev Dedistilled works exactly like regular Flux Dev:

1. Load your Flux Dev Dedistilled model in diffusers format
2. Configure training parameters (recommended LR: `1e-4` for LoRA)
3. Set guidance_scale in training tab (recommended: `3.5`)
4. Train normally - everything else is automatic!

Sample images during training will automatically use traditional CFG if your model is dedistilled.

## Additional Improvements in This Fork

### Block-Wise Learning Rate Control for Flux

-   ✅ **Individual Block LRs**: Set different learning rate multipliers for each Flux transformer block (19 double blocks + 38 single blocks)
-   ✅ **GUI Interface**: Access via three-dot button (…) next to Layer Filter dropdown when "blocks" preset is selected
-   ✅ **Slider Control**: Each block has a slider (0.0-1.0) to control its LR multiplier
-   ✅ **Advanced Training**: Enable techniques like layer-wise LR decay, selective fine-tuning, and preservation training
-   ✅ **LoRA & Fine-Tune**: Works with both training methods
-   ✅ **Default Behavior**: All multipliers default to 1.0 (no change from standard training)

See [BLOCK_LEARNING_RATE_FEATURE.md](BLOCK_LEARNING_RATE_FEATURE.md) for detailed documentation.

### GPU Selection Fix

This fork includes comprehensive fixes for GPU selection in single-GPU mode:

-   ✅ **Device Index Support**: The `device_indexes` parameter now works correctly without enabling multi-GPU mode
-   ✅ **GUI Compatibility**: Select any GPU directly from the GUI without multi-GPU overhead
-   ✅ **CLI Parity**: GUI now behaves the same as CLI for GPU selection
-   ✅ **Proper Device Assignment**: All model components (VAE, text encoders) and data tensors are placed on the specified GPU
-   ✅ **Latent Caching Support**: Works with both latent caching enabled and disabled
-   ✅ **Backward Compatible**: Existing configurations continue to work as before

### T5 Token Limit Unlocked for Flux

-   ✅ **Extended Token Limit**: T5 text encoder (text_encoder_2) now supports up to **512 tokens** (increased from 77)
-   ✅ **Long Prompts**: Write detailed, complex prompts without truncation
-   ✅ **CLIP Unchanged**: CLIP text encoder (text_encoder_1) remains at 77 tokens as per its architecture
-   ✅ **Performance Note**: Longer prompts may use slightly more VRAM and take marginally longer to encode

### Flux Dev Sampling Guidance Fix

-   ✅ **Proper Sample Quality**: Flux Dev samples now use embedded guidance value of 3.5 during sampling (instead of copying training value of 1.0)
-   ✅ **No More Washed Out Samples**: Fixes the issue where samples appeared washed out with poor contrast
-   ✅ **Training Unchanged**: Training still uses guidance_scale = 1.0 as intended
-   ✅ **Dedistilled Unaffected**: Flux Dev Dedistilled continues to work correctly with CFG = 3.5
-   ✅ **Automatic Detection**: No configuration changes needed - the fix automatically detects Flux Dev and applies appropriate guidance

**How to Use:**
1. In the GUI, keep "Multi-GPU" switch **OFF**
2. Set "Device Indexes" to your desired GPU number (e.g., `1` for GPU 1, `2` for GPU 2)
3. Train on the selected GPU without distributed training overhead

**Previously:** 
- Setting `device_indexes` in the GUI had no effect unless multi-GPU mode was enabled, always defaulting to GPU 0
- VAE and data tensors could end up on different GPUs, causing device mismatch errors

**Now:** 
- You can select any GPU for single-GPU training, matching the CLI behavior
- All model components and data are correctly placed on the specified GPU, preventing device mismatch errors

## Usage

OneTrainer can be used in **two primary modes**: a graphical user interface (GUI) and a **command-line interface (CLI)** for finer control.

For a technically focused quick start, see the [Quick Start Guide](docs/QuickStartGuide.md) and for a broader overview, see the [Overview documentation](docs/Overview.md). Otherwise visit [our wiki!](https://github.com/Nerogar/OneTrainer)

### GUI Mode

#### Windows

-   To start the UI, navigate to the OneTrainer folder and double-click `start-ui.bat`

#### Unix-based systems

-   Execute `start-ui.sh` and the GUI will pop up.

### CLI Mode

If you need more control or a headless approach OT also supports the command-line interface. All commands **need** to be run inside the active venv created during installation.

All functionality is split into different scripts located in the `scripts` directory. This currently includes:

-   `train.py` The central training script
-   `train_ui.py` A UI for training
-   `caption_ui.py` A UI for manual or automatic captioning and mask creation for masked training
-   `convert_model_ui.py` A UI for model conversions
-   `convert_model.py` A utility to convert between different model formats
-   `sample.py` A utility to sample any model
-   `create_train_files.py` A utility to create files needed when training only from the CLI
-   `generate_captions.py` A utility to automatically create captions for your dataset
-   `generate_masks.py` A utility to automatically create masks for your dataset
-   `calculate_loss.py` A utility to calculate the training loss of every image in your dataset

To learn more about the different parameters, execute `<script-name> -h`. For example `python scripts\train.py -h`

If you are on Mac or Linux, you can also read [the launch script documentation](LAUNCH-SCRIPTS.md) for detailed information about how to run OneTrainer and its various scripts on your system.

## Troubleshooting

For general troubleshooting or questions, ask in [Discussions](https://github.com/Nerogar/OneTrainer/discussions), check the [Wiki](https://github.com/Nerogar/OneTrainer/wiki) or join our [Discord](https://discord.gg/KwgcQd5scF).

If you encounter a reproducible error you first must run update.bat or update.sh and confirm the issue is still able to be reproduced. Then export anonymized debug information to help us solve an issue you are facing and upload it as part of your Github Issues submission.

-   On Windows double click `export_debug.bat`
-   On Unix-based systems execute `./run-cmd.sh generate_debug_report`

These will both create a `debug_report.log`.

> [!WARNING]
> We require this file for GitHub issues going forward. Failure to provide it or not manually providing the necessary info will lead to the issue being closed in most circumstances

## Contributing

Contributions are always welcome in any form. You can open issues, participate in discussions, or even open pull
requests for new or improved functionality. You can find more information about contributing [here](docs/Contributing.md).

Before you start looking at the code, I recommend reading about the project structure [here](docs/ProjectStructure.md).
For in depth discussions, you should consider joining the [Discord](https://discord.gg/KwgcQd5scF) server.

You also **NEED** to **install the required developer dependencies** for your current user and enable the Git commit hooks, via the following commands (works on all platforms; Windows, Linux and Mac):

> [!IMPORTANT]
> Be sure to run those commands _without activating your venv or Conda environment_, since [pre-commit](https://pre-commit.com/) is supposed to be installed outside any environment.

```sh
cd OneTrainer
pip install -r requirements-dev.txt
pre-commit install
```

Now all of your commits will automatically be verified for common errors and code style issues, so that code reviewers can focus on the architecture of your changes without wasting time on style/formatting issues, thus greatly improving the chances that your pull request will be accepted quickly and effortlessly.

## Related Projects

-   **[MGDS](https://github.com/Nerogar/mgds)**: A custom dataset implementation for Pytorch that is built around the idea of a node based graph.
-   **[Stability Matrix](https://github.com/LykosAI/StabilityMatrix)**: A swiss-army knife installer which wraps and installs a broad range of diffusion software packages including OneTrainer
-   **[Visions of Chaos](https://softology.pro/voc.htm)**: A collection of machine learning tools that also includes OneTrainer.
-   **[StableTuner](https://github.com/devilismyfriend/StableTuner)**: A now defunct (archived) training application for Stable Diffusion. OneTrainer takes a lot of inspiration from StableTuner and wouldn't exist without it.
