# OneTrainer - Flux Dev Dedistilled Fork

OneTrainer is a one-stop solution for all your Diffusion training needs.

> [!NOTE]
> **This fork adds complete support for Flux Dev Dedistilled models**, including traditional CFG during sampling, negative prompts, and crash prevention. All changes are automatic and backward-compatible with regular Flux Dev. See [Flux Dev Dedistilled Support](#flux-dev-dedistilled-support) section below for details.

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
