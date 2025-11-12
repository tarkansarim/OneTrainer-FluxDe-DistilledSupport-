# Quick Start Guide

This Guide is intended to explain the basic principles of fine-tuning a model with OneTrainer in the order you will see
them in the UI. For more in-depth explanations, please check the wiki here: https://github.com/Nerogar/OneTrainer/wiki

### Preparing a Dataset

Place the images you want to train on in any directory you want. For best results, you should add prompts to every
image. There are two options:

1. For each image, add a .txt file with the same name in the same directory. The text file should contain the prompt to
   train on. i.e image1.jpg and image1.txt
2. Rename all images with the prompt you want to train on.

### Presets

To quickly get started, OneTrainer provides a set of presets for the most useful configurations. Select the one that
best
fits your needs, then change settings as you need. You can also add your own presets.

### Workspaces

OneTrainer uses workspaces to separate your training runs. A workspace contains all the backup, sampling and TensorBoard
data of a single run. When you start a new training run, it is recommended that you select an empty directory as your
workspace.

### Caching

Caching speeds up the training, by saving intermediate data from your training images on disc. This data only needs to
be created once, and can be re-used between epochs or even different training runs. If you change any data related
settings, you should either clear the cache directory, or choose a new one.

### Tensorboard

TensorBoard enables easy tracking of the training progress during training. To use it, enable Tensorboard. During the
training run, click on the TensorBoard button at the bottom to open the web interface. This web interface will display
loss values and samples.

### Input Model Settings

Select a base model to train on. This can be a model in many different formats:

1. A filename of a checkpoint either in ckpt or safetensors format
2. A directory name of a model in diffusers format
3. A Hugging Face repository name
4. A backup directory from a previous training run

You also need to specify the type of the input model.

### Output Model Settings

Simply specify the file or directory name where you want to save the final model, and the output format.
The data type defines the precision for the final model. Float16 will create smaller models, but the quality can be
reduced.

### Data Settings

Aspect Ratio Bucketing makes it possible to train the model on images with different aspect ratios at the same time. All
images are resized to roughly match the same total pixel count.

### Latent caching

Latent caching (as described in the caching section) will speed up the training by saving intermediate data. If this
setting is enabled, some data will be calculated based on your training images and saved to disc. If you enable data
augmentation, you should increase the "Image Variations" setting of that concept, otherwise only a single augmentation
version will be cached.

### Adding concepts.

Concepts define the data you want to train on. You can first create a configuration. Then you can add as many concepts
to that configuration as you want.

After adding a concept, you can click on the new widget that appears to open the
details window. In that window you can optionally select a name, then specify the directory where the images are saved.
The prompt source defines where the prompts are loaded from. You can choose to load them from `.txt` files in the same
directory, from the image filenames, or from a single `.txt` file that contains multiple lines to randomly choose from. If
you choose to use a single `.txt` file, you also need to specify where that file is saved. Optionally, you can also
include subdirectories of the selected directory in the concept definition.

On the augmentation tab, you can define how OneTrainer should modify your images before training on them. These
modifications can increase the diversity of your training data.

### Training Settings

Here you can set all the parameters that are used during training. You probably won't need to change most of these
settings, but here are some examples of the most important ones:

- Learning Rate: This setting defines how quickly you want the model to learn. But setting it too high will break the
  model.
- Epochs: This defines how many steps the model should be trained on each image in the dataset.
- Batch Size: Defines the number of images to train at once. A higher setting will lead to more generalized results, but
  will take longer to train and requires more VRAM.
- Accumulation Steps: If you can't increase the Batch Size because of VRAM constraints, you can increase this setting to
  create the same effect, but slow down the training even more.

"Train Text Encoder" and "Train UNet" can be deactivated to only train parts of the model. In most cases, both should
stay active. You can also specify how many epochs each part should be trained for, as well as a specific learning rate
for each part.

### Train Data Type

Internally, this sets the mixed precision data type when doing the forward pass through the model. This setting trades
precision for speed during training. Not all data types are supported on all GPUs. In practice, float16 only slightly
reduces the quality, while providing a significant speed boost.

### Resolution

Your training images will be resized to that resolution when training. You don't need to do any manual resizing. You can
specify multiple resolutions as a comma-separated list. All resolutions will be trained at the same time.

### Detail Crop Generation

To preserve small-area detail in very large source images, each concept now exposes `image.detail_crops`. When enabled,
OneTrainer will derive additional tiles every epoch directly from the high-resolution originals instead of relying on
pre-generated folders.

- `enabled`: toggles the feature per concept. If disabled, the loader behaves exactly as before.
- `tile_resolution`: square size (in pixels) for each crop. This is typically set to match the training resolution
  (e.g. 1024).
- `overlap`: pixels that neighbouring tiles share. The loader uses dynamic overlap to keep edge tiles full sized while
  avoiding duplicates.
- `blank_std_threshold` / `blank_edge_threshold`: thresholds used to skip near-empty crops.
- `scales`: list of upscale factors (e.g. `[2, 4]`). For each factor, the loader first resizes the image to
  `training_resolution * factor` and then extracts `tile_resolution` crops, mirroring tiled upscaling passes.
- `include_context_tiles`: also slice a version of the image resized back to `tile_resolution` so the model still sees
  full-scene context.
- `include_full_images`: keep the original full-frame image in the dataset alongside the generated tiles. Disable if you
  want the concept to train purely on crops.
- `regenerate_each_epoch`: re-run the tiling pass every epoch. Leave disabled to reuse the cached grid for faster
  training when the layout doesn’t change.
- `enable_captioning`: when enabled, generates captions for crops on the fly through a local Ollama instance.
- `caption_probability`: fraction of detail crops that should receive generated captions (0.0–1.0).
- `caption_model`: Ollama model tag to call (default `qwen2.5vl:3b`).
- `caption_system_prompt` / `caption_user_prompt`: prompt templates used for the vision-language model; `{context}` will be
  replaced with the source image caption.
- `caption_endpoint`, `caption_timeout`, `caption_max_retries`: HTTP endpoint and reliability settings (`caption_timeout`
  defaults to 120s, `caption_max_retries` to 4). Captions are written
- `caption_auto_pull`: when enabled, automatically runs `
- `parallel_workers`: CPU workers used when generating tiles. Set to 0 to pick an automatic value based on available cores.