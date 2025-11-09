# SRPO Wrapper Integration

This project now ships an optional wrapper around Tencent-Hunyuan's
Semantic Relative Preference Optimization (SRPO) training stack. The
wrapper keeps the upstream code isolated while still letting you
kick off full Flux finetunes from OneTrainer.

## Prerequisites

- `git` must be available on your `PATH`; the wrapper clones the SRPO
  repository into `third_party/srpo/vendor` on first use.
- A Bash-compatible shell is required to execute the upstream
  `SRPO_training_hpsv2.sh` script. On Windows you can provide a custom
  path via the `--bash` flag on `scripts/srpo_launcher.py` or set the
  `SRPO_BASH` environment variable.
- Prepare an SRPO working directory that mirrors the structure expected
  by the original repo (for example `data/flux`, `data/rl_embeddings`,
  and any reward model checkpoints).

## Configuring OneTrainer

1. **Select the SRPO model type** – In the top bar, choose
   `Flux Dev (SRPO)`. The training method is restricted to full
   finetuning because SRPO does not currently expose a LoRA pipeline.

2. **Populate SRPO settings** – The *Model* tab now exposes a dedicated
   “SRPO Integration” section:

   | Field | Description |
   | ----- | ----------- |
   | `Working Directory` | Root folder passed as the SRPO process working directory. Place your `data/` tree here. |
   | `Args JSON` | Optional JSON payload consumed by the SRPO launcher for advanced overrides. |
   | `Training Script` | Relative path inside the SRPO repo (defaults to `scripts/finetune/SRPO_training_hpsv2.sh`). |
   | `Repository Ref` | Optional branch/tag/commit to checkout before training. |
   | `Force re-clone` | When enabled, the SRPO repo is recloned at launch to ensure a clean state. |
   | `Extra Launch Args` | Additional command-line arguments appended to the SRPO script (space separated). |

3. **Base model configuration** – The usual Flux model fields remain
   available (base/vae/prior paths). These values are exported as
   environment variables (`SRPO_BASE_MODEL`, `SRPO_VAE_MODEL`, etc.)
   so the upstream script can locate your checkpoints.

## Running from the command line

The helper script can be invoked directly if you want to test the setup
outside the UI:

```
python scripts/srpo_launcher.py \
    --workspace . \
    --working-dir /abs/path/to/your/srpo_workspace \
    --repo-ref main \
    --dry-run
```

Drop the `--dry-run` flag to execute the configured SRPO script. Any
additional arguments can be appended after `--`.

## Customisation

- Set `TrainConfig.srpo_args_file` to point at a JSON descriptor if you
  want to reproduce the upstream shell scripts exactly (the file should
  contain keys for `args`, `env`, or `working_dir`).
- Use `TrainConfig.srpo_extra_args` for lightweight overrides such as
  `--max_train_steps 1000`.
- Environment variables passed by OneTrainer include:
  - `SRPO_WORKSPACE_DIR`
  - `SRPO_CACHE_DIR`
  - `SRPO_BASE_MODEL`
  - `SRPO_VAE_MODEL`
  - `SRPO_LORA` (if populated)
  - `SRPO_CONCEPT_FILE`

These can be consumed in custom wrappers or by modifying the SRPO shell
scripts.

## Troubleshooting

- If the clone step fails, check that `git` is accessible and that the
  target path (`third_party/srpo/vendor`) is writable.
- On Windows, provide a Bash executable via `SRPO_BASH` if you do not
  have a global `bash` command (Git Bash works well).
- The wrapper propagates OneTrainer stop requests; pressing *Stop
  Training* will terminate the SRPO process gracefully.



