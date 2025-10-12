# Flux Dedistilled Fixes for OneTrainer

This document describes the fixes required to enable **Flux Dev Dedistilled** model support in OneTrainer.

## Table of Contents

- [Overview](#overview)
- [What's Fixed](#whats-fixed)
- [Quick Start](#quick-start)
- [When to Use This Tool](#when-to-use-this-tool)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Manual Application](#manual-application)
- [Technical Details](#technical-details)
- [Rollback Instructions](#rollback-instructions)

---

## Overview

Flux Dev Dedistilled is a variant of Flux Dev that removes the guidance distillation, allowing for traditional Classifier-Free Guidance (CFG) during inference. However, OneTrainer's default implementation assumes all Flux models use embedded guidance, causing crashes and poor sampling quality with dedistilled models.

These fixes enable:
- ✅ Training Flux Dev Dedistilled without crashes
- ✅ Traditional CFG during sample generation
- ✅ Proper guidance scale handling
- ✅ Support for negative prompts in sampling

**Regular Flux Dev models remain completely unaffected** - the fixes automatically detect model type.

---

## What's Fixed

### Fix 1: Crash Prevention (Critical)

**Problem:** Dedistilled models don't use embedded guidance, causing `AttributeError: 'NoneType' object has no attribute 'to'`

**Solution:** Added None-checks before calling `.to()` on guidance parameter

**Files Modified:**
- `modules/modelSampler/FluxSampler.py` (2 locations)
- `modules/modelSetup/BaseFluxSetup.py` (1 location)

### Fix 2: Traditional CFG Support (Feature)

**Problem:** Dedistilled models can't use traditional CFG (unconditional + conditional approach) during sampling, resulting in poor sample quality

**Solution:** Implemented automatic CFG detection and two-pass inference for dedistilled models

**Features:**
- Automatic detection via `transformer.config.guidance_embeds == False`
- Negative prompt encoding and support
- Two-pass transformer forward (unconditional + conditional)
- Proper CFG formula: `uncond + cfg_scale * (cond - uncond)`
- Correct handling of positional encodings (not doubled)

**Files Modified:**
- `modules/modelSampler/FluxSampler.py` (`__sample_base` and `__sample_inpainting` methods)

### Fix 3: Guidance Scale Parameter Copy

**Problem:** Guidance scale from training config wasn't copied to sample generation config, causing sample images to use default value instead of training-specified value

**Solution:** Added `cfg_scale` parameter copy in `SampleConfig.from_train_config()`

**Files Modified:**
- `modules/util/config/SampleConfig.py`

---

## Quick Start

### Option 1: Windows Batch File (Easiest)

1. Navigate to your OneTrainer directory
2. Double-click `apply_flux_dedistilled_fixes.bat`
3. Wait for the script to complete
4. Check the output for success message

### Option 2: Python Script (Cross-Platform)

```bash
cd /path/to/OneTrainer
python apply_flux_dedistilled_fixes.py
```

### Option 3: Git Patch (Advanced)

```bash
cd /path/to/OneTrainer
git apply flux-dedistilled-fixes.patch
```

---

## When to Use This Tool

### Required Scenarios:

1. **After cloning/updating OneTrainer** from the official repository
2. **Before training Flux Dev Dedistilled** for the first time
3. **After OneTrainer updates** that modify the affected files

### Not Required If:

- You're only using regular Flux Dev (not dedistilled)
- You've already applied the fixes and haven't updated OneTrainer
- You're using a fork that already includes these fixes

---

## How It Works

The fix application tool uses a smart two-tier approach:

### Tier 1: Git Patch (Fast Path)

1. Attempts to apply `flux-dedistilled-fixes.patch` using `git apply`
2. If successful, all fixes are applied instantly
3. If conflicts detected, falls back to Tier 2

### Tier 2: Manual Fixes (Fallback)

1. Creates `.bak` backup files
2. Applies each fix individually using intelligent find/replace
3. Verifies all fixes were applied correctly
4. Reports success/failure for each file

### Safety Features:

- ✅ Automatic backups before modification
- ✅ Verification checks after application
- ✅ Rollback capability if something fails
- ✅ Detailed error reporting
- ✅ No changes if fixes already applied

---

## Troubleshooting

### Error: "Required files not found"

**Cause:** Script not run from OneTrainer root directory

**Solution:** 
```bash
cd /path/to/OneTrainer
python apply_flux_dedistilled_fixes.py
```

### Error: "Patch cannot be applied cleanly"

**Cause:** OneTrainer code has changed (newer version or modifications)

**Solution:** The script automatically falls back to manual fixes. If manual fixes also fail, see [Manual Application](#manual-application).

### Error: "Verification failed"

**Cause:** Fixes couldn't be applied due to significant code changes

**Solution:**
1. Check which specific checks failed in the output
2. Try manual application (see below)
3. Restore backups: `cp modules/modelSampler/FluxSampler.py.bak modules/modelSampler/FluxSampler.py`

### Warning: "Pattern not found"

**Cause:** Code has changed significantly or fixes already applied differently

**Solution:** Usually safe to ignore if verification passes. Otherwise, apply manually.

---

## Manual Application

If the automated tool fails, apply fixes manually:

### Fix 1: Guidance None-Check (FluxSampler.py)

**Location 1:** Line ~129 in `__sample_base()` method

Find:
```python
guidance=guidance.to(dtype=self.model.train_dtype.torch_dtype()),
```

Replace with:
```python
guidance=guidance.to(dtype=self.model.train_dtype.torch_dtype()) if guidance is not None else None,
```

**Location 2:** Line ~398 in `__sample_inpainting()` method (same change)

### Fix 2: CFG Implementation (FluxSampler.py)

**Location:** After prompt encoding in `__sample_base()` and `__sample_inpainting()`

Add before denoising loop:
```python
# encode negative prompt for CFG (dedistilled models)
use_cfg = not transformer.config.guidance_embeds and cfg_scale > 1.0
if use_cfg:
    negative_prompt_embedding, negative_pooled_prompt_embedding = self.model.encode_text(
        text=negative_prompt if negative_prompt else "",
        train_device=self.train_device,
        text_encoder_1_layer_skip=text_encoder_1_layer_skip,
        text_encoder_2_layer_skip=text_encoder_2_layer_skip,
        apply_attention_mask=prior_attention_mask,
    )
```

Replace the single transformer forward pass with:
```python
if use_cfg:
    # Flux Dedistilled with CFG: two-pass approach
    latent_model_input_doubled = torch.cat([latent_model_input, latent_model_input], dim=0)
    expanded_timestep_doubled = torch.cat([expanded_timestep, expanded_timestep], dim=0)
    
    combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding], dim=0)
    combined_pooled_embedding = torch.cat([negative_pooled_prompt_embedding, pooled_prompt_embedding], dim=0)
    
    # Note: text_ids and image_ids are NOT doubled (they are positional encodings)
    
    noise_pred_combined = transformer(
        hidden_states=latent_model_input_doubled.to(dtype=self.model.train_dtype.torch_dtype()),
        timestep=expanded_timestep_doubled / 1000,
        guidance=None,
        pooled_projections=combined_pooled_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
        encoder_hidden_states=combined_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
        txt_ids=text_ids.to(dtype=self.model.train_dtype.torch_dtype()),
        img_ids=image_ids.to(dtype=self.model.train_dtype.torch_dtype()),
        joint_attention_kwargs=None,
        return_dict=True
    ).sample
    
    noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2)
    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
else:
    # Single pass (regular Flux Dev with embedded guidance)
    noise_pred = transformer(
        hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
        timestep=expanded_timestep / 1000,
        guidance=guidance.to(dtype=self.model.train_dtype.torch_dtype()) if guidance is not None else None,
        pooled_projections=pooled_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
        encoder_hidden_states=prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
        txt_ids=text_ids.to(dtype=self.model.train_dtype.torch_dtype()),
        img_ids=image_ids.to(dtype=self.model.train_dtype.torch_dtype()),
        joint_attention_kwargs=None,
        return_dict=True
    ).sample
```

### Fix 3: Guidance None-Check (BaseFluxSetup.py)

**Location:** Line ~293 in `predict()` method

Find:
```python
guidance=guidance.to(dtype=model.train_dtype.torch_dtype()),
```

Replace with:
```python
guidance=guidance.to(dtype=model.train_dtype.torch_dtype()) if guidance is not None else None,
```

### Fix 4: Guidance Scale Copy (SampleConfig.py)

**Location:** In `from_train_config()` method, after existing parameter copies

Add:
```python
self.cfg_scale = train_config.prior.guidance_scale
```

---

## Technical Details

### How Dedistilled Detection Works

```python
use_cfg = not transformer.config.guidance_embeds and cfg_scale > 1.0
```

- **Regular Flux Dev:** `guidance_embeds = True` → uses embedded guidance, single pass
- **Flux Dedistilled:** `guidance_embeds = False` → uses traditional CFG, two passes

### Why Positional IDs Aren't Doubled

In the CFG implementation, `text_ids` and `image_ids` represent **positional encodings** - they indicate where each token/patch is in the sequence. These positions are identical for both unconditional and conditional passes, so they should NOT be doubled.

**Incorrect (causes tensor size mismatch):**
```python
text_ids_doubled = torch.cat([text_ids, text_ids], dim=0)
```

**Correct (matches official pipeline):**
```python
txt_ids=text_ids.to(dtype=...)  # Used for both passes
```

### Training vs Sampling

- **Training:** Does NOT use traditional CFG (uses embedded guidance or None)
- **Sampling:** Uses traditional CFG for dedistilled models (two-pass approach)

This is intentional and matches the official implementation and other trainers like Kohya.

### Files Modified Summary

| File | Purpose | Lines Changed | Critical |
|------|---------|--------------|----------|
| `FluxSampler.py` | Sampling/inference | ~80 | Yes |
| `BaseFluxSetup.py` | Training | ~1 | Yes |
| `SampleConfig.py` | Config handling | ~1 | Medium |

---

## Rollback Instructions

### Automatic Rollback (if backups exist)

```bash
# Windows
copy modules\modelSampler\FluxSampler.py.bak modules\modelSampler\FluxSampler.py
copy modules\modelSetup\BaseFluxSetup.py.bak modules\modelSetup\BaseFluxSetup.py
copy modules\util\config\SampleConfig.py.bak modules\util\config\SampleConfig.py

# Linux/Mac
cp modules/modelSampler/FluxSampler.py.bak modules/modelSampler/FluxSampler.py
cp modules/modelSetup/BaseFluxSetup.py.bak modules/modelSetup/BaseFluxSetup.py
cp modules/util/config/SampleConfig.py.bak modules/util/config/SampleConfig.py
```

### Git Rollback (if no local changes)

```bash
git checkout modules/modelSampler/FluxSampler.py
git checkout modules/modelSetup/BaseFluxSetup.py
git checkout modules/util/config/SampleConfig.py
```

---

## Verification

To verify fixes are applied correctly, check for these patterns:

### FluxSampler.py
```bash
grep -n "use_cfg = not transformer.config.guidance_embeds" modules/modelSampler/FluxSampler.py
grep -n "if guidance is not None else None" modules/modelSampler/FluxSampler.py
```

Should find multiple matches.

### BaseFluxSetup.py
```bash
grep -n "if guidance is not None else None" modules/modelSetup/BaseFluxSetup.py
```

Should find 1 match around line 293.

### SampleConfig.py
```bash
grep -n "self.cfg_scale = train_config.prior.guidance_scale" modules/util/config/SampleConfig.py
```

Should find 1 match in `from_train_config()` method.

---

## FAQ

**Q: Will these fixes affect my regular Flux Dev training?**

A: No. The fixes automatically detect model type via `guidance_embeds` config parameter. Regular Flux Dev continues using embedded guidance as before.

**Q: Do I need to retrain my models?**

A: No. These fixes only affect OneTrainer's code, not your trained models.

**Q: Will this work with future OneTrainer updates?**

A: The tool should handle minor updates automatically. For major updates, the manual fallback may be needed. You'll need to reapply fixes after each OneTrainer update.

**Q: Can I contribute these fixes to OneTrainer?**

A: Yes! Consider submitting a pull request to the official OneTrainer repository to benefit all users.

**Q: What if I'm using a different Flux variant?**

A: These fixes specifically target models with `guidance_embeds = False`. Other variants should work as long as they follow this pattern.

---

## Credits

These fixes were developed to enable proper Flux Dev Dedistilled support in OneTrainer, based on analysis of:
- Official `pipeline_flux_de_distill.py` from Hugging Face
- Kohya's `flux_train_utils.py` implementation
- OneTrainer's existing Flux infrastructure

## License

These fixes are provided as-is for use with OneTrainer. Follow OneTrainer's license terms for distribution.

---

**Last Updated:** 2025-10-12
**Compatible with:** OneTrainer (master branch as of October 2025)
**Tested with:** Flux Dev Dedistilled models from Hugging Face

