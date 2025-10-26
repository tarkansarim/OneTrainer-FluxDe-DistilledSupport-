# Block-Wise Learning Rate Control for Flux Models

## Overview

This feature adds the ability to set individual learning rate multipliers (0-1) for each Flux transformer block (double and single blocks) through a GUI window accessible via a three-dot button next to the "Layer Filter" dropdown in the Training tab.

## What's New

### GUI Enhancement

- Added a three-dot button (…) next to the "Layer Filter" dropdown in the Training tab
- When Flux model is selected and "blocks" preset is chosen, clicking the button opens the Block Learning Rate window
- The window displays sliders for all double blocks (0-18) and single blocks (0-37)
- Each slider ranges from 0.0 to 1.0, representing a multiplier for the base learning rate
- Includes a "Reset All to 1.0" button to restore default values

### Configuration

- New config field: `block_learning_rate_multiplier` (dict mapping block names to multipliers)
- Block names format: `double_block_0`, `double_block_1`, ..., `single_block_0`, `single_block_1`, ...
- Default multiplier is 1.0 for all blocks (no change from current behavior)

### Backend Implementation

- Modified `FluxLoRASetup` to create separate parameter groups for each block when block-wise LRs are enabled
- Modified `FluxFineTuneSetup` with the same functionality for fine-tuning
- Created `flux_block_util.py` with utility functions for block detection and parameter parsing
- The base learning rate (from `config.prior.learning_rate`) is multiplied by each block's multiplier

## Use Cases

This feature enables advanced training techniques such as:

1. **Layer-wise LR decay**: Lower learning rates for early layers (e.g., 0.3-0.5), higher for later layers (0.8-1.0)
2. **Selective fine-tuning**: Focus training on specific blocks by reducing LR on others
3. **Preservation training**: Keep early layers mostly frozen (0.1-0.2) while training later layers (0.8-1.0)

## Files Modified

1. `modules/util/config/TrainConfig.py` - Added `block_learning_rate_multiplier` field
2. `modules/util/flux_block_util.py` - NEW: Utility functions for block detection
3. `modules/ui/BlockLearningRateWindow.py` - NEW: GUI window with sliders for block LR configuration
4. `modules/ui/TrainingTab.py` - Modified to use `options_adv` and added handler for three-dot button
5. `modules/modelSetup/FluxLoRASetup.py` - Implemented block-wise parameter groups
6. `modules/modelSetup/FluxFineTuneSetup.py` - Implemented block-wise parameter groups

## How to Use

1. Load a Flux model in the GUI
2. Select "blocks" from the Layer Filter preset dropdown in the Training tab
3. Click the three-dot button (…) next to the dropdown
4. Adjust sliders for individual blocks (0.0 = no learning, 1.0 = full base LR)
5. Click "OK" to save the configuration
6. Start training - the optimizer will use different learning rates for each block

## Technical Details

### Parameter Grouping

When block-wise learning rates are enabled:
- Each transformer block's parameters are grouped separately
- For LoRA training: Iterates through `LoRAModuleWrapper.lora_modules` dict to access LoRA modules
- For Fine-tuning: Iterates through the transformer's `named_parameters()` directly
- Each group gets its own learning rate: `block_lr = base_lr * multiplier`
- Parameters not belonging to specific blocks (e.g., norm layers) are grouped as "other"

### Compatibility

- Works with both LoRA and Fine-Tuning methods
- Compatible with existing layer filters and freeze settings
- Saves and loads correctly with training configs

### Performance

- No performance impact when not using block-wise LRs (default behavior unchanged)
- Minimal overhead when enabled (just creates more parameter groups for optimizer)

## Example Configuration

```json
{
  "layer_filter_preset": "blocks",
  "block_learning_rate_multiplier": {
    "double_block_0": 0.3,
    "double_block_1": 0.4,
    "double_block_2": 0.5,
    ...
    "single_block_0": 0.7,
    "single_block_1": 0.8,
    ...
  }
}
```

## Notes

- The three-dot button only appears when using `options_adv` for the layer filter
- The window only opens for Flux models with "blocks" preset selected
- Empty or missing multipliers default to 1.0 (no change from base LR)
- This follows the same pattern as other advanced parameter windows (Optimizer, Scheduler, etc.)


