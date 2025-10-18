# TODO: Performance Optimizations and Bug Fixes

## ✅ COMPLETED: GPU Selection Bug Fix

### Issue
**Location:** `modules/trainer/BaseTrainer.py` line 33
**Problem:** `device_indexes` parameter was ignored when `multi_gpu` was False, always defaulting to GPU 0

### Solution Applied
Modified `BaseTrainer.__init__()` to respect `device_indexes` even in single-GPU mode:
- When `multi_gpu=False` and `device_indexes` is set (e.g., "1"), it now uses the specified GPU
- When `multi_gpu=True`, MultiTrainer handles device selection as before
- Maintains backward compatibility with existing configurations

### Testing
All test cases passed:
- ✓ Default behavior (no device_indexes) → uses cuda (GPU 0)
- ✓ device_indexes='1', multi_gpu=False → uses cuda:1
- ✓ device_indexes='2', multi_gpu=False → uses cuda:2
- ✓ device_indexes='1,2', multi_gpu=False → uses cuda:1 (first in list)
- ✓ multi_gpu=True → delegates to MultiTrainer

### Usage
Users can now select a specific GPU without enabling multi-GPU mode:
1. Keep "Multi-GPU" switch **OFF**
2. Set "Device Indexes" to desired GPU number (e.g., "1" for GPU 1)
3. Train on that single GPU without distributed training overhead

---

## TODO: GUI Performance Optimizations

### 1. Excessive Config Serialization (HIGH PRIORITY)
**Location:** `modules/util/config/TrainConfig.py` line 788-797
**Problem:** `to_settings_dict()` does unnecessary round-trip serialization:
- Converts config to dict
- Creates new default config
- Deserializes back with `from_dict()`
- Serializes again to save
- This is 3x the work needed!

**Current Code:**
```python
def to_settings_dict(self, secrets: bool) -> dict:
    config = TrainConfig.default_values().from_dict(self.to_dict())
    config.concepts = None
    config.samples = None
    config_dict = config.to_dict()
    if not secrets:
        config_dict.pop('secrets',None)
    return config_dict
```

**Proposed Fix:**
```python
def to_settings_dict(self, secrets: bool) -> dict:
    # Don't create a new config and re-parse, just modify the dict directly
    config_dict = self.to_dict()
    config_dict.pop('concepts', None)
    config_dict.pop('samples', None)
    if not secrets:
        config_dict.pop('secrets', None)
    return config_dict
```

**Expected Impact:** 3-5x faster config saves

---

### 2. Tkinter Variable Traces (MEDIUM PRIORITY)
**Location:** `modules/util/ui/UIState.py` lines 188-209
**Problem:** Every UI field has `trace_add("write", ...)` that fires on EVERY keystroke
- Hundreds of config fields = cascade of updates
- No debouncing

**Proposed Fix:** 
- Add debouncing so updates only happen after 500ms of no typing
- Reduce unnecessary trace callbacks

---

### 3. Recursive Config Updates (MEDIUM PRIORITY)
**Location:** `modules/util/ui/UIState.py` line 243 (`__set_vars()`)
**Problem:** Recursively updates all nested config objects even if unchanged

**Proposed Fix:** 
- Add dirty checking to only update changed values
- Cache previous values to avoid unnecessary updates

---

### 4. Synchronous File I/O (LOW PRIORITY)
**Location:** `modules/util/path_util.py` lines 31-34
**Problem:** `write_json_atomic()` writes entire JSON synchronously on main UI thread
- Blocks interface during save

**Current Code:**
```python
def write_json_atomic(path: str, obj: Any):
    with open(path + ".write", "w") as f:
        json.dump(obj, f, indent=4)
    os.replace(path + ".write", path)
```

**Proposed Fix:** 
- Move JSON writing to background thread
- Show progress indicator during save

---

### 5. No Debouncing on Save Operations (MEDIUM PRIORITY)
**Problem:** Saves happen immediately on every change
**Proposed Fix:** 
- Delay saves until 500ms after last change
- Batch multiple rapid changes into single save

---

## Implementation Priority
1. ✅ **DONE**: Fix GPU selection bug
2. **NEXT**: Fix `to_settings_dict()` - Quick win, biggest impact
3. Add debouncing to save operations
4. Add debouncing to trace callbacks
5. Optimize recursive updates
6. Make file writes async

---

## Testing Checklist
- [ ] Measure save time before/after with large configs
- [ ] Test with multiple concepts (10+)
- [ ] Test with rapid typing in text fields
- [ ] Test GPU selection with different device indexes
- [ ] Test backward compatibility with existing configs

