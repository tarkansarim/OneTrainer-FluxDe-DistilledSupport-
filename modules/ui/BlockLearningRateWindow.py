from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components, dialogs
from modules.util.ui.UIState import UIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
import tkinter.messagebox as messagebox


class BlockLearningRateWindow(ctk.CTkToplevel):
    _CUSTOM_PRESET_NAME = "Custom"
    _BASE_PRESET_VALUE = 0.01
    _STRUCTURE_SINGLE_BLOCKS = [9, 10, 11, 12, 13, 14]
    _DETAIL_SINGLE_BLOCKS = [7, 12, 16, 20]

    def __init__(
            self,
            parent,
            train_config: TrainConfig,
            ui_state: UIState,
            num_double_blocks: int = 19,
            num_single_blocks: int = 38,
            *args, **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state
        self.num_double_blocks = num_double_blocks
        self.num_single_blocks = num_single_blocks

        self.sliders: dict[str, ctk.CTkSlider] = {}
        self.value_labels: dict[str, ctk.CTkLabel] = {}

        self._applying_preset = False
        self._ignore_preset_callback = False

        self._sanitize_current_multipliers()
        self.custom_presets = self._load_custom_presets()
        self.default_presets = self._build_default_presets()
        self._protected_presets = {name.lower() for name in self.default_presets}

        self.current_preset_name = self._CUSTOM_PRESET_NAME

        self.protocol("WM_DELETE_WINDOW", self.on_window_close)

        self.title("Block-Wise Learning Rate Multipliers")
        self.geometry("600x700")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.controls_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.controls_frame, text="Presets").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.preset_var = ctk.StringVar(value=self._CUSTOM_PRESET_NAME)
        self.preset_option_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            values=self._get_preset_names(),
            variable=self.preset_var,
            command=self._on_preset_selected,
            width=220,
        )
        self.preset_option_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.save_button = ctk.CTkButton(
            self.controls_frame,
            text="Save Preset",
            command=self._prompt_save_preset,
            width=110,
        )
        self.save_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        self.delete_button = ctk.CTkButton(
            self.controls_frame,
            text="Delete Preset",
            command=self._delete_selected_preset,
            width=110,
        )
        self.delete_button.grid(row=0, column=3, padx=5, pady=5, sticky="e")

        self.frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=0)

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=0)
        button_frame.grid_columnconfigure(2, weight=0)

        components.button(button_frame, 0, 1, "Reset All to 1.0", command=self.reset_all, width=120)
        components.button(button_frame, 0, 2, "OK", command=self.on_window_close, width=80)

        self.create_sliders()
        self._initialize_preset_menu()

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def _sanitize_mapping(self, mapping: dict[str, float]) -> dict[str, float]:
        sanitized: dict[str, float] = {}
        for key, value in mapping.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if key.startswith("double_block_"):
                try:
                    idx = int(key[len("double_block_"):])
                except ValueError:
                    continue
                if 0 <= idx < self.num_double_blocks:
                    sanitized[key] = numeric_value
            elif key.startswith("single_block_"):
                try:
                    idx = int(key[len("single_block_"):])
                except ValueError:
                    continue
                if 0 <= idx < self.num_single_blocks:
                    sanitized[key] = numeric_value
            else:
                sanitized[key] = numeric_value
        return sanitized

    def _sanitize_current_multipliers(self) -> None:
        current = self.train_config.block_learning_rate_multiplier or {}
        sanitized = self._sanitize_mapping(current)
        self.train_config.block_learning_rate_multiplier = sanitized

    def _load_custom_presets(self) -> dict[str, dict[str, float]]:
        stored = self.train_config.block_learning_rate_custom_presets or {}
        normalized: dict[str, dict[str, float]] = {}
        for name, preset in stored.items():
            if not isinstance(preset, dict):
                continue
            normalized_preset = self._sanitize_mapping(preset)
            if normalized_preset:
                normalized[name] = normalized_preset
        return normalized

    @staticmethod
    def _max_index(mapping: dict[str, float], prefix: str) -> int:
        max_idx = -1
        for key in mapping.keys():
            if key.startswith(prefix):
                try:
                    idx = int(key[len(prefix):])
                except ValueError:
                    continue
                max_idx = max(max_idx, idx)
        return max_idx

    def _get_all_block_names(self) -> list[str]:
        block_names = [f"double_block_{i}" for i in range(self.num_double_blocks)]
        block_names.extend(f"single_block_{i}" for i in range(self.num_single_blocks))
        return block_names

    def _make_single_block_preset(self, single_blocks_with_one: list[int]) -> dict[str, float]:
        preset = {name: self._BASE_PRESET_VALUE for name in self._get_all_block_names()}
        for idx in single_blocks_with_one:
            if 0 <= idx < self.num_single_blocks:
                preset[f"single_block_{idx}"] = 1.0
        return preset

    def _build_default_presets(self) -> dict[str, dict[str, float]]:
        presets = {
            "Structure": self._make_single_block_preset(self._STRUCTURE_SINGLE_BLOCKS),
            "Detail": self._make_single_block_preset(self._DETAIL_SINGLE_BLOCKS),
        }
        # Flux Luma Normalize preset: zero-out specified blocks to avoid contrast-shifted training
        luma_zero: dict[str, float] = {}
        for idx in [0, 1, 2, 3, 4, 5, 6]:
            if 0 <= idx < self.num_double_blocks:
                luma_zero[f"double_block_{idx}"] = 0.0
        for idx in [15, 16, 22, 29, 32, 34, 35, 36, 37]:
            if 0 <= idx < self.num_single_blocks:
                luma_zero[f"single_block_{idx}"] = 0.0
        if luma_zero:
            presets["Luma Normalize"] = luma_zero
        return presets

    def _get_preset_names(self) -> list[str]:
        names = [self._CUSTOM_PRESET_NAME]
        names.extend(self.default_presets.keys())
        names.extend(sorted(self.custom_presets.keys(), key=str.lower))
        return names

    def _initialize_preset_menu(self) -> None:
        self._refresh_preset_menu(selected=self._find_matching_preset(), trigger_command=False)

    def _refresh_preset_menu(self, selected: str | None = None, trigger_command: bool = False) -> None:
        values = self._get_preset_names()
        self.preset_option_menu.configure(values=values)
        if not selected or selected not in values:
            selected = self._CUSTOM_PRESET_NAME
        self._set_preset_selection(selected, trigger_command=trigger_command)

    def _set_preset_selection(self, preset_name: str, trigger_command: bool = True) -> None:
        if preset_name not in self._get_preset_names():
            preset_name = self._CUSTOM_PRESET_NAME
        if not trigger_command:
            self._ignore_preset_callback = True
        try:
            self.preset_var.set(preset_name)
            self.preset_option_menu.set(preset_name)
        finally:
            if not trigger_command:
                self._ignore_preset_callback = False
        self.current_preset_name = preset_name
        self._update_delete_button_state()

    def _update_delete_button_state(self) -> None:
        state = "normal" if self.current_preset_name in self.custom_presets else "disabled"
        self.delete_button.configure(state=state)

    def _find_matching_preset(self) -> str:
        current = self._capture_current_block_values()
        for name, preset in self.default_presets.items():
            if self._compare_presets(current, preset):
                return name
        for name, preset in self.custom_presets.items():
            if self._compare_presets(current, preset):
                return name
        return self._CUSTOM_PRESET_NAME

    @staticmethod
    def _compare_presets(a: dict[str, float], b: dict[str, float], tolerance: float = 1e-4) -> bool:
        keys = set(a.keys()) | set(b.keys())
        for key in keys:
            if abs(a.get(key, 1.0) - b.get(key, 1.0)) > tolerance:
                return False
        return True

    def _capture_current_block_values(self) -> dict[str, float]:
        return {block_name: float(self.sliders[block_name].get()) for block_name in self.sliders}

    def create_sliders(self):
        """Create sliders for all double and single blocks"""
        current_row = 0

        block_multipliers = self.train_config.block_learning_rate_multiplier or {}

        components.label(
            self.frame,
            current_row,
            0,
            "Double Blocks",
            tooltip="Learning rate multipliers for double transformer blocks",
        )
        current_row += 1

        for i in range(self.num_double_blocks):
            block_name = f"double_block_{i}"
            current_multiplier = block_multipliers.get(block_name, 1.0)
            self.create_block_slider(current_row, block_name, f"Block {i}", current_multiplier)
            current_row += 1

        current_row += 1

        components.label(
            self.frame,
            current_row,
            0,
            "Single Blocks",
            tooltip="Learning rate multipliers for single transformer blocks",
        )
        current_row += 1

        for i in range(self.num_single_blocks):
            block_name = f"single_block_{i}"
            current_multiplier = block_multipliers.get(block_name, 1.0)
            self.create_block_slider(current_row, block_name, f"Block {i}", current_multiplier)
            current_row += 1

    def create_block_slider(self, row: int, block_name: str, display_name: str, initial_value: float):
        label = ctk.CTkLabel(self.frame, text=display_name, width=80)
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        slider = ctk.CTkSlider(
            self.frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=lambda value, name=block_name: self.on_slider_change(name, value),
        )
        slider.set(initial_value)
        slider.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        value_label = ctk.CTkLabel(self.frame, text=f"{initial_value:.2f}", width=50)
        value_label.grid(row=row, column=2, padx=5, pady=5, sticky="e")

        self.sliders[block_name] = slider
        self.value_labels[block_name] = value_label

    def on_slider_change(self, block_name: str, value: float):
        self.value_labels[block_name].configure(text=f"{value:.2f}")

        if self.train_config.block_learning_rate_multiplier is None:
            self.train_config.block_learning_rate_multiplier = {}

        self.train_config.block_learning_rate_multiplier[block_name] = float(value)

        if not self._applying_preset:
            self._mark_custom_state()

    def _mark_custom_state(self):
        if self.current_preset_name != self._CUSTOM_PRESET_NAME:
            self._set_preset_selection(self._CUSTOM_PRESET_NAME, trigger_command=False)

    def reset_all(self):
        if self.train_config.block_learning_rate_multiplier is None:
            self.train_config.block_learning_rate_multiplier = {}

        for block_name, slider in self.sliders.items():
            slider.set(1.0)
            self.value_labels[block_name].configure(text="1.00")
            self.train_config.block_learning_rate_multiplier[block_name] = 1.0

        self._mark_custom_state()

    def _prompt_save_preset(self):
        dialogs.StringInputDialog(
            self,
            "Save Preset",
            "Preset name",
            self._save_preset,
            validate_callback=self._validate_new_preset_name,
        )

    def _validate_new_preset_name(self, name: str) -> bool:
        trimmed = name.strip()
        if not trimmed:
            messagebox.showwarning("Invalid Name", "Preset name cannot be empty.")
            return False

        lower_name = trimmed.lower()
        if lower_name in self._protected_presets:
            messagebox.showwarning("Preset Exists", f"'{trimmed}' is a built-in preset and cannot be overwritten.")
            return False

        if lower_name in {existing.lower() for existing in self.custom_presets.keys()}:
            messagebox.showwarning("Preset Exists", f"A preset named '{trimmed}' already exists.")
            return False

        return True

    def _save_preset(self, name: str):
        trimmed = name.strip()
        if not trimmed:
            return

        self.custom_presets[trimmed] = self._capture_current_block_values()
        self.train_config.block_learning_rate_custom_presets = {
            preset_name: preset.copy() for preset_name, preset in self.custom_presets.items()
        }
        self._refresh_preset_menu(selected=trimmed, trigger_command=False)

    def _delete_selected_preset(self):
        selected = self.current_preset_name
        if selected not in self.custom_presets:
            return

        if not messagebox.askyesno("Delete Preset", f"Delete custom preset '{selected}'?"):
            return

        self.custom_presets.pop(selected, None)
        self.train_config.block_learning_rate_custom_presets = {
            preset_name: preset.copy() for preset_name, preset in self.custom_presets.items()
        }
        self._refresh_preset_menu(selected=self._CUSTOM_PRESET_NAME, trigger_command=False)

    def _on_preset_selected(self, selected: str):
        if self._ignore_preset_callback:
            return

        if not selected:
            selected = self._CUSTOM_PRESET_NAME

        if selected == self._CUSTOM_PRESET_NAME:
            self._set_preset_selection(selected, trigger_command=False)
            return

        preset_map = self.default_presets.get(selected) or self.custom_presets.get(selected)
        if preset_map is None:
            self._set_preset_selection(self._CUSTOM_PRESET_NAME, trigger_command=False)
            return

        self._apply_preset(selected, preset_map)

    def _apply_preset(self, name: str, preset_map: dict[str, float]):
        existing = self.train_config.block_learning_rate_multiplier or {}
        extras = {k: v for k, v in existing.items() if k not in preset_map}
        new_map = {**extras, **{k: float(v) for k, v in preset_map.items()}}
        self.train_config.block_learning_rate_multiplier = new_map

        self._applying_preset = True
        try:
            for block_name, value in preset_map.items():
                slider = self.sliders.get(block_name)
                if slider is None:
                    continue
                slider.set(value)
                self.value_labels[block_name].configure(text=f"{value:.2f}")
        finally:
            self._applying_preset = False

        self._set_preset_selection(name, trigger_command=False)

    def on_window_close(self):
        self.train_config.block_learning_rate_multiplier = self._capture_current_block_values()
        self.train_config.block_learning_rate_custom_presets = {
            preset_name: preset.copy() for preset_name, preset in self.custom_presets.items()
        }
        self.grab_release()
        self.destroy()
