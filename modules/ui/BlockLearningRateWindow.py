from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class BlockLearningRateWindow(ctk.CTkToplevel):
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
        
        # Store slider widgets for easy access
        self.sliders = {}
        self.value_labels = {}
        
        self.protocol("WM_DELETE_WINDOW", self.on_window_close)

        self.title("Block-Wise Learning Rate Multipliers")
        self.geometry("600x700")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # Scrollable frame for all the sliders
        self.frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=0)

        # Button frame at bottom
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=0)
        button_frame.grid_columnconfigure(2, weight=0)

        components.button(button_frame, 0, 1, "Reset All to 1.0", command=self.reset_all, width=120)
        components.button(button_frame, 0, 2, "OK", command=self.on_window_close, width=80)

        self.create_sliders()

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def create_sliders(self):
        """Create sliders for all double and single blocks"""
        current_row = 0
        
        # Get current multipliers from config
        block_multipliers = self.train_config.block_learning_rate_multiplier or {}
        
        # Double Blocks Section
        components.label(self.frame, current_row, 0, "Double Blocks", 
                        tooltip="Learning rate multipliers for double transformer blocks")
        current_row += 1
        
        for i in range(self.num_double_blocks):
            block_name = f"double_block_{i}"
            current_multiplier = block_multipliers.get(block_name, 1.0)
            self.create_block_slider(current_row, block_name, f"Block {i}", current_multiplier)
            current_row += 1
        
        # Spacer
        current_row += 1
        
        # Single Blocks Section
        components.label(self.frame, current_row, 0, "Single Blocks",
                        tooltip="Learning rate multipliers for single transformer blocks")
        current_row += 1
        
        for i in range(self.num_single_blocks):
            block_name = f"single_block_{i}"
            current_multiplier = block_multipliers.get(block_name, 1.0)
            self.create_block_slider(current_row, block_name, f"Block {i}", current_multiplier)
            current_row += 1

    def create_block_slider(self, row: int, block_name: str, display_name: str, initial_value: float):
        """Create a slider for a single block"""
        # Label
        label = ctk.CTkLabel(self.frame, text=display_name, width=80)
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        
        # Slider
        slider = ctk.CTkSlider(
            self.frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=lambda value, name=block_name: self.on_slider_change(name, value)
        )
        slider.set(initial_value)
        slider.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        
        # Value label
        value_label = ctk.CTkLabel(self.frame, text=f"{initial_value:.2f}", width=50)
        value_label.grid(row=row, column=2, padx=5, pady=5, sticky="e")
        
        self.sliders[block_name] = slider
        self.value_labels[block_name] = value_label

    def on_slider_change(self, block_name: str, value: float):
        """Handle slider value change"""
        # Update the value label
        self.value_labels[block_name].configure(text=f"{value:.2f}")
        
        # Update the config
        if self.train_config.block_learning_rate_multiplier is None:
            self.train_config.block_learning_rate_multiplier = {}
        
        self.train_config.block_learning_rate_multiplier[block_name] = float(value)

    def reset_all(self):
        """Reset all sliders to 1.0"""
        for block_name, slider in self.sliders.items():
            slider.set(1.0)
            self.value_labels[block_name].configure(text="1.00")
            
            if self.train_config.block_learning_rate_multiplier is None:
                self.train_config.block_learning_rate_multiplier = {}
            
            self.train_config.block_learning_rate_multiplier[block_name] = 1.0

    def on_window_close(self):
        """Handle window close event"""
        self.grab_release()
        self.destroy()


