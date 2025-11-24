from collections.abc import Callable

import customtkinter as ctk


class StopTrainingDialog(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            callback: Callable[[bool], None],  # True = terminate immediately, False = wait for backup
            *args, **kwargs
    ):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.callback = callback
        self.result = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.title("Stop Training")
        self.geometry("450x150")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.question_label = ctk.CTkLabel(
            self, 
            text="Do you want to wait for backup the current state or terminate immediately?",
            wraplength=400
        )
        self.question_label.grid(row=0, column=0, columnspan=2, sticky="we", padx=20, pady=20)

        self.wait_button = ctk.CTkButton(
            self, 
            text="Wait for Backup", 
            command=lambda: self._choose(False),
            width=150
        )
        self.wait_button.grid(row=1, column=0, sticky="we", padx=10, pady=10)

        self.terminate_button = ctk.CTkButton(
            self, 
            text="Terminate Immediately", 
            command=lambda: self._choose(True),
            width=150,
            fg_color="#d32f2f",
            hover_color="#b71c1c"
        )
        self.terminate_button.grid(row=1, column=1, sticky="we", padx=10, pady=10)

    def _choose(self, terminate_immediately: bool):
        self.result = terminate_immediately
        self.callback(terminate_immediately)
        self.destroy()


class StringInputDialog(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            title: str,
            question: str,
            callback: Callable[[str], None],
            default_value: str = None,
            validate_callback: Callable[[str], bool] = None,
            *args, **kwargs
    ):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        self.callback = callback
        self.validate_callback = validate_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.title(title)
        self.geometry("300x120")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.question_label = ctk.CTkLabel(self, text=question)
        self.question_label.grid(row=0, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        self.entry = ctk.CTkEntry(self, width=150)
        self.entry.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=5)

        self.ok_button = ctk.CTkButton(self, width=30, text="ok", command=self.ok)
        self.ok_button.grid(row=2, column=0, sticky="we", padx=10, pady=5)

        self.ok_button = ctk.CTkButton(self, width=30, text="cancel", command=self.cancel)
        self.ok_button.grid(row=2, column=1, sticky="we", padx=10, pady=5)

        if default_value is not None:
            self.entry.insert(0, default_value)

    def ok(self):
        if self.validate_callback is None or self.validate_callback(self.entry.get()):
            self.callback(self.entry.get())
            self.destroy()

    def cancel(self):
        self.destroy()
