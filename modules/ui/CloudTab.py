
import webbrowser

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class CloudTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState, parent):
        super().__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state
        self.parent = parent
        self.reattach = False

        self.frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=0)
        self.frame.grid_columnconfigure(3, weight=1)
        self.frame.grid_columnconfigure(4, weight=0)
        self.frame.grid_columnconfigure(5, weight=1)

        components.label(self.frame, 0, 0, "Enabled",
                         tooltip="Enable cloud training")
        components.switch(self.frame, 0, 1, self.ui_state, "cloud.enabled")

        components.label(self.frame, 1, 0, "Type",
                         tooltip="Choose LINUX to connect to a linux machine via SSH. Choose RUNPOD for additional functionality such as automatically creating and deleting pods.")
        components.options_kv(self.frame, 1, 1, [
            ("RUNPOD", CloudType.RUNPOD),
            ("LINUX", CloudType.LINUX),
        ], self.ui_state, "cloud.type")

        components.label(self.frame, 2, 0, "File sync method",
                         tooltip="Choose NATIVE_SCP to use scp.exe to transfer files. FABRIC_SFTP uses the Paramiko/Fabric SFTP implementation for file transfers instead.")
        components.options_kv(self.frame, 2, 1, [
            ("NATIVE_SCP", CloudFileSync.NATIVE_SCP),
            ("FABRIC_SFTP", CloudFileSync.FABRIC_SFTP),
        ], self.ui_state, "cloud.file_sync")

        components.label(self.frame, 3, 0, "API key",
                         tooltip="Cloud service API key for RUNPOD. Leave empty for LINUX. This value is stored separately, not saved to your configuration file. ")
        components.entry(self.frame, 3, 1, self.ui_state, "secrets.cloud.api_key")

        components.label(self.frame, 4, 0, "Hostname",
                         tooltip="SSH server hostname or IP. Leave empty if you have a Cloud ID or want to automatically create a new cloud.")
        components.entry(self.frame, 4, 1, self.ui_state, "secrets.cloud.host")

        components.label(self.frame, 5, 0, "Port",
                         tooltip="SSH server port. Leave empty if you have a Cloud ID or want to automatically create a new cloud.")
        components.entry(self.frame, 5, 1, self.ui_state, "secrets.cloud.port")

        components.label(self.frame, 6, 0, "User",
                         tooltip='SSH username. Use "root" for RUNPOD.')
        components.entry(self.frame, 6, 1, self.ui_state, "secrets.cloud.user")

        components.label(self.frame, 7, 0, "SSH Key Email",
                         tooltip='Optional: Search for a public key containing this email/text and use the corresponding private key. Useful if you have multiple SSH keys.')
        components.entry(self.frame, 7, 1, self.ui_state, "secrets.cloud.key_comment")

        components.label(self.frame, 8, 0, "Cloud id",
                         tooltip="RUNPOD Cloud ID. The cloud service must have a public IP and SSH service. Leave empty if you want to automatically create a new RUNPOD cloud, or if you're connecting to another cloud provider via SSH Hostname and Port.")
        components.entry(self.frame, 8, 1, self.ui_state, "secrets.cloud.id")

        components.label(self.frame, 9, 0, "Tensorboard TCP tunnel",
                         tooltip="Instead of starting tensorboard locally, make a TCP tunnel to a tensorboard on the cloud")
        components.switch(self.frame, 9, 1, self.ui_state, "cloud.tensorboard_tunnel")



        components.label(self.frame, 1, 2, "Remote Directory",
                         tooltip="The directory on the cloud where files will be uploaded and downloaded.")
        components.entry(self.frame, 1, 3, self.ui_state, "cloud.remote_dir")
        components.label(self.frame, 2, 2, "OneTrainer Directory",
                         tooltip="The directory for OneTrainer on the cloud.")
        components.entry(self.frame, 2, 3, self.ui_state, "cloud.onetrainer_dir")
        components.label(self.frame, 3, 2, "Huggingface cache Directory",
                         tooltip="Huggingface models are downloaded to this remote directory.")
        components.entry(self.frame, 3, 3, self.ui_state, "cloud.huggingface_cache_dir")
        components.label(self.frame, 4, 2, "Install OneTrainer",
                         tooltip="Automatically install OneTrainer from GitHub if the directory doesn't already exist.")
        components.switch(self.frame, 4, 3, self.ui_state, "cloud.install_onetrainer")
        components.label(self.frame, 5, 2, "Install command",
                         tooltip="The command for installing OneTrainer. Leave the default, unless you want to use a development branch of OneTrainer.")
        components.entry(self.frame, 5, 3, self.ui_state, "cloud.install_cmd")
        components.label(self.frame, 6, 2, "Update OneTrainer",
                         tooltip="Update OneTrainer if it already exists on the cloud.")
        components.switch(self.frame, 6, 3, self.ui_state, "cloud.update_onetrainer")

        components.label(self.frame, 8, 2, "Detach remote trainer",
                         tooltip="Allows the trainer to keep running even if your connection to the cloud is lost.")
        components.switch(self.frame, 8, 3, self.ui_state, "cloud.detach_trainer")
        components.label(self.frame, 9, 2, "Reattach id",
                         tooltip="An id identifying the remotely running trainer. In case you have lost connection or closed OneTrainer, it will try to reattach to this id instead of starting a new remote trainer.")
        reattach_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        reattach_frame.grid(row=9, column=3, padx=0, pady=0, sticky="new")
        reattach_frame.grid_columnconfigure(0, weight=1)
        reattach_frame.grid_columnconfigure(1, weight=1)
        components.entry(reattach_frame, 0, 0, self.ui_state, "cloud.run_id", width=60)
        components.button(reattach_frame, 0, 1, "Reattach now", self.__reattach)

        components.label(self.frame, 11, 2, "Download samples",
                         tooltip="Download samples from the remote workspace directory to your local machine.")
        components.switch(self.frame, 11, 3, self.ui_state, "cloud.download_samples")
        components.label(self.frame, 12, 2, "Download output model",
                         tooltip="Download the final model after training. You can disable this if you plan to use an automatically saved checkpoint instead.")
        components.switch(self.frame, 12, 3, self.ui_state, "cloud.download_output_model")
        components.label(self.frame, 13, 2, "Download saved checkpoints",
                         tooltip="Download the automatically saved training checkpoints from the remote workspace directory to your local machine.")
        components.switch(self.frame, 13, 3, self.ui_state, "cloud.download_saves")
        components.label(self.frame, 14, 2, "Download backups",
                         tooltip="Download backups from the remote workspace directory to your local machine. It's usually not necessary to download them, because as long as the backups are still available on the cloud, the training can be restarted using one of the cloud's backups.")
        components.switch(self.frame, 14, 3, self.ui_state, "cloud.download_backups")
        components.label(self.frame, 15, 2, "Download tensorboard logs",
                         tooltip="Download TensorBoard event logs from the remote workspace directory to your local machine. They can then be viewed locally in TensorBoard. It is recommended to disable \"Sample to TensorBoard\" to reduce the event log size.")
        components.switch(self.frame, 15, 3, self.ui_state, "cloud.download_tensorboard")
        components.label(self.frame, 16, 2, "Delete remote workspace",
                         tooltip="Delete the workspace directory on the cloud after training has finished successfully and data has been downloaded.")
        components.switch(self.frame, 16, 3, self.ui_state, "cloud.delete_workspace")

        components.label(self.frame, 1, 4, "Create cloud via API",
                         tooltip="Automatically creates a new cloud instance if both Host:Port and Cloud ID are empty. Currently supported for RUNPOD.")
        create_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        create_frame.grid(row=1, column=5, padx=0, pady=0, sticky="new")
        create_frame.grid_columnconfigure(0, weight=0)
        create_frame.grid_columnconfigure(1, weight=1)
        components.switch(create_frame, 0, 0, self.ui_state, "cloud.create")
        components.button(create_frame, 0, 1, "Create cloud via website", self.__create_cloud)

        components.label(self.frame, 2, 4, "Cloud name",
                         tooltip="The name of the new cloud instance.")
        components.entry(self.frame, 2, 5, self.ui_state, "cloud.name")
        components.label(self.frame, 3, 4, "Type",
                         tooltip="Select the RunPod cloud type. See RunPod's website for details.")
        components.options_kv(self.frame, 3, 5, [
            ("", ""),
            ("Community", "COMMUNITY"),
            ("Secure", "SECURE"),
        ], self.ui_state, "cloud.sub_type")

        components.label(self.frame, 4, 4, "Multi-GPU",
                         tooltip="Enable multi-GPU training on the cloud instance.")
        components.switch(self.frame, 4, 5, self.ui_state, "cloud.multi_gpu")

        components.label(self.frame, 5, 4, "GPU count",
                         tooltip="Number of GPUs to request for the Runpod instance. Enable Multi-GPU above to use multiple GPUs for training.")
        self.gpu_count_entry = components.entry(
            self.frame, 5, 5, self.ui_state, "cloud.gpu_count",
            tooltip="Number of GPUs to request. Enable Multi-GPU above first to use multiple GPUs for training."
        )

        components.label(self.frame, 6, 4, "Device Indexes",
                         tooltip="A comma-separated list of device indexes (e.g., \"0,1,2,3\"). If empty, all available GPUs are used. Only used when Multi-GPU is enabled.")
        components.entry(self.frame, 6, 5, self.ui_state, "cloud.device_indexes")

        components.label(self.frame, 7, 4, "GPU",
                         tooltip="Select the GPU type. Enter an API key before pressing the button.")

        _,gpu_components=components.options_adv(self.frame, 7, 5, [("")], self.ui_state, "cloud.gpu_type",adv_command=self.__set_gpu_types)
        self.gpu_types_menu=gpu_components['component']
        
        # Store original colors for re-enabling
        try:
            self.gpu_count_entry_normal_fg = self.gpu_count_entry.cget("fg_color")
            self.gpu_count_entry_normal_text = self.gpu_count_entry.cget("text_color")
        except Exception:
            self.gpu_count_entry_normal_fg = None
            self.gpu_count_entry_normal_text = None
        
        # Enable/disable GPU count field based on cloud.multi_gpu setting
        # Also sync gpu_count with cloud.device_indexes when multi-GPU is enabled
        def update_gpu_count_state(*_):
            multi_gpu_enabled = self.ui_state.get_var("cloud.multi_gpu").get()
            gpu_count_var = self.ui_state.get_var("cloud.gpu_count")
            device_indexes_var = self.ui_state.get_var("cloud.device_indexes")
            
            if multi_gpu_enabled:
                self.gpu_count_entry.configure(state="normal")
                # Restore normal colors when enabled
                if self.gpu_count_entry_normal_fg:
                    self.gpu_count_entry.configure(fg_color=self.gpu_count_entry_normal_fg)
                if self.gpu_count_entry_normal_text:
                    self.gpu_count_entry.configure(text_color=self.gpu_count_entry_normal_text)
                
                # Sync gpu_count with device_indexes if device_indexes is specified
                device_indexes_str = device_indexes_var.get() or ""
                device_indexes_str = device_indexes_str.strip()
                if device_indexes_str:
                    # Parse device indexes and set gpu_count to match
                    try:
                        device_list = [d.strip() for d in device_indexes_str.split(",") if d.strip()]
                        if device_list:
                            gpu_count_var.set(str(len(device_list)))
                    except Exception:
                        pass  # If parsing fails, leave gpu_count as-is
                # If device_indexes is empty, gpu_count stays as user set (for "use all GPUs")
            else:
                self.gpu_count_entry.configure(state="disabled")
                # Make it visually grayed out
                self.gpu_count_entry.configure(fg_color="#2b2b2b")
                self.gpu_count_entry.configure(text_color="#7f7f7f")
                # Reset GPU count to 1 if multi-GPU is disabled
                try:
                    current_count = int(gpu_count_var.get() or "1")
                    if current_count > 1:
                        gpu_count_var.set("1")
                except (ValueError, TypeError):
                    gpu_count_var.set("1")
        
        # Set initial state
        update_gpu_count_state()
        # Watch for changes to cloud.multi_gpu and cloud.device_indexes
        # Access the nested UIState first to ensure it's initialized
        cloud_ui_state = self.ui_state.get_var("cloud")
        cloud_ui_state.add_var_trace("multi_gpu", update_gpu_count_state)
        cloud_ui_state.add_var_trace("device_indexes", update_gpu_count_state)

        components.label(self.frame, 8, 4, "Volume size",
                         tooltip="Set the storage volume size in GB. This volume persists only until the cloud is deleted - not a RunPod network volume")
        components.entry(self.frame, 8, 5, self.ui_state, "cloud.volume_size")

        components.label(self.frame, 9, 4, "Min download",
                         tooltip="Set the minimum download speed of the cloud in Mbps.")
        components.entry(self.frame, 9, 5, self.ui_state, "cloud.min_download")

        components.label(self.frame, 10, 4, "Action on finish",
                         tooltip="What to do when training finishes and the data has been fully downloaded: Stop or delete the cloud, or do nothing.")
        components.options_kv(self.frame, 10, 5, [
            ("None", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("Delete", CloudAction.DELETE),
        ], self.ui_state, "cloud.on_finish")

        components.label(self.frame, 11, 4, "Action on error",
                         tooltip="What to do if training stops due to an error: Stop or delete the cloud, or do nothing. Data may be lost.")
        components.options_kv(self.frame, 11, 5, [
            ("None", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("Delete", CloudAction.DELETE),
        ], self.ui_state, "cloud.on_error")

        components.label(self.frame, 12, 4, "Action on detached finish",
                         tooltip="What to do when training finishes, but the client has been detached and cannot download data. Data may be lost.")
        components.options_kv(self.frame, 12, 5, [
            ("None", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("Delete", CloudAction.DELETE),
        ], self.ui_state, "cloud.on_detached_finish")

        components.label(self.frame, 13, 4, "Action on detached error",
                         tooltip="What to if training stops due to an error, but the client has been detached and cannot download data. Data may be lost.")
        components.options_kv(self.frame, 13, 5, [
            ("None", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("Delete", CloudAction.DELETE),
        ], self.ui_state, "cloud.on_detached_error")

        self.frame.pack(fill="both", expand=1)

    def __set_gpu_types(self):
        self.gpu_types_menu.configure(values=[])
        if self.train_config.cloud.type == CloudType.RUNPOD:
            import runpod
            runpod.api_key=self.train_config.secrets.cloud.api_key
            gpus=runpod.get_gpus()
            self.gpu_types_menu.configure(values=[gpu['id'] for gpu in gpus])

    def __reattach(self):
        self.reattach=True
        try:
            self.parent.start_training()
        finally:
            self.reattach=False

    def __create_cloud(self):
        if self.train_config.cloud.type == CloudType.RUNPOD:
            webbrowser.open("https://www.runpod.io/console/deploy?template=1a33vbssq9&type=gpu", new=0, autoraise=False)
