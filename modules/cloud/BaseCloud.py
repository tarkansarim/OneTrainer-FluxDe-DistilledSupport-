import json
import shlex
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.CloudConfig import CloudConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.time_util import get_string_timestamp


class BaseCloud(metaclass=ABCMeta):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.file_sync=None
        self._connection_update_callback: Optional[Callable[[str, str, str], None]] = None


    def setup(self):
        self._connect()

        if (self.config.cloud.install_onetrainer or self.config.cloud.update_onetrainer) and not self.can_reattach():
            self._install_onetrainer(update=self.config.cloud.update_onetrainer)

        if self.config.cloud.tensorboard_tunnel:
            self._make_tensorboard_tunnel()

    def download_output_model(self):
        local=Path(self.config.local_output_model_destination)
        remote=Path(self.config.output_model_destination)
        
        # Check if remote file exists before attempting download
        # The remote path may not exist if the model was saved to a different location
        # (e.g., workspace/save/ instead of models/)
        try:
            self.file_sync.sync_connection.open()
            check_cmd = f'test -f {shlex.quote(remote.as_posix())} && echo "exists" || echo "missing"'
            result = self.file_sync.sync_connection.run(check_cmd, in_stream=False, warn=True, hide='both')
            if "exists" in (result.stdout or ""):
                self.file_sync.sync_down_file(local=local,remote=remote)
            else:
                print(f"Warning: Output model not found at remote path {remote.as_posix()}. "
                      f"It may have been saved to a different location (e.g., workspace/save/).")
        except Exception as e:
            print(f"Warning: Could not check remote output model path: {e}")
            # Try downloading anyway, sync_down_file will handle errors gracefully
        
        # Check embeddings directory
        embeddings_remote = remote.with_suffix(remote.suffix+"_embeddings")
        try:
            check_cmd = f'test -d {shlex.quote(embeddings_remote.as_posix())} && echo "exists" || echo "missing"'
            result = self.file_sync.sync_connection.run(check_cmd, in_stream=False, warn=True, hide='both')
            if "exists" in (result.stdout or ""):
                self.file_sync.sync_down_dir(local=local.with_suffix(local.suffix+"_embeddings"),
                                   remote=embeddings_remote)
        except Exception as e:
            print(f"Warning: Could not check remote embeddings directory: {e}")

    def upload_config(self,commands : TrainCommands=None):
        local_config_path=Path(self.config.local_workspace_dir,f"remote_config-{get_string_timestamp()}.json")
        #no need to upload secrets - hugging face token is transferred via environment variable:
        with local_config_path.open(mode="w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)
        self._upload_config_file(local_config_path)

        if hasattr(self.config,"local_base_model_name"):
            self.file_sync.sync_up(local=Path(self.config.local_base_model_name),remote=Path(self.config.base_model_name))
        if hasattr(self.config.prior,"local_model_name"):
            self.file_sync.sync_up(local=Path(self.config.prior.local_model_name),remote=Path(self.config.prior.model_name))
        if hasattr(self.config,"local_lora_model_name"):
            self.file_sync.sync_up(local=Path(self.config.local_lora_model_name),remote=Path(self.config.lora_model_name))

        if hasattr(self.config.embedding,"local_model_name"):
            self.file_sync.sync_up(local=Path(self.config.embedding.local_model_name),remote=Path(self.config.embedding.model_name))
        for add_embedding in self.config.additional_embeddings:
            if hasattr(add_embedding,"local_model_name"):
                self.file_sync.sync_up(local=Path(add_embedding.local_model_name),remote=Path(add_embedding.model_name))

        for concept in self.config.concepts:
            print(f"uploading concept {concept.name}...")
            if commands and commands.get_stop_command():
                return

            if hasattr(concept,"local_path"):
                self.file_sync.sync_up_dir(
                    local=Path(concept.local_path),
                    remote=Path(concept.path),
                    recursive=concept.include_subdirectories)

            if hasattr(concept.text,"local_prompt_path"):
                self.file_sync.sync_up_file(local=Path(concept.text.local_prompt_path),remote=Path(concept.text.prompt_path))

        # If training should continue from the last backup, upload the latest local backup folder as well
        try:
            if getattr(self.config, "continue_last_backup", False):
                local_backups_dir = Path(self.config.local_workspace_dir, "backup")
                if local_backups_dir.exists() and local_backups_dir.is_dir():
                    # Find latest backup folder by lexicographic (timestamp-first) order, descending
                    backup_dirs = sorted(
                        [p for p in local_backups_dir.iterdir() if p.is_dir()],
                        key=lambda p: p.name,
                        reverse=True,
                    )
                    if len(backup_dirs) > 0:
                        latest_backup = backup_dirs[0]
                        remote_backup = Path(self.config.workspace_dir, "backup", latest_backup.name)

                        should_upload = True
                        try:
                            self.file_sync.sync_connection.open()
                            check_cmd = f'test -d {shlex.quote(remote_backup.as_posix())}'
                            result = self.file_sync.sync_connection.run(
                                check_cmd,
                                in_stream=False,
                                warn=True,
                                hide=True,
                            )
                            if result.exited == 0:
                                should_upload = False
                                print(
                                    f"Skipping backup upload; remote folder {remote_backup} already exists."
                                )
                        except Exception as check_exc:
                            print(
                                f"Warning: Could not verify remote backup path {remote_backup}: {check_exc}"
                            )

                        if should_upload:
                            print(f"uploading latest backup {latest_backup} -> {remote_backup}")
                            self.file_sync.sync_up_dir(
                                local=latest_backup,
                                remote=remote_backup,
                                recursive=True,
                            )
        except Exception:
            # Do not fail the whole upload if backup upload fails; training can still proceed
            pass

    def set_connection_update_callback(self, callback: Optional[Callable[[str, str, str], None]]):
        self._connection_update_callback = callback

    def _notify_connection_update(self):
        if not self._connection_update_callback:
            return

        secrets = self.config.secrets.cloud
        host = secrets.host or ""
        port = secrets.port if secrets.port is not None else ""
        port_str = str(port) if port != "" else ""
        cloud_id = getattr(secrets, "id", "") or ""

        try:
            self._connection_update_callback(host, port_str, cloud_id)
        except Exception:
            pass

    @staticmethod
    def _filter_download(config : CloudConfig,path : Path):
        if 'samples' in path.parts:
            return config.download_samples
        elif 'save' in path.parts:
            return config.download_saves
        elif 'backup' in path.parts:
            return config.download_backups
        elif 'tensorboard' in path.parts:
            return config.download_tensorboard
        else:
            return True


    @abstractmethod
    def run_trainer(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def exec_callback(self,callbacks : TrainCallbacks):
        pass

    @abstractmethod
    def send_commands(self,commands : TrainCommands):
        pass

    @abstractmethod
    def sync_workspace(self):
        pass

    @abstractmethod
    def can_reattach(self):
        pass

    def _create(self):
        raise NotImplementedError("creating clouds not supported for this cloud type")

    def delete(self):
        raise NotImplementedError("deleting this cloud type not supported")

    def stop(self):
        raise NotImplementedError("stopping this cloud type not supported")

    @abstractmethod
    def _install_onetrainer(self, update: bool=False):
        pass

    @abstractmethod
    def _make_tensorboard_tunnel(self):
        raise NotImplementedError("Tensorboard tunnel not supported on this cloud type")

    @abstractmethod
    def _upload_config_file(self,local : Path):
        pass

    @abstractmethod
    def delete_workspace(self):
        pass
