import os
import threading
import time
import traceback
from contextlib import suppress
from pathlib import Path, PurePosixPath

from modules.cloud.LinuxCloud import LinuxCloud
from modules.cloud.RunpodCloud import RunpodCloud
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudType import CloudType


class CloudTrainer(BaseTrainer):

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands, reattach: bool=False):
        super().__init__(config, callbacks, commands)
        self.error_caught=False
        self.callback_thread=None
        self.sync_thread=None
        self.stop_event=None
        self.cloud=None
        self.reattach=reattach
        self.remote_config=CloudTrainer.__make_remote_config(config)

        tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        if config.tensorboard and not config.cloud.tensorboard_tunnel and not config.tensorboard_always_on:
            super()._start_tensorboard()

        match config.cloud.type:
            case CloudType.RUNPOD:
                self.cloud=RunpodCloud(self.remote_config)
            case CloudType.LINUX:
                self.cloud=LinuxCloud(self.remote_config)

        self.cloud.set_connection_update_callback(self.__handle_cloud_connection_update)

    def start(self):
        try:
            self.callbacks.on_update_status("setting up cloud")
            self.cloud.setup()

            if self.reattach:
                if not self.cloud.can_reattach():
                    raise ValueError(f"There is no detached trainer with run id {self.config.cloud.run_id} on this cloud")
            else:
                if self.cloud.can_reattach():
                    raise ValueError(f"a detached trainer with id {self.config.cloud.run_id} is still running. Use \"Reattach now\" to reattach to this trainer!")
                self.callbacks.on_update_status("uploading config")
                self.cloud.upload_config(self.commands)
                
                # Warn if multi-GPU is enabled on cloud
                if self.remote_config.multi_gpu:
                    print("Info: Multi-GPU training is enabled on cloud. Sampling will run only on master rank for consistency.")
        except:
            self.error_caught=True
            raise

        def on_command(commands : TrainCommands):
            backup_on_command = commands.get_and_reset_on_command() #don't pickle a Callable
            self.cloud.send_commands(commands)
            commands.set_on_command(backup_on_command)
        self.commands.set_on_command(on_command)

        self.stop_event=threading.Event()

        def callback():
            while not self.stop_event.is_set():
                try:
                    self.cloud.exec_callback(self.callbacks)
                except Exception:
                    traceback.print_exc()
                    self.callbacks.on_update_status("error: check the console for more information")
                time.sleep(1)

        self.callback_thread = threading.Thread(target=callback)
        self.callback_thread.start()

        def sync():
            while not self.stop_event.is_set():
                try:
                    self.cloud.sync_workspace()
                except Exception:
                    traceback.print_exc()
                    self.callbacks.on_update_status("error: check the console for more information")
                time.sleep(5)

        self.sync_thread = threading.Thread(target=sync)
        self.sync_thread.start()

        if self.config.continue_last_backup:
            print('info: latest backup will be uploaded before starting training (if found).')

    def __handle_cloud_connection_update(self, host: str, port: str, cloud_id: str):
        secrets = self.config.secrets.cloud
        if host is not None:
            secrets.host = str(host)
        if port is not None and port != "":
            secrets.port = str(port)
        if cloud_id is not None:
            secrets.id = str(cloud_id)

        if self.callbacks:
            self.callbacks.on_update_cloud_connection(str(host or ""), str(port or ""), str(cloud_id or ""))

    def train(self):
        try:
            if self.commands.get_stop_command():
                return

            self.callbacks.on_update_status("starting trainer on cloud")
            self.cloud.run_trainer()

            if self.config.cloud.download_output_model:
                self.callbacks.on_update_status("downloading output model")
                self.cloud.download_output_model()
        except Exception:
            self.error_caught=True
            raise
        finally:
            self.stop_event.set()
            self.callback_thread.join()
            self.callbacks.on_update_status("waiting for downloads")
            self.sync_thread.join()
            self.cloud.sync_workspace()

    def end(self):
        try:
            if self.config.tensorboard and not self.config.cloud.tensorboard_tunnel and not self.config.tensorboard_always_on:
                super()._stop_tensorboard()

            if self.config.cloud.delete_workspace and not self.error_caught and not self.commands.get_stop_command():
                self.callbacks.on_update_status("Deleting remote workspace")
                self.cloud.delete_workspace()

            self.cloud.close()
        except Exception:
            self.error_caught=True
            raise
        finally:
            if self.error_caught:
                action=self.config.cloud.on_error
            elif self.commands.get_stop_command():
                action=CloudAction.NONE
            else:
                action=self.config.cloud.on_finish

            with suppress(Exception): #can fail if the cloud was not successfully created
                if action == CloudAction.DELETE:
                    self.cloud.delete()
                elif action == CloudAction.STOP:
                    self.cloud.stop()

            del self.cloud

    @staticmethod
    def __make_remote_config(local : TrainConfig):
        local.normalize_local_paths(allow_if_cloud_enabled=True)
        remote = TrainConfig.default_values().from_dict(local.to_pack_dict(secrets=True))
        #share cloud config, so UI can be updated to IP, port, cloudid:
        remote.cloud = local.cloud
        remote.secrets.cloud = local.secrets.cloud
        # Override multi_gpu and device_indexes with cloud-specific settings
        # Cloud tab has its own Multi-GPU settings independent of General tab
        remote.multi_gpu = local.cloud.multi_gpu
        remote.device_indexes = local.cloud.device_indexes

        def adjust(config, attribute: str, if_exists: bool=False):
            path=getattr(config,attribute)
            path = CloudTrainer.__normalize_local_path(path, remote.cloud.remote_dir)
            if path.startswith("cloud:"):
                setattr(config,attribute,path.replace("cloud:","",1))
            elif path != "" and (not if_exists or Path(path).exists()):
                local_path = Path(path).expanduser()
                try:
                    resolved_local = local_path.resolve()
                except Exception:
                    resolved_local = local_path.absolute()

                setattr(config,"local_"+attribute,str(resolved_local))
                path=CloudTrainer.__adjust_path(str(local_path),remote.cloud.remote_dir)
                setattr(config,attribute,path)

        adjust(remote,"debug_dir")
        adjust(remote,"workspace_dir")
        adjust(remote,"cache_dir")
        adjust(remote,"base_model_name", if_exists=True)
        adjust(remote.prior,"model_name", if_exists=True)
        adjust(remote,"output_model_destination")
        adjust(remote,"lora_model_name")

        adjust(remote.embedding,"model_name")
        for add_embedding in remote.additional_embeddings:
            adjust(add_embedding,"model_name")

        remote.concept_file_name=""
        remote.concepts = [concept for concept in remote.concepts if concept.enabled]

        for concept in remote.concepts:
            adjust(concept,"path", if_exists=True)
            adjust(concept.text,"prompt_path")

        if remote.train_device == "cpu":
            #if there is no local GPU, "cpu" is the default, but not correct for cloud training
            print("warning: replacing Train Device cpu with cuda")
            remote.train_device = "cuda"

        return remote

    @staticmethod
    def __adjust_path(pathstr : str,remote_dir : str):
        if len(pathstr.strip()) > 0:
            path=Path(pathstr)
            if path.is_absolute():
                path=path.relative_to(path.anchor)  #remove windows drive name C:
            return (Path(remote_dir,"remote") / path).as_posix()
        else:
            return ""

    @staticmethod
    def __normalize_local_path(pathstr: str, remote_dir: str) -> str:
        if not pathstr:
            return pathstr

        normalized = pathstr
        previous = None
        while normalized and normalized != previous:
            previous = normalized
            normalized = CloudTrainer.__remote_to_local_once(normalized, remote_dir)

        return normalized

    @staticmethod
    def __remote_to_local_once(pathstr: str, remote_dir: str) -> str:
        if not pathstr:
            return pathstr

        path_posix = pathstr.replace("\\", "/")
        remote_prefixes = set()

        if remote_dir:
            remote_dir_posix = remote_dir.replace("\\", "/").rstrip("/")
            if remote_dir_posix:
                remote_prefixes.add(f"{remote_dir_posix}/remote/")
                remote_prefixes.add(f"{remote_dir_posix.lstrip('/')}/remote/")

        for prefix in remote_prefixes:
            if path_posix.startswith(prefix):
                remainder = path_posix[len(prefix):]
                return CloudTrainer.__posix_to_os_path(remainder)

        return pathstr

    @staticmethod
    def __posix_to_os_path(pathstr: str) -> str:
        if not pathstr:
            return ""
        pure = PurePosixPath(pathstr)
        return str(Path(*pure.parts))
