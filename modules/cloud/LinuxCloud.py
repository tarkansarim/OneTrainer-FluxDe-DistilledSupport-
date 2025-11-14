import pickle
import re
import shlex
import threading
import time
from pathlib import Path

from modules.cloud.BaseCloud import BaseCloud
from modules.cloud.FabricFileSync import FabricFileSync
from modules.cloud.NativeSCPFileSync import NativeSCPFileSync
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.time_util import get_string_timestamp

import fabric


class LinuxCloud(BaseCloud):
    def __init__(self, config: TrainConfig):
        super().__init__(config)
        self.connection=None
        self.callback_connection=None
        self.command_connection=None
        self.tensorboard_tunnel_stop=None
        self._cuda_thresholds = [
            # (min_driver, index_url, torch_version, tv_version)
            ((560, 28, 0), "https://download.pytorch.org/whl/cu128", "2.7.1+cu128", "0.22.1+cu128"),
            ((550, 54, 14), "https://download.pytorch.org/whl/cu124", "2.6.0+cu124", "0.21.0+cu124"),
            ((530, 30, 2), "https://download.pytorch.org/whl/cu121", "2.4.1+cu121", "0.19.1+cu121"),
            ((520, 61, 5), "https://download.pytorch.org/whl/cu118", "2.3.1+cu118", "0.18.1+cu118"),
        ]
        # Initialize standard remote paths
        self._ensure_remote_paths()

    @staticmethod
    def _parse_driver_version(ver: str) -> tuple[int, int, int]:
        parts = [p for p in ver.strip().split(".") if p.isdigit()]
        nums = [int(p) for p in parts[:3]]
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums)  # type: ignore[return-value]

    def _select_torch_wheels_for_driver(self, driver_version: str) -> tuple[str, str, str]:
        drv = self._parse_driver_version(driver_version)
        for min_drv, index_url, torch_ver, tv_ver in self._cuda_thresholds:
            if drv >= min_drv:
                return index_url, torch_ver, tv_ver
        # Extremely old driver: pick the oldest known compatible
        _, index_url, torch_ver, tv_ver = self._cuda_thresholds[-1]
        return index_url, torch_ver, tv_ver

    def _ensure_remote_paths(self):
        """
        Ensure that command/callback/config/pid/log/exit paths are initialized.
        Derive a consistent 'name' from existing config_file if present,
        otherwise from run_id (detached) or a timestamp (attached runs).
        """
        remote_dir = self.config.cloud.remote_dir
        name = None
        try:
            if getattr(self, "config_file", None):
                name = Path(self.config_file).stem
        except Exception:
            name = None
        if not name:
            try:
                if self.config.cloud.detach_trainer and self.config.cloud.run_id:
                    name = self.config.cloud.run_id
                else:
                    name = get_string_timestamp()
            except Exception:
                name = "job1"
        # Set all paths consistently
        self.callback_file = f'{remote_dir}/{name}.callback'
        self.command_pipe = f'{remote_dir}/{name}.command'
        self.config_file = f'{remote_dir}/{name}.json'
        self.exit_status_file = f'{remote_dir}/{name}.exit'
        self.log_file = f'{remote_dir}/{name}.log'
        self.pid_file = f'{remote_dir}/{name}.pid'

        name=config.cloud.run_id if config.cloud.detach_trainer else get_string_timestamp()
        self.callback_file=f'{config.cloud.remote_dir}/{name}.callback'
        self.command_pipe=f'{config.cloud.remote_dir}/{name}.command'
        self.config_file=f'{config.cloud.remote_dir}/{name}.json'
        self.exit_status_file=f'{config.cloud.remote_dir}/{name}.exit'
        self.log_file=f'{config.cloud.remote_dir}/{name}.log'
        self.pid_file=f'{config.cloud.remote_dir}/{name}.pid'

    def _connect(self):
        if self.connection:
            return

        config=self.config.cloud
        secrets=self.config.secrets.cloud

        if secrets.host == '' or secrets.port == '':
            raise ValueError('Host and port required for SSH connection')

        try:
            self.connection=fabric.Connection(host=secrets.host,port=secrets.port,user=secrets.user)
            self.connection.open()
            self.connection.transport.set_keepalive(30)

            self.callback_connection=fabric.Connection(host=secrets.host,port=secrets.port,user=secrets.user)

            self.command_connection=fabric.Connection(host=secrets.host,port=secrets.port,user=secrets.user)
            #the command connection isn't used for long periods of time; prevent remote from closing it:
            self.command_connection.open()
            self.command_connection.transport.set_keepalive(30)

            match config.file_sync:
                case CloudFileSync.NATIVE_SCP:
                    self.file_sync=NativeSCPFileSync(config,secrets)
                case CloudFileSync.FABRIC_SFTP:
                    self.file_sync=FabricFileSync(config,secrets)

            self._notify_connection_update()

        except Exception:
            if self.connection:
                self.connection.close()
                self.connection=None
            if self.command_connection:
                self.command_connection.close()
            raise


    def setup(self):
        super().setup()
        # Ensure remote path attributes exist
        self._ensure_remote_paths()
        self.connection.run(f'mkfifo {shlex.quote(self.command_pipe)}',warn=True,hide=True,in_stream=False)

    def _verify_mgds_import(self, onetrainer_dir: str) -> bool:
        """
        Verify that mgds can be imported. If not, attempt to reinstall it.
        
        Returns:
            True if mgds is importable, False otherwise
        """
        venv_python = f"{onetrainer_dir}/venv/bin/python"
        venv_pip = f"{onetrainer_dir}/venv/bin/pip"
        
        # Test if mgds can be imported
        test_cmd = f"cd {shlex.quote(onetrainer_dir)} && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu && {venv_python} -c \"import mgds.MGDS; print('mgds import successful')\" 2>&1"
        result = self.connection.run(test_cmd, in_stream=False, warn=True, hide='both')
        
        if result.exited == 0:
            return True
        
        # Capture error output for debugging
        error_output = (result.stdout or "") + (result.stderr or "")
        print(f"mgds import failed: {error_output}")
        
        # Check if editable install files exist
        check_pth_cmd = f"cd {shlex.quote(onetrainer_dir)} && ls -la venv/lib/python*/site-packages/*.pth 2>/dev/null | grep mgds || echo 'no mgds pth found'"
        pth_result = self.connection.run(check_pth_cmd, in_stream=False, warn=True, hide='both')
        print(f"mgds .pth check: {pth_result.stdout}")
        
        # Check if source directory exists
        check_src_cmd = f"test -d {onetrainer_dir}/venv/src/mgds && echo 'mgds src exists' || echo 'mgds src missing'"
        src_result = self.connection.run(check_src_cmd, in_stream=False, warn=True, hide='both')
        print(f"mgds source: {src_result.stdout}")
        
        # Import failed, attempt to reinstall mgds
        print("Warning: mgds not importable, attempting reinstall...")
        # Remove the existing git clone directory to force a fresh checkout
        # Uninstall first, then remove the source directory, then reinstall
        # Use the same commit as original OneTrainer (50a2394)
        reinstall_cmd = f"cd {shlex.quote(onetrainer_dir)} && {venv_pip} uninstall -y mgds && rm -rf venv/src/mgds && {venv_pip} install --upgrade --force-reinstall --no-cache-dir -e git+https://github.com/Nerogar/mgds.git@50a2394#egg=mgds"
        self.connection.run(reinstall_cmd, in_stream=False, warn=True)
        
        # Verify import again after reinstall
        result = self.connection.run(test_cmd, in_stream=False, warn=True, hide='both')
        if result.exited == 0:
            print("mgds reinstall successful")
            return True
        else:
            error_output = (result.stdout or "") + (result.stderr or "")
            print(f"Warning: mgds reinstall failed or still not importable: {error_output}")
            return False

    def _install_onetrainer(self, update: bool=False):
        config=self.config.cloud
        parent=Path(config.onetrainer_dir).parent.as_posix()
        
        # Always target our fork; ignore any saved install_cmd repo to avoid pulling the original
        default_repo_path = f"{parent}/OneTrainer-Plus"
        
        # Clone or update the repo if directory doesn't exist
        self.connection.run(f'test -e {shlex.quote(config.onetrainer_dir)} \
                              || (mkdir -p {shlex.quote(parent)} \
                                  && cd {shlex.quote(parent)} \
                                  && git clone https://github.com/tarkansarim/OneTrainer-Plus OneTrainer-Plus)',in_stream=False)
        
        # Find the actual directory that was created (could be repo name or install_cmd target)
        # NEVER rename or move directories - use whatever git creates
        actual_repo_dir = None
        
        # Check if expected directory exists
        check_expected_cmd = f'test -d {shlex.quote(config.onetrainer_dir)} && echo "exists" || echo "missing"'
        result_expected = self.connection.run(check_expected_cmd, in_stream=False, warn=True, hide='both')
        if "exists" in (result_expected.stdout or ""):
            actual_repo_dir = config.onetrainer_dir
        
        # If expected doesn't exist, check for repo-named directory
        if not actual_repo_dir and default_repo_path:
            check_default_cmd = f'test -d {shlex.quote(default_repo_path)} && echo "exists" || echo "missing"'
            result_default = self.connection.run(check_default_cmd, in_stream=False, warn=True, hide='both')
            if "exists" in (result_default.stdout or ""):
                actual_repo_dir = default_repo_path
        
        if not actual_repo_dir:
            raise RuntimeError(f"Repository directory not found at {config.onetrainer_dir} or {default_repo_path} after clone. Install command: {config.install_cmd}")
        
        # Update config to use the actual directory that exists (NEVER rename the directory)
        config.onetrainer_dir = actual_repo_dir
        print(f"Using repository directory: {actual_repo_dir}")
        
        # CRITICAL: Always verify and fix git remote
        # Force expected repo to our fork, regardless of any saved install command
        expected_repo_url = "https://github.com/tarkansarim/OneTrainer-Plus.git"
        
        cmd_env_check = f"export PATH=$PATH:/usr/local/cuda/bin:/venv/main/bin && cd {shlex.quote(actual_repo_dir)}"
        verify_remote_cmd = f"{cmd_env_check} && (git remote get-url origin 2>/dev/null || echo '')"
        result_remote = self.connection.run(verify_remote_cmd, in_stream=False, warn=True, hide='both')
        current_remote = (result_remote.stdout or "").strip()
        
        # Normalize for comparison
        current_remote_normalized = current_remote.rstrip('.git')
        expected_repo_url_normalized = expected_repo_url.rstrip('.git')
        
        # If remote doesn't match fork URL, set it correctly
        if current_remote and expected_repo_url_normalized not in current_remote_normalized:
            print(f"Warning: Repository remote points to {current_remote}, not the fork. Fixing...")
            fix_remote_cmd = f"{cmd_env_check} && git remote set-url origin {shlex.quote(expected_repo_url)}"
            self.connection.run(fix_remote_cmd, in_stream=False)
            print(f"Set git remote to: {expected_repo_url}")

        # Always pull latest code so runtime fixes (e.g., ZLUDA guard, CUDA preflight) are present,
        # even when recreating the venv (fresh install path).
        pull_latest_cmd = f"{cmd_env_check} && git fetch --all --prune && git reset --hard origin/master"
        self.connection.run(pull_latest_cmd, in_stream=False, warn=True)

        result=self.connection.run(f"test -d {shlex.quote(config.onetrainer_dir)}/venv",warn=True,in_stream=False)

        #many docker images, including the default ones on RunPod and vast.ai, only set up $PATH correctly
        #for interactive shells. On RunPod, cuda is missing from $PATH; on vast.ai, python is missing.
        #We cannot pretend to be interactive either, because then vast.ai starts a tmux screen.
        #Add these paths manually:
        # CRITICAL: DO NOT set OT_LAZY_UPDATES=true before install.sh/update.sh
        # This would cause prepare_runtime_environment to skip dependency installation
        # if the git hash hasn't changed, which breaks mgds editable install
        base_cmd_env = f"export PATH=$PATH:/usr/local/cuda/bin:/venv/main/bin \
                         && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu \
                         && cd {shlex.quote(config.onetrainer_dir)}"

        if result.exited == 0:
            if update:
                # Always enforce our fork URL on updates
                expected_repo_url = "https://github.com/tarkansarim/OneTrainer-Plus.git"
                
                # Check current origin remote URL
                verify_remote_cmd = f"{base_cmd_env} && (git remote get-url origin 2>/dev/null || echo '')"
                result = self.connection.run(verify_remote_cmd, in_stream=False, warn=True, hide='both')
                current_remote = (result.stdout or "").strip()
                
                # Normalize for comparison
                current_remote_normalized = current_remote.rstrip('.git')
                expected_repo_url_normalized = expected_repo_url.rstrip('.git')
                
                # If remote doesn't match fork URL, set it correctly
                if expected_repo_url_normalized not in current_remote_normalized:
                    if current_remote:
                        print(f"Warning: Repository remote points to {current_remote}, not the fork. Fixing...")
                    fix_remote_cmd = f"{base_cmd_env} && git remote set-url origin {shlex.quote(expected_repo_url)}"
                    self.connection.run(fix_remote_cmd, in_stream=False)
                    print(f"Set git remote to: {expected_repo_url}")
                
                # Update repo and ensure dependencies are installed
                # CRITICAL: Unset OT_LAZY_UPDATES to force dependency installation
                self.connection.run(base_cmd_env + " && unset OT_LAZY_UPDATES && ./update.sh", in_stream=False)
        else:
            # Fresh install: unset OT_LAZY_UPDATES to ensure all dependencies install correctly
            self.connection.run(base_cmd_env + " && unset OT_LAZY_UPDATES && ./install.sh", in_stream=False)
        
        # Validate CUDA driver and Torch compatibility; fall back to a lower CUDA tag if necessary.
        try:
            venv_python = f"{config.onetrainer_dir}/venv/bin/python"
            venv_pip = f"{config.onetrainer_dir}/venv/bin/pip"
            # Get driver version (may fail on some images)
            drv = self.connection.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1", in_stream=False, warn=True, hide='both')
            driver_version = (drv.stdout or "").strip()
            # Test if torch can initialize CUDA; catch any exception in Python side and return specific exit code 42
            test_cuda = (
                f"{shlex.quote(venv_python)} -c "
                "\"import sys; import torch; "
                "try:\\n import torch.cuda as _c; n=_c.device_count(); "
                "print(f'CUDA_DEVICES={n}'); "
                "sys.exit(0 if n and n>0 else 42)\\n"
                "except Exception as e:\\n "
                " print('CUDA_INIT_FAIL:', e); sys.exit(42)\""
            )
            result = self.connection.run(test_cuda, in_stream=False, warn=True, hide='both')
            if result.exited == 42:
                index_url, torch_ver, tv_ver = self._select_torch_wheels_for_driver(driver_version)
                print(f"CUDA initialization failed (driver={driver_version}). Installing Torch {torch_ver} / torchvision {tv_ver} from {index_url}.")
                uninstall = (
                    f"{shlex.quote(venv_pip)} uninstall -y torch torchvision triton "
                    "nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 "
                    "nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 "
                    "nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 "
                    "nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-cufile-cu12"
                )
                install = (
                    f"{shlex.quote(venv_pip)} install --upgrade --force-reinstall --no-cache-dir "
                    f"--index-url {shlex.quote(index_url)} "
                    f"torch=={torch_ver} torchvision=={tv_ver}"
                )
                self.connection.run(uninstall, in_stream=False, warn=True)
                self.connection.run(install, in_stream=False, warn=True)
        except Exception:
            # Never derail installation if detection fails
            pass
        
        # Verify mgds can be imported after installation/update
        # This ensures the editable install worked correctly
        venv_exists = self.connection.run(f"test -d {shlex.quote(config.onetrainer_dir)}/venv", warn=True, in_stream=False)
        if venv_exists.exited == 0:
            self._verify_mgds_import(config.onetrainer_dir)

    def _make_tensorboard_tunnel(self):
        self.tensorboard_tunnel_stop=threading.Event()
        self.tensorboard_tunnel=fabric.tunnels.TunnelManager(
            local_host='localhost',
            local_port=self.config.tensorboard_port,
            remote_host='localhost',
            remote_port=self.config.tensorboard_port,
            transport=self.connection.client.get_transport(),
            finished=self.tensorboard_tunnel_stop
        )
        self.tensorboard_tunnel.start()


    def close(self):
        if self.tensorboard_tunnel_stop is not None:
            self.tensorboard_tunnel_stop.set()
        if self.callback_connection:
            self.callback_connection.close()
        if self.command_connection:
            self.command_connection.close()
        if self.file_sync:
            self.file_sync.close()
        if self.connection:
            self.connection.close()
            self.connection=None

    def can_reattach(self):
        # Ensure paths are present, then check pid file
        try:
            self._ensure_remote_paths()
        except Exception:
            return False
        result=self.connection.run(f"test -f {shlex.quote(self.pid_file)}",warn=True,in_stream=False)
        return result.exited == 0

    def _get_action_cmd(self,action : CloudAction):
        if action != CloudAction.NONE:
            raise NotImplementedError("Action on detached not supported for this cloud type")
        return ":"

    def run_trainer(self):
        config=self.config.cloud
        if self.can_reattach():
            self.__trail_detached_trainer()
            return
        
        # Last-minute CUDA compatibility preflight: select a single Torch/CUDA build based on driver version.
        try:
            venv_python = f"{config.onetrainer_dir}/venv/bin/python"
            venv_pip = f"{config.onetrainer_dir}/venv/bin/pip"
            drv = self.connection.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1", in_stream=False, warn=True, hide='both')
            driver_version = (drv.stdout or "").strip()
            test_cuda = (
                f"{shlex.quote(venv_python)} -c "
                "\"import sys; import torch; "
                "try:\\n import torch.cuda as _c; n=_c.device_count(); "
                "print(f'CUDA_DEVICES={n}'); "
                "sys.exit(0 if n and n>0 else 42)\\n"
                "except Exception as e:\\n "
                " print('CUDA_INIT_FAIL:', e); sys.exit(42)\""
            )
            result = self.connection.run(test_cuda, in_stream=False, warn=True, hide='both')
            if result.exited == 42:
                index_url, torch_ver, tv_ver = self._select_torch_wheels_for_driver(driver_version)
                print(f"CUDA preflight failed (driver={driver_version}). Installing Torch {torch_ver} / torchvision {tv_ver} from {index_url}...")
                uninstall = (
                    f"{shlex.quote(venv_pip)} uninstall -y torch torchvision triton "
                    "nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 "
                    "nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 "
                    "nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 "
                    "nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-cufile-cu12"
                )
                install = (
                    f"{shlex.quote(venv_pip)} install --upgrade --force-reinstall --no-cache-dir "
                    f"--index-url {shlex.quote(index_url)} "
                    f"torch=={torch_ver} torchvision=={tv_ver}"
                )
                self.connection.run(uninstall, in_stream=False, warn=True)
                self.connection.run(install, in_stream=False, warn=True)
                # Re-test and if still failing, do a single fallback step to next-older bucket
                r2 = self.connection.run(test_cuda, in_stream=False, warn=True, hide='both')
                if r2.exited == 42:
                    # choose next-older option, if available
                    drv_tuple = self._parse_driver_version(driver_version)
                    chosen_idx = 0
                    for idx, (min_drv, _, _, _) in enumerate(self._cuda_thresholds):
                        if drv_tuple >= min_drv:
                            chosen_idx = idx
                            break
                    next_idx = min(chosen_idx + 1, len(self._cuda_thresholds) - 1)
                    _, index_url2, torch_ver2, tv_ver2 = self._cuda_thresholds[next_idx]
                    print(f"CUDA still failing; trying next-older Torch {torch_ver2} / {tv_ver2}.")
                    self.connection.run(uninstall, in_stream=False, warn=True)
                    install2 = (
                        f"{shlex.quote(venv_pip)} install --upgrade --force-reinstall --no-cache-dir "
                        f"--index-url {shlex.quote(index_url2)} "
                        f"torch=={torch_ver2} torchvision=={tv_ver2}"
                    )
                    self.connection.run(install2, in_stream=False, warn=True)
        except Exception:
            pass

        # Verify mgds can be imported before training starts
        # This catches any issues with editable installs before training fails
        self._verify_mgds_import(config.onetrainer_dir)

        cmd="export PATH=$PATH:/usr/local/cuda/bin:/venv/main/bin \
             && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu \
             && export OT_DISABLE_ZLUDA=1 \
             && export PYTHONUNBUFFERED=1 \
             && export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1} \
             && export NCCL_DEBUG=${NCCL_DEBUG:-INFO} \
             && export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} \
             && export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1} \
             && export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1} \
             && export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}"

        if self.config.secrets.huggingface_token != "":
            cmd+=f" && export HF_TOKEN={self.config.secrets.huggingface_token}"
        if config.huggingface_cache_dir != "":
            cmd+=f" && export HF_HOME={config.huggingface_cache_dir}"

        # Ensure prepare_runtime_environment installs dependencies properly
        # by unsetting OT_LAZY_UPDATES, which would otherwise skip dependency checks
        # Also force-sync repo to latest master on the pod to ensure runtime fixes are applied.
        cmd+=f' && cd {shlex.quote(config.onetrainer_dir)} && git fetch --all --prune && git reset --hard origin/master && unset OT_LAZY_UPDATES'
        # Ensure accelerate is available in the venv (some base images may have stale envs)
        venv_python = f"{config.onetrainer_dir}/venv/bin/python"
        venv_pip = f"{config.onetrainer_dir}/venv/bin/pip"
        cmd+=f" && ({shlex.quote(venv_python)} -c 'import accelerate' || {shlex.quote(venv_pip)} install --upgrade --no-cache-dir accelerate==1.7.0)"

        cmd+=f' && {config.onetrainer_dir}/run-cmd.sh train_remote --config-path={shlex.quote(self.config_file)} \
                                                                   --callback-path={shlex.quote(self.callback_file)} \
                                                                   --command-path={shlex.quote(self.command_pipe)}'

        if config.detach_trainer:
            self.connection.run(f'rm -f {self.exit_status_file}',in_stream=False)

            cmd=f"({cmd} ; exit_status=$? ; echo $exit_status > {self.exit_status_file}; exit $exit_status)"

            #if the callback file still exists 10 seconds after the trainer has exited, the client must be detached, because the clients reads and deletes this file:
            cmd+=f" && (sleep 10 && test -f {shlex.quote(self.callback_file)} && {self._get_action_cmd(config.on_detached_finish)} || true) \
                    || (sleep 10 && test -f {shlex.quote(self.callback_file)} && {self._get_action_cmd(config.on_detached_error)})"

            cmd=f'(nohup true && {cmd}) > {self.log_file} 2>&1 & echo $! > {self.pid_file}'
            self.connection.run(cmd,disown=True)
            self.__trail_detached_trainer()
        else:
            self.connection.run(cmd,in_stream=False)

    def __trail_detached_trainer(self):
        cmd=f'tail -f {self.log_file} --pid $(<{self.pid_file})'
        self.connection.run(cmd,in_stream=False)
        #trainer has exited, don't reattach:
        self.connection.run(f'rm -f {self.pid_file}',in_stream=False)
        #raise an exception if the training process return an exit code != 0:
        self.connection.run(f'exit $(<{self.exit_status_file})',in_stream=False)




    def exec_callback(self,callbacks : TrainCallbacks):
        #callbacks are a file instead of a named pipe, because of the blocking behaviour of linux pipes:
        #writing to pipes on the cloud can slow down training, and would cause issues in case
        #of a detached cloud trainer.
        #
        #use 'rename' as the atomic operation, to avoid race conditions between reader and writer:

        file=f'{shlex.quote(self.callback_file)}'
        cmd=f'mv "{file}" "{file}.read" \
           && cat "{file}.read" \
           && rm "{file}.read"'

        self.callback_connection.open()
        in_file,out_file,err_file=self.callback_connection.client.exec_command(cmd)

        try:
            while True:
                try:
                    while not out_file.channel.recv_ready() and not out_file.channel.exit_status_ready():
                        #even though reading from out_file is blocking, it doesn't block if there
                        #is *no* data available yet, which results in an unpickling error.
                        #wait until there is at least some data before reading:
                        time.sleep(0.1)
                    name=pickle.load(out_file)
                except EOFError:
                    return
                params=pickle.load(out_file)

                fun=getattr(callbacks,name)
                fun(*params)
        finally:
            in_file.close()
            out_file.close()
            err_file.close()


    def send_commands(self,commands : TrainCommands):
        try:
            self.command_connection.open()
            in_file,out_file,err_file=self.command_connection.client.exec_command(
                f'test -e {shlex.quote(self.command_pipe)} \
                && cat > {shlex.quote(self.command_pipe)}'
            )
            try:
                pickle.dump(commands,in_file)
                in_file.flush()
                in_file.channel.shutdown_write()
            finally:
                in_file.close()
                out_file.close()
                err_file.close()

            commands.reset()
        except Exception:
            if not self.command_connection.is_connected:
                print("\n\nCommand SSH connection lost. Attempting to reconnect...")
                self.command_connection.open()
                self.send_commands(commands)
            else:
                raise

    def _upload_config_file(self,local : Path):
        self.file_sync.sync_up_file(local,Path(self.config_file))

    def sync_workspace(self):
        local_workspace = Path(self.config.local_workspace_dir).expanduser()
        if not local_workspace.is_absolute():
            local_workspace = (Path.cwd() / local_workspace).resolve()
        else:
            local_workspace = local_workspace.resolve()

        local_workspace.mkdir(parents=True, exist_ok=True)

        self.file_sync.sync_down_dir(
            local=local_workspace,
            remote=Path(self.config.workspace_dir),
            filter=lambda path: BaseCloud._filter_download(config=self.config.cloud, path=path),
        )

    def delete_workspace(self):
        self.connection.run(f"rm -r {shlex.quote(self.config.workspace_dir)}",in_stream=False)
