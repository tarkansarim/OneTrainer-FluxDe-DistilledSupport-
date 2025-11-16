import datetime
import os
import platform
import subprocess
import traceback
from contextlib import suppress
import socket

import modules.util.multi_gpu_util as multi
from modules.trainer.BaseTrainer import BaseTrainer
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig

import torch


class MultiTrainer(BaseTrainer):
    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)
        if config.samples_to_tensorboard:
            print("Warning: If 'Samples To Tensorboard' is enabled, only one GPU is used for sampling!")
        if not config.latent_caching:
            print("Warning: Latent caching is disabled, but recommended for multi-GPU training!")

    def start(self):
        # Prefer explicit loopback to avoid name resolution issues on Windows
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        # Select a free TCP port if none provided to avoid bind conflicts (Windows often blocks fixed ports)
        if 'MASTER_PORT' not in os.environ or not os.environ['MASTER_PORT']:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('127.0.0.1', 0))
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    free_port = sock.getsockname()[1]
            except Exception:
                # Fall back to a common default if dynamic allocation fails
                free_port = 29500
            os.environ['MASTER_PORT'] = str(free_port)

        if platform.system() == 'Linux': #topo is not supported on Windows
            with suppress(subprocess.CalledProcessError):
                subprocess.run(['nvidia-smi', 'topo', '-p2p', 'rw'], check=True)
                print("\nWarning: If the matrix above does not show OK everywhere, your GPUs cannot communicate directly with each other. "
                        "Multi-GPU training with values other than OK is still possible, but this can have a major performance impact, "
                        "especially for full finetuning.\n")

    @staticmethod #must be static and not use __ prefix, otherwise the pickling done by torch.multiprocessing fails
    def _train_process(spawn_rank: int, world_size: int, config_dict: dict, devices: list[torch.device], callbacks: TrainCallbacks=None):
        if callbacks is None:
            callbacks = TrainCallbacks()
        rank = spawn_rank + 1
        config = TrainConfig.default_values().from_dict(config_dict)
        device = torch.device(devices[rank]) if devices else torch.device(config.train_device, rank)

        #set timeout to 24 hours, because caching is only done on 1 GPU and can take a significant time for large datasets.
        #The other GPU processes have to wait without timing out:
        timeout = datetime.timedelta(hours=24)

        torch.distributed.init_process_group(rank=rank, world_size=world_size, device_id=device, timeout=timeout,
            backend='gloo' if platform.system() == 'Windows' else 'nccl',
        )
        torch.cuda.set_device(device.index)

        #use barrier synchronisation now already, to discover NCCL communication issues early:
        if multi.is_master():
            print("Synchronizing GPUs. If this stalls, this likely means that your NCCL installation is broken:")
        for _ in multi.sequential():
            print(f"GPU #{multi.rank()}  device: {device} ({torch.cuda.get_device_name()})  "
                  f"backend: {torch.distributed.get_backend()}  world size: {torch.distributed.get_world_size()}")
        if multi.is_master():
            print("GPUs synchronized.")


        trainer = GenericTrainer(config, callbacks, TrainCommands())
        try:
            for _ in multi.sequential(enabled=config.sequential_model_setup):
                trainer.start()
            trainer.train()
        except Exception:
            traceback.print_exc()
            raise
        finally:
            trainer.end()
            # Synchronize all ranks before destroying process group to prevent deadlock
            # where rank 0 tears down TCPStore while other ranks are still using it
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            torch.distributed.destroy_process_group()

    def train(self):
        config_dict = self.config.to_pack_dict(secrets=True)

        devices = self.config.device_indexes.split(',')
        if len(devices) == 1 and not devices[0]:
            devices = None
            world_size = torch.cuda.device_count()
        else:
            devices = [torch.device(self.config.train_device, int(d)) for d in devices]
            world_size = len(devices)

        workers = torch.multiprocessing.spawn(MultiTrainer._train_process, args=(world_size, config_dict, devices), nprocs=world_size - 1, join=False)

        multi.set_global_commands(self.commands)
        MultiTrainer._train_process(-1, world_size, config_dict, devices, callbacks=self.callbacks) #main process is rank #0
        workers.join()

    def end(self):
        pass
