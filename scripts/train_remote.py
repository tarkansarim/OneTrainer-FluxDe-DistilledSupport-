from util.import_util import script_imports

script_imports()

import json
import os
import pickle
import threading
import traceback
from contextlib import suppress

# Disable ZLUDA probing on remote to avoid CUDA init issues due to driver/runtime mismatch.
os.environ["OT_DISABLE_ZLUDA"] = "1"

from modules.util import create
# As an extra safeguard, ensure any ZLUDA device probing is a no-op on remote.
try:
    from modules.zluda import ZLUDA as _ZLUDA
    _ZLUDA.initialize_devices = lambda *_, **__: None
except Exception:
    pass
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig


def write_request(filename,name, *params):
    try:
        with suppress(FileNotFoundError):
            os.rename(filename,filename+'.write')
        with open(filename+'.write', 'ab') as f:
            pickle.dump(name,f)
            pickle.dump(params,f)
        os.rename(filename+'.write',filename)
    except Exception:
        #TrainCallbacks is suppressing all exceptions; at least print them:
        traceback.print_exc()
        raise

def close_pipe(filename):
    with open(filename, 'wb'): #send EOF by closing
        os.remove(filename)

def _cuda_preflight() -> bool:
	"""
	Hardened CUDA preflight for remote runs.
	Returns True if a tiny CUDA tensor can be allocated; otherwise False.
	"""
	try:
		import torch
		# Early checks
		if not torch.cuda.is_available():
			print("CUDA_INIT_FAIL: torch.cuda.is_available() is False")
			return False
		n = torch.cuda.device_count()
		# Allocation test catches driver/runtime mismatches (e.g., error 803)
		_ = torch.empty(1, device="cuda:0")
		print(f"CUDA_PREFLIGHT_OK devices={n}")
		return True
	except Exception as e:
		print("CUDA_INIT_FAIL:", e)
		return False



def command_thread_function(commands: TrainCommands,filename : str,stop_event):
    while not stop_event.is_set():
        try:
            with open(filename, 'rb') as f:
                remote_commands=pickle.load(f)
        except FileNotFoundError:
            break
        except EOFError:
            continue

        commands.merge(remote_commands)


def main():
    args = TrainArgs.parse_args()
    if args.callback_path:
        callbacks = TrainCallbacks(
            on_update_train_progress=lambda *fargs:write_request(args.callback_path,"on_update_train_progress",*fargs),
            on_update_status=lambda *fargs:write_request(args.callback_path,"on_update_status",*fargs),
            on_sample_default=lambda *fargs:write_request(args.callback_path,"on_sample_default",*fargs),
            on_update_sample_default_progress=lambda *fargs:write_request(args.callback_path,"on_update_sample_default_progress",*fargs),
            on_sample_custom=lambda *fargs:write_request(args.callback_path,"on_sample_custom",*fargs),
            on_update_sample_custom_progress=lambda *fargs:write_request(args.callback_path,"on_update_sample_custom_progress",*fargs),
        )
    else:
        callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    os.environ["OT_REMOTE_SKIP_PATH_RESTORE"] = "1"

    for attr in ("workspace_dir", "cache_dir", "debug_dir"):
        value = getattr(train_config, attr, "")
        if value:
            os.makedirs(value, exist_ok=True)

    train_config.cloud.enabled=False

    try:
        with open("secrets.json" if args.secrets_path is None else args.secrets_path, "r") as f:
            secrets_dict=json.load(f)
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
    except FileNotFoundError:
        if args.secrets_path is not None:
            raise

    trainer = None
    # Optional CUDA preflight: default to skip (cloud sets up CUDA); can force by setting OT_SKIP_PREFLIGHT=0
    if os.environ.get("OT_SKIP_PREFLIGHT", "1") != "1":
        if not _cuda_preflight():
            raise SystemExit(42)

    trainer = create.create_trainer(train_config, callbacks, commands)

    if args.command_path:
        stop_event=threading.Event()
        command_thread = threading.Thread(target=command_thread_function,args=(commands,args.command_path,stop_event))
        command_thread.start()

    try:
        if trainer is not None:
            trainer.start()
            trainer.train()

    finally:
        if args.command_path:
            stop_event.set()
            close_pipe(args.command_path)
            command_thread.join()

        if trainer is not None:
            trainer.end()



if __name__ == '__main__':
    main()
