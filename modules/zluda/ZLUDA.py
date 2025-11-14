
from modules.util.config.TrainConfig import TrainConfig
import os

import torch
from torch._prims_common import DeviceLikeType


def is_zluda(device: DeviceLikeType):
    device = torch.device(device)
    if device.type in ["cpu", "mps"]:
        return False
    # Guard against environments where CUDA cannot be initialized (driver/runtime mismatch).
    try:
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_name(device).endswith("[ZLUDA]")
    except Exception:
        # If CUDA cannot init or device is invalid, it's not ZLUDA for our purposes.
        return False


def test(device: DeviceLikeType) -> Exception | None:
    device = torch.device(device)
    try:
        ten1 = torch.randn((2, 4,), device=device)
        ten2 = torch.randn((4, 8,), device=device)
        out = torch.mm(ten1, ten2)
        assert out.sum().is_nonzero()
        return None
    except Exception as e:
        return e


def initialize():
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def initialize_devices(config: TrainConfig):
    # Allow disabling ZLUDA probing via environment (used on cloud when drivers mismatch)
    if os.environ.get("OT_DISABLE_ZLUDA", "").lower() in ("1", "true", "yes"):
        return
    if not is_zluda(config.train_device) and not is_zluda(config.temp_device):
        return
    devices = [config.train_device, config.temp_device,]
    for i in range(2):
        device = torch.device(devices[i])
        result = test(device)
        if result is not None:
            dev_name = ""
            try:
                dev_name = torch.cuda.get_device_name(device)
            except Exception:
                dev_name = "<unavailable>"
            print(f'ZLUDA device failed to pass basic operation test: index={device.index}, device_name={dev_name}')
            print(result)
            devices[i] = 'cpu'
    config.train_device, config.temp_device = devices
