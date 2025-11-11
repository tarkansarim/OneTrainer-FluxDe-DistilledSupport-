import os
import platform
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class _State:
    managed_process: Optional[subprocess.Popen] = None
    service_was_running: bool = False
    prepared: bool = False
    last_device_index: Optional[str] = None
    last_env: Optional[dict[str, str]] = None


_state = _State()


def prepare(train_device: str, device_indexes: str, multi_gpu: bool):
    """Restart Ollama bound to the training GPU before training starts."""
    if _state.prepared:
        return

    system = platform.system().lower()
    device_index = _select_device(train_device, device_indexes, multi_gpu)
    if device_index is None:
        return

    try:
        if system == "windows":
            _state.service_was_running = _stop_windows_service()
            _kill_ollama_processes_windows()
        else:
            _kill_ollama_processes_posix()

        # Ensure no other Ollama servers are running that might block the port
        _kill_ollama_processes_windows() if system == "windows" else _kill_ollama_processes_posix()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_index
        print(f"[Ollama] Restarting ollama serve with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        _state.last_device_index = device_index
        _state.last_env = env
        _launch_ollama_process(env)
        _state.prepared = True
    except FileNotFoundError:
        _state.managed_process = None
        _state.prepared = False
        if _state.service_was_running:
            _start_windows_service()
        raise RuntimeError("Could not find 'ollama' executable. Install Ollama or add it to PATH.")
    except Exception:
        if _state.managed_process:
            with suppress(Exception):
                _state.managed_process.terminate()
            _state.managed_process = None
        if _state.service_was_running:
            _start_windows_service()
        _state.prepared = False
        raise


def cleanup():
    """Restore Ollama to its pre-training state."""
    if not _state.prepared:
        return

    if _state.managed_process:
        try:
            _state.managed_process.terminate()
            _state.managed_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _state.managed_process.kill()
        finally:
            _state.managed_process = None

    if platform.system().lower() == "windows" and _state.service_was_running:
        _start_windows_service()

    _state.service_was_running = False
    _state.prepared = False
    _state.last_device_index = None
    _state.last_env = None


def restart():
    """Force a restart of the managed Ollama process."""
    ensure_running(force_restart=True)


def _select_device(train_device: Optional[str], device_indexes: str, multi_gpu: bool) -> Optional[str]:
    indexes = [idx.strip() for idx in (device_indexes or "").split(",") if idx.strip()]
    if indexes:
        if len(indexes) > 1:
            return ",".join(indexes)
        return indexes[0]
    if train_device:
        device = train_device.strip().lower()
        if device.startswith("cuda"):
            parts = device.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                return parts[1]
            return "0"
    return None


def ensure_running(force_restart: bool = False):
    if not _state.prepared:
        return

    proc = _state.managed_process
    if not force_restart and proc and proc.poll() is None:
        return

    device_index = _state.last_device_index
    env = _state.last_env
    if device_index is None or env is None:
        return

    system = platform.system().lower()
    if system == "windows":
        _stop_windows_service()
        _kill_ollama_processes_windows()
    else:
        _kill_ollama_processes_posix()

    if proc:
        with suppress(Exception):
            proc.terminate()
        with suppress(Exception):
            proc.wait(timeout=5)

    _launch_ollama_process(env)


def _stop_windows_service() -> bool:
    try:
        result = subprocess.run(
            ["sc", "query", "Ollama"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "RUNNING" not in result.stdout:
            return False
        subprocess.run(["sc", "stop", "Ollama"], check=False, capture_output=True)
        _wait_for_service_state("Ollama", "STOPPED")
        return True
    except Exception:
        return False


def _start_windows_service():
    try:
        subprocess.run(["sc", "start", "Ollama"], check=False, capture_output=True)
    except Exception:
        pass


def _wait_for_service_state(service: str, state: str, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["sc", "query", service],
            capture_output=True,
            text=True,
            check=False,
        )
        if state in result.stdout:
            return
        time.sleep(0.5)


def _kill_ollama_processes_windows():
    subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"], check=False, capture_output=True)


def _kill_ollama_processes_posix():
    subprocess.run(["pkill", "-f", "ollama"], check=False, capture_output=True)


def _wait_for_server(timeout: float = 60.0):
    url = "http://127.0.0.1:11434/api/tags"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _state.managed_process and _state.managed_process.poll() is not None:
            raise RuntimeError("Ollama process exited unexpectedly during startup")
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.0)
    raise RuntimeError("Timed out waiting for Ollama server to become ready")


def _launch_ollama_process(env: dict[str, str]):
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    _state.managed_process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_server()
