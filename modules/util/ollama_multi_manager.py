import contextlib
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import requests


@dataclass
class _ProcessInfo:
    gpu_index: int
    port: int
    process: subprocess.Popen


class MultiOllamaManager:
    """
    Launches one Ollama server per GPU so caption generation can run in parallel.
    Each GPU binds to its own port: host=127.0.0.1:(base_port + gpu_index)
    """

    def __init__(
        self,
        gpu_indices: Sequence[int],
        base_port: int = 12134,
        show_console: bool = False,
    ) -> None:
        if not gpu_indices:
            raise ValueError("MultiOllamaManager requires at least one GPU index.")

        self._gpu_indices: List[int] = self._normalize_indices(gpu_indices)
        self._base_port = int(base_port)
        self._processes: Dict[int, _ProcessInfo] = {}
        self._hosts: Dict[int, str] = {}
        self._show_console = show_console

    @staticmethod
    def _normalize_indices(indices: Sequence[int]) -> List[int]:
        normalized: List[int] = []
        for value in indices:
            try:
                normalized.append(int(value))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid GPU index: {value}") from None
        deduped = sorted(dict.fromkeys(normalized))
        return deduped

    def start_all(self, timeout: float = 90.0) -> None:
        if self._processes:
            return

        for gpu_index in self._gpu_indices:
            try:
                self._start_instance(gpu_index, timeout)
            except Exception:
                self.stop_all()
                raise

        print(
            f"[Detail Captions] Multi-GPU captioning online "
            f"(GPUs={self._gpu_indices}, ports={[self._port_for_gpu(idx) for idx in self._gpu_indices]})"
        )

    def _start_instance(self, gpu_index: int, timeout: float) -> None:
        port = self._port_for_gpu(gpu_index)
        host = f"127.0.0.1:{port}"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["OLLAMA_HOST"] = host
        env.setdefault("OLLAMA_NUM_PARALLEL", "4")

        if self._show_console and os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            env.setdefault("OLLAMA_DEBUG", "1")
            env.setdefault("OLLAMA_LOG_LEVEL", "debug")
            log_root = os.path.join(os.getcwd(), "workspace-cache", "run", "ollama-logs")
            os.makedirs(log_root, exist_ok=True)
            log_path = os.path.join(log_root, f"ollama_gpu{gpu_index}.log")
            log_literal = log_path.replace("'", "''")
            ps_command = (
                "$Host.UI.RawUI.WindowTitle = "
                f"'Ollama GPU {gpu_index} (port {port})'; "
                f"Write-Host 'Starting ollama serve on GPU {gpu_index} (port {port}) logging to {log_path}'; "
                f"ollama serve | Tee-Object -FilePath '{log_literal}' -Append"
            )
            process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoExit",
                    "-Command",
                    ps_command,
                ],
                env=env,
                stdout=None,
                stderr=None,
                creationflags=creationflags,
            )
        else:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                ["ollama", "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )

        self._processes[gpu_index] = _ProcessInfo(gpu_index=gpu_index, port=port, process=process)
        self._hosts[gpu_index] = f"http://{host}"
        self._wait_for_server(port, timeout, process)

    def stop_all(self, force: bool = False) -> None:
        for info in list(self._processes.values()):
            proc = info.process
            if proc.poll() is None:
                if force:
                    proc.kill()
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            self._processes.pop(info.gpu_index, None)
        self._hosts.clear()

    def get_hosts(self) -> List[str]:
        ordered = []
        for gpu_index in self._gpu_indices:
            host = self._hosts.get(gpu_index)
            if host:
                ordered.append(host)
        return ordered

    def _wait_for_server(self, port: int, timeout: float, process: subprocess.Popen) -> None:
        url = f"http://127.0.0.1:{port}/api/tags"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Ollama server on port {port} exited during startup.")
            try:
                response = requests.get(url, timeout=2.0)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"Ollama server on port {port} did not become ready within {timeout} seconds.")

    def _port_for_gpu(self, gpu_index: int) -> int:
        return self._base_port + gpu_index

    def __del__(self):
        with contextlib.suppress(Exception):
            self.stop_all()

__all__ = ["MultiOllamaManager"]

