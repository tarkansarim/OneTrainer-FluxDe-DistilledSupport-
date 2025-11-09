from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import shutil

from .config import SRPO_REPO_REF, SRPO_REPO_URL, vendor_path


class SRPOBootstrapError(RuntimeError):
    """Raised when the SRPO repository cannot be prepared."""


@dataclass(frozen=True)
class BootstrapResult:
    repo_path: Path
    refreshed: bool


def _run_command(cmd: Iterable[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SRPOBootstrapError(
            "Required executable not found while preparing the SRPO repository."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SRPOBootstrapError(
            f"Command {' '.join(cmd)} failed with exit code {exc.returncode}."
        ) from exc


def _clone_repo(target: Path, repo_url: str, ref: str | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone", "--depth", "1", "--recurse-submodules"]
    if ref:
        cmd.extend(["--branch", ref])
    cmd.extend([repo_url, str(target)])

    _run_command(cmd)


def _checkout_ref(target: Path, ref: str) -> None:
    _run_command(["git", "fetch", "origin", ref, "--depth", "1"], cwd=target)
    _run_command(["git", "checkout", "FETCH_HEAD"], cwd=target)


def _sync_submodules(target: Path) -> None:
    _run_command(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=target,
    )


def ensure_srpo_repo(
    workspace_root: Path,
    *,
    repo_url: str = SRPO_REPO_URL,
    ref: str | None = SRPO_REPO_REF,
    force_refresh: bool = False,
) -> BootstrapResult:
    """Ensure the SRPO repository has been cloned locally.

    Parameters
    ----------
    workspace_root:
        Project root (where the OneTrainer workspace lives).
    repo_url:
        Upstream Git URL containing the SRPO training stack.
    ref:
        Optional commit-ish (tag/branch/sha) to checkout. Defaults to
        :data:`SRPO_REPO_REF`.
    force_refresh:
        When true, the repository will be recloned even if it already exists.

    Returns
    -------
    BootstrapResult
        Contains the final repo path and whether it was freshly cloned.
    """

    repo_path = vendor_path(workspace_root)

    if repo_path.exists() and not force_refresh:
        if ref:
            try:
                current_ref = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=repo_path,
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                ).stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                current_ref = None

            if current_ref and current_ref.startswith(ref):
                _sync_submodules(repo_path)
                return BootstrapResult(repo_path=repo_path, refreshed=False)

        if not ref:
            _sync_submodules(repo_path)
            return BootstrapResult(repo_path=repo_path, refreshed=False)

    if repo_path.exists():
        # Remove stale repository before cloning again to keep things clean.
        shutil.rmtree(repo_path)

    _clone_repo(repo_path, repo_url, ref)
    _sync_submodules(repo_path)

    # When ref is provided as a commit SHA rather than branch, ensure checkout.
    if ref and len(ref) == 40:
        _checkout_ref(repo_path, ref)
        _sync_submodules(repo_path)

    return BootstrapResult(repo_path=repo_path, refreshed=True)

