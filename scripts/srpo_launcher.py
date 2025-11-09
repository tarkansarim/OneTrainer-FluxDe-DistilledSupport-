from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from third_party.srpo import (
    SRPOBootstrapError,
    SRPO_REPO_REF,
    SRPO_REPO_URL,
    ensure_srpo_repo,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and optionally launch Tencent-Hunyuan's SRPO training stack.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="OneTrainer workspace root (defaults to current directory).",
    )
    parser.add_argument(
        "--repo-url",
        default=SRPO_REPO_URL,
        help="Override the upstream SRPO repository URL.",
    )
    parser.add_argument(
        "--repo-ref",
        default=SRPO_REPO_REF,
        help="Optional commit/tag/branch to checkout after cloning.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force recloning the SRPO repository even if it already exists.",
    )
    parser.add_argument(
        "--script",
        default="scripts/finetune/SRPO_training_hpsv2.sh",
        help="Relative path to the SRPO launch script to execute.",
    )
    parser.add_argument(
        "--args-file",
        type=Path,
        help="JSON file containing additional launcher settings (env/args/working_dir).",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        help="Override the working directory used to execute the SRPO script.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment variable to propagate to the SRPO process (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the repository but skip executing the SRPO script.",
    )
    parser.add_argument(
        "--bash",
        default=os.environ.get("SRPO_BASH"),
        help="Path to bash executable (defaults to $SRPO_BASH or assumes it is on PATH).",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the SRPO script after '--'.",
    )
    return parser.parse_args()


def _load_args_file(args_file: Path | None) -> tuple[List[str], Dict[str, str], Path | None]:
    if not args_file:
        return [], {}, None

    try:
        payload = json.loads(args_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {args_file}: {exc}") from exc

    extra_args = payload.get("args", [])
    if not isinstance(extra_args, list):
        raise SystemExit("Expected 'args' field in args-file to be a list")

    env_updates = payload.get("env", {})
    if not isinstance(env_updates, dict):
        raise SystemExit("Expected 'env' field in args-file to be an object")

    working_dir = payload.get("working_dir")
    if working_dir is not None:
        working_dir = Path(working_dir)

    string_env = {str(k): str(v) for k, v in env_updates.items()}

    return [str(item) for item in extra_args], string_env, working_dir


def _resolve_launcher(script_path: Path, bash_path: str | None) -> List[str]:
    if script_path.suffix == ".sh":
        if os.name == "nt":
            powershell_stub = Path(__file__).resolve().with_name("srpo_windows_runner.ps1")
            if powershell_stub.exists():
                return [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(powershell_stub),
                    "--%",
                    str(script_path),
                ]
        exe = bash_path or "bash"
        return [exe, str(script_path)]
    if script_path.suffix == ".py":
        return [sys.executable, str(script_path)]
    # Fallback: attempt to execute directly.
    return [str(script_path)]


def _flag_exists(args: List[str], flag: str) -> bool:
    for idx, value in enumerate(args):
        if value == flag:
            return True
        if value.startswith(f"{flag}="):
            return True
        if value.startswith(f"{flag} "):
            return True
        if idx < len(args) - 1 and value == flag:
            return True
    return False


def _ensure_flag(args: List[str], flag: str, value: str) -> None:
    if _flag_exists(args, flag):
        return
    args.extend([flag, value])


def main() -> int:
    args = _parse_args()
    workspace_root = args.workspace.resolve()

    try:
        bootstrap_result = ensure_srpo_repo(
            workspace_root,
            repo_url=args.repo_url,
            ref=args.repo_ref,
            force_refresh=args.force_refresh,
        )
    except SRPOBootstrapError as exc:
        print(f"[SRPO] Failed to prepare repository: {exc}", file=sys.stderr)
        return 1

    repo_path = bootstrap_result.repo_path

    script_path = (repo_path / args.script).resolve()
    if not script_path.exists():
        print(f"[SRPO] Expected script not found: {script_path}", file=sys.stderr)
        return 1

    extra_args, env_updates, working_dir_override = _load_args_file(args.args_file)

    if args.working_dir is not None:
        working_dir_override = args.working_dir

    for env_entry in args.env:
        if "=" not in env_entry:
            raise SystemExit(f"Invalid --env entry (expected KEY=VALUE): {env_entry}")
        key, value = env_entry.split("=", 1)
        env_updates[key] = value

    if args.dry_run:
        print(f"[SRPO] Repository ready at: {repo_path}")
        print(f"[SRPO] Launch script located at: {script_path}")
        return 0

    command = _resolve_launcher(script_path, args.bash)

    env = os.environ.copy()
    env.update(env_updates)
    env.setdefault("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_path), env["PYTHONPATH"]]))

    working_dir = working_dir_override or repo_path
    auto_args_raw = env_updates.get("SRPO_AUTO_ARGS") or os.environ.get("SRPO_AUTO_ARGS")
    auto_args_pairs: List[tuple[str, Any]] = []
    if auto_args_raw:
        try:
            decoded = json.loads(auto_args_raw)
            if isinstance(decoded, list):
                for item in decoded:
                    if isinstance(item, list) and item:
                        flag = item[0]
                        rest = item[1:]
                        if isinstance(flag, str):
                            if not rest:
                                auto_args_pairs.append((flag, None))
                            elif len(rest) == 1:
                                auto_args_pairs.append((flag, str(rest[0])))
                            else:
                                auto_args_pairs.append((flag, [str(v) for v in rest]))
        except json.JSONDecodeError:
            print(f"[SRPO] Warning: failed to parse SRPO_AUTO_ARGS={auto_args_raw}", file=sys.stderr)
            auto_args_pairs = []

    data_json_path = env_updates.get("SRPO_DATA_JSON") or os.environ.get("SRPO_DATA_JSON")
    if data_json_path:
        data_json_path = Path(data_json_path).resolve()
        try:
            data_json_path = data_json_path.relative_to(working_dir.resolve())
        except ValueError:
            pass
        data_json_value = str(data_json_path)
    else:
        data_json_value = None

    passthrough = []
    if args.passthrough:
        # argparse includes the literal '--' as the first item; drop it.
        passthrough = [item for item in args.passthrough if item != "--"]

    script_args: List[str] = list(extra_args)
    if data_json_value:
        already_has_flag = _flag_exists(script_args, "--data_json_path") or _flag_exists(passthrough, "--data_json_path")
        if not already_has_flag:
            script_args.extend(["--data_json_path", data_json_value])

    for flag, value in auto_args_pairs:
        if _flag_exists(script_args, flag) or _flag_exists(passthrough, flag):
            continue
        if value is None:
            script_args.append(flag)
        elif isinstance(value, list):
            script_args.append(flag)
            script_args.extend(value)
        else:
            script_args.extend([flag, str(value)])

    script_args.extend(passthrough)
    command.extend(script_args)

    world_size_raw = env.get("WORLD_SIZE") or "1"
    try:
        world_size_int = int(str(world_size_raw))
    except ValueError:
        world_size_int = 1
        world_size_raw = "1"
    env["WORLD_SIZE"] = str(world_size_int)

    if world_size_int == 1:
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        env["LOCAL_WORLD_SIZE"] = "1"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "29500"
    else:
        env.setdefault("RANK", "0")
        env.setdefault("LOCAL_RANK", "0")
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "29500")

    if platform.system().lower().startswith("win"):
        env.setdefault("SRPO_DIST_BACKEND", "gloo")

    try:
        subprocess.run(command, cwd=str(working_dir), env=env, check=True)
    except FileNotFoundError as exc:
        print(
            f"[SRPO] Unable to execute '{command[0]}'. Ensure bash/Python is installed and reachable.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"[SRPO] Launch failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

