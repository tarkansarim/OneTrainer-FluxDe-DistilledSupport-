from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "third_party" / "srpo" / "vendor" / "hpsv2"
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="xswu/HPSv2",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


if __name__ == "__main__":
    main()




