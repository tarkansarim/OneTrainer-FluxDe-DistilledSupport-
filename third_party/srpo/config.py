from __future__ import annotations

from pathlib import Path


# Directory (relative to project root) where the SRPO repository will be vendored.
SRPO_VENDOR_DIR_NAME = "third_party/srpo/vendor"

# Upstream repository metadata. The default ref keeps us anchored to a
# reproducible revision should Tencent update their main branch in breaking ways.
SRPO_REPO_URL = "https://github.com/Tencent-Hunyuan/SRPO.git"

# Default ref; keep None to track main, or set to a specific commit/tag for
# reproducibility. Leaving it None makes the bootstrapper fall back to the
# repository's default branch.
SRPO_REPO_REF: str | None = None


def vendor_path(workspace_root: Path) -> Path:
    """Return the absolute path where the SRPO repo is vendored."""

    return (workspace_root / SRPO_VENDOR_DIR_NAME).resolve()





