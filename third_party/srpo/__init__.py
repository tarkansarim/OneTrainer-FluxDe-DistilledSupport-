"""Utilities for integrating Tencent-Hunyuan's SRPO training stack."""

from .bootstrap import ensure_srpo_repo, SRPOBootstrapError
from .config import SRPO_REPO_URL, SRPO_REPO_REF, SRPO_VENDOR_DIR_NAME

__all__ = [
    "ensure_srpo_repo",
    "SRPOBootstrapError",
    "SRPO_REPO_URL",
    "SRPO_REPO_REF",
    "SRPO_VENDOR_DIR_NAME",
]





