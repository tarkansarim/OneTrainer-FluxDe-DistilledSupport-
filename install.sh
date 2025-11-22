#!/usr/bin/env bash

set -e

# Install Ollama early if not present (Linux/RunPod only)
if [[ "$(uname -s)" == "Linux" ]]; then
    if ! command -v ollama &> /dev/null; then
        echo "[OneTrainer] Installing Ollama..."
        if command -v curl &> /dev/null; then
            curl -fsSL https://ollama.com/install.sh | sh
        else
            echo "[OneTrainer] Warning: curl not found, skipping auto-install of Ollama."
        fi
    else
        echo "[OneTrainer] Ollama is already installed."
    fi
fi

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

prepare_runtime_environment
