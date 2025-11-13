import argparse
import base64
import io
import json
from typing import Any, Dict

import requests
from PIL import Image


DEFAULT_SYSTEM_PROMPT = (
    "You are an image tagging assistant for training detail crops. Produce concise comma-separated descriptors of what "
    "is visible in the crop. Use the provided context caption when it helps disambiguate close-ups. Never invent "
    "objects that are not clearly present."
)

DEFAULT_USER_PROMPT = (
    "Context caption: \"{context}\"\n"
    "Describe the visual content of this crop using comma-separated tags (max 30 words). Focus on objects, materials, "
    "textures, colors, lighting, and fine details. If the crop is too ambiguous, respond with \"unclear_crop\"."
)


def encode_image(path: str) -> str:
    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def format_prompt(template: str, context: str, metadata: Dict[str, Any]) -> str:
    safe_context = (context or "No additional context.").replace("\r", " ").replace("\n", " ").strip()
    substitutions = _DefaultDict({
        "context": safe_context,
        "crop_type": metadata.get("type"),
        "scale": metadata.get("scale"),
        "index": metadata.get("index"),
        "coords": metadata.get("coords"),
        "source_resolution": metadata.get("source_resolution"),
        "image_path": metadata.get("image_path"),
    })
    return template.format_map(substitutions)


def request_caption(
        image_b64: str,
        endpoint: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
) -> str:
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    response = requests.post(f"{endpoint.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    caption = str(body.get("response", "")).strip()
    if not caption:
        raise RuntimeError("Received empty caption from Ollama response.")
    return caption


def main():
    parser = argparse.ArgumentParser(description="Generate captions for detail crops via Ollama.")
    parser.add_argument("image", help="Path to the crop image to caption.")
    parser.add_argument("--context", default="", help="Context caption from the full image.")
    parser.add_argument("--endpoint", default="http://localhost:11434", help="Ollama endpoint URL.")
    parser.add_argument("--model", default="qwen2.5vl:3b", help="Ollama model name to use.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Override system prompt.")
    parser.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT, help="Override user prompt template.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--metadata", type=json.loads, default="{}",
                        help="Optional JSON dict with metadata placeholders (type, scale, index, coords, etc.).")
    args = parser.parse_args()

    image_b64 = encode_image(args.image)
    user_prompt = format_prompt(args.user_prompt, args.context, args.metadata)
    caption = request_caption(
        image_b64=image_b64,
        endpoint=args.endpoint,
        model=args.model,
        system_prompt=args.system_prompt,
        user_prompt=user_prompt,
        timeout=args.timeout,
    )

    print("Caption:")
    print(caption)


class _DefaultDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


if __name__ == "__main__":
    main()

