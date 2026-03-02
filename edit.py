#!/usr/bin/env python3
"""Texture expansion tool using OpenRouter API.

Takes a texture image, describes it via Gemini Flash, then generates
an expanded seamless version via Gemini Pro Image.
"""

import argparse
import base64
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────

API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = Path(__file__).parent / "key.txt"

EXPAND_SIZE = 4100  # px, target canvas for the generation step

MODEL_DESCRIBE = "google/gemini-3-flash-preview"
MODEL_GENERATE = "google/gemini-3-pro-image-preview"

PROMPT_DESCRIBE = """\
You are a professional 3D artist and texture engineer specialized in \
physically based rendering (PBR). Your task is to analyze the provided \
texture image and generate a concise, highly descriptive prompt suitable \
for an AI image generator (like Stable Diffusion or Flux) to replicate \
and expand this texture seamlessly.

Analyze the image and provide the output strictly in the following format. \
Do not use conversational filler (e.g., "Here is the description").

**Output Format:**
[Subject/Material]: <What is it? e.g. "Whitewashed Oak Wood Flooring">
[Color Palette]: <Dominant colors and undertones, hex codes if approximate \
is possible, e.g. "Pale beige, cool gray, hints of warm taupe">
[Structure/Layout]: <How is it organized? e.g. "Horizontal wooden planks \
of varying lengths, staggered joint pattern">
[Surface Details]: <Micro-details, glossiness, imperfections, e.g. \
"Visible wood grain, subtle knots, matte finish, slight weathering">
[Style/Vibe]: <Aesthetic description, e.g. "Scandi-style, modern, \
minimalist, clean">"""

PROMPT_GENERATE_BASE = """\
Basing on the reference image and description of the texture, expand \
texture to fill missing parts.
PRIMARY INSTRUCTION: prefer color and texture consistency, care of tile borders.

"""


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_api_key(path: Path = KEY_FILE) -> str:
    """Read the API key from a file, stripping whitespace."""
    return path.read_text().strip()


def encode_image_to_base64(image_path: str) -> str:
    """Read a file from disk and return its Base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_image_from_base64(output_path: str, b64_data: str) -> None:
    """Decode a Base64 string and write the result to a file."""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_data))


def expand_to_square(b64_data: str, size: int = EXPAND_SIZE) -> str:
    """Paste the image onto a white square canvas of *size*×*size* (centered).

    Returns a new Base64-encoded PNG string.
    """
    img = Image.open(BytesIO(base64.b64decode(b64_data)))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))

    offset_x = 0
    offset_y = 0
    # offset_x = (size - img.width) // 2
    # offset_y = (size - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))

    # canvas.save("buf.png", format="PNG")
    buf = BytesIO()
    canvas.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_data_url(b64_data: str, mime: str = "image/png") -> str:
    """Wrap Base64 payload into a data-URL."""
    return f"data:{mime};base64,{b64_data}"


# ── API calls ────────────────────────────────────────────────────────────────


def describe_texture(api_key: str, data_url: str) -> str:
    """Ask the model to describe the texture in a structured format."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_DESCRIBE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_DESCRIBE},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "reasoning": {"enabled": True},
    }

    resp = requests.post(API_URL, headers=headers, json=payload)
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]


def generate_image(api_key: str, prompt: str, data_url: str) -> str | None:
    """Send the prompt + reference image and return the Base64 image data."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_GENERATE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "modalities": ["image", "text"],
        "image_config": {
            "aspect_ratio": "1:1",
            "image_size": "4K",
        },
    }

    resp = requests.post(API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    result = resp.json()

    choices = result.get("choices", [])
    if not choices:
        return None

    message = choices[0]["message"]
    images = message.get("images", [])
    if not images:
        return None

    raw_url: str = images[0]["image_url"]["url"]
    idx = raw_url.find("base64,")
    if idx < 0:
        return None

    return raw_url[idx + 7 :]


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand a texture image using Gemini via OpenRouter.",
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input texture image.",
    )
    parser.add_argument(
        "-o", "--output",
        default="output.png",
        help="Path for the generated output image (default: output.png).",
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    api_key = load_api_key()

    # Encode the source image
    b64_original = encode_image_to_base64(args.input)
    data_url_original = make_data_url(b64_original)

    # Step 1 — describe the texture
    print("⏳ Describing texture…")
    description = describe_texture(api_key, data_url_original)
    prompt = PROMPT_GENERATE_BASE + description
    print(f"📝 Prompt:\n{prompt}\n")

    # Step 2 — expand image to 4100×4100 canvas and generate
    print("⏳ Expanding canvas to 4100×4100…")
    b64_expanded = expand_to_square(b64_original, EXPAND_SIZE)
    data_url_expanded = make_data_url(b64_expanded)

    print("⏳ Generating expanded texture…")
    b64_result = generate_image(api_key, prompt, data_url_expanded)

    if b64_result:
        save_image_from_base64(args.output, b64_result)
        print(f"✅ Saved result to {args.output}")
    else:
        print("❌ No image was returned by the model.")


if __name__ == "__main__":
    main()
