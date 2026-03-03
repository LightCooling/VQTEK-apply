#!/usr/bin/env python3
"""Texture expansion tool using OpenRouter API.

Takes a texture image, describes it via Gemini Flash, then generates
an expanded seamless version via Gemini Pro Image with tiling support
for outputs larger than 4096×4096.
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

TILE_SIZE = 4096       # max generation resolution (one side)
MIN_OVERLAP = 0.20     # 20% minimum overlap between adjacent tiles
DEFAULT_EXPAND = 4100  # default target size when not specified

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


def image_to_b64(img: Image.Image) -> str:
    """Encode a PIL Image to a Base64-encoded PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_image(b64_data: str) -> Image.Image:
    """Decode a Base64 string to a PIL Image."""
    return Image.open(BytesIO(base64.b64decode(b64_data)))


def make_data_url(b64_data: str, mime: str = "image/png") -> str:
    """Wrap Base64 payload into a data-URL."""
    return f"data:{mime};base64,{b64_data}"


# ── Tiling helpers ───────────────────────────────────────────────────────────


def compute_tile_positions(target: int, tile: int = TILE_SIZE,
                           min_overlap: float = MIN_OVERLAP) -> list[int]:
    """Return tile start positions along one axis.

    Guarantees:
    * Every pixel in [0, target) is covered.
    * Adjacent tiles overlap by at least *min_overlap* fraction of *tile*.
    * The last tile is right-aligned to *target*.
    """
    if target <= tile:
        return [0]

    step = int(tile * (1.0 - min_overlap))  # e.g. 4096 * 0.8 = 3276
    positions: list[int] = []
    pos = 0
    while pos + tile < target:
        positions.append(pos)
        pos += step
    # last tile flush with the right/bottom edge
    positions.append(target - tile)
    # deduplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def extract_tile(canvas: Image.Image, x: int, y: int,
                 tile: int = TILE_SIZE) -> Image.Image:
    """Crop a *tile*×*tile* region from *canvas* starting at (x, y)."""
    return canvas.crop((x, y, x + tile, y + tile))


def paste_tile(canvas: Image.Image, tile_img: Image.Image,
               x: int, y: int) -> None:
    """Paste *tile_img* onto *canvas* at position (x, y) in-place."""
    canvas.paste(tile_img, (x, y))


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

    return raw_url[idx + 7:]


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
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=None,
        help="Target width in pixels (default: same as EXPAND_SIZE).",
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=None,
        help="Target height in pixels (default: same as EXPAND_SIZE).",
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    api_key = load_api_key()

    target_w = args.width or DEFAULT_EXPAND
    target_h = args.height or DEFAULT_EXPAND

    # ── Load original image ──────────────────────────────────────────────
    original = Image.open(args.input).convert("RGB")
    orig_w, orig_h = original.size
    print(f"📐 Original size: {orig_w}×{orig_h}")
    print(f"🎯 Target size:   {target_w}×{target_h}")

    if target_w < orig_w or target_h < orig_h:
        print("❌ Target size must be >= original image size.")
        return

    # ── Step 1 — describe the texture ────────────────────────────────────
    b64_original = encode_image_to_base64(args.input)
    data_url_original = make_data_url(b64_original)

    print("⏳ Describing texture…")
    description = describe_texture(api_key, data_url_original)
    prompt = PROMPT_GENERATE_BASE + description
    print(f"📝 Prompt:\n{prompt}\n")

    # ── Step 2 — create target canvas and place original ─────────────────
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(original, (0, 0))

    # Track which pixels are "filled" (have real data)
    # We use a simple 2D boolean grid at pixel level via a 1-bit image
    filled = Image.new("L", (target_w, target_h), 0)  # 0 = empty
    filled.paste(
        Image.new("L", (orig_w, orig_h), 255),  # 255 = filled
        (0, 0),
    )

    # ── Step 3 — compute tile grid ───────────────────────────────────────
    xs = compute_tile_positions(target_w, TILE_SIZE, MIN_OVERLAP)
    ys = compute_tile_positions(target_h, TILE_SIZE, MIN_OVERLAP)

    total_tiles = len(xs) * len(ys)
    print(f"🧩 Tile grid: {len(xs)} cols × {len(ys)} rows = {total_tiles} tiles")
    print(f"   X positions: {xs}")
    print(f"   Y positions: {ys}")

    # ── Step 4 — iterate tiles (row-major) ───────────────────────────────
    tile_idx = 0
    for ty in ys:
        for tx in xs:
            tile_idx += 1

            # Check if this tile region has any unfilled pixels
            tile_filled = filled.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
            filled_pixels = sum(1 for p in tile_filled.getdata() if p > 0)
            total_pixels = TILE_SIZE * TILE_SIZE

            if filled_pixels == total_pixels:
                print(f"   ⏭️  Tile {tile_idx}/{total_tiles} "
                      f"at ({tx},{ty}) — fully filled, skipping.")
                continue

            pct_filled = filled_pixels / total_pixels * 100
            print(f"   ⏳ Tile {tile_idx}/{total_tiles} "
                  f"at ({tx},{ty}) — {pct_filled:.0f}% filled, generating…")

            # Extract tile from current canvas state
            tile_crop = extract_tile(canvas, tx, ty, TILE_SIZE)
            b64_tile = image_to_b64(tile_crop)
            data_url_tile = make_data_url(b64_tile)

            # Generate
            b64_result = generate_image(api_key, prompt, data_url_tile)

            if b64_result is None:
                print(f"   ⚠️  Tile {tile_idx}/{total_tiles} "
                      f"— no image returned, skipping.")
                continue

            # Paste result back onto canvas
            result_img = b64_to_image(b64_result)
            # Resize if the model returned a different size
            if result_img.size != (TILE_SIZE, TILE_SIZE):
                result_img = result_img.resize(
                    (TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS,
                )
            paste_tile(canvas, result_img, tx, ty)

            # Mark this region as filled
            filled.paste(
                Image.new("L", (TILE_SIZE, TILE_SIZE), 255),
                (tx, ty),
            )

            print(f"   ✅ Tile {tile_idx}/{total_tiles} done.")

    # ── Step 5 — save ────────────────────────────────────────────────────
    canvas.save(args.output, format="PNG")
    print(f"\n✅ Saved result ({target_w}×{target_h}) to {args.output}")


if __name__ == "__main__":
    main()
