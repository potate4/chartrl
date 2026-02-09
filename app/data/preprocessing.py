"""Image and text preprocessing utilities."""

from typing import Tuple, Optional, Union
from PIL import Image
import io


# Default pixel constraints for Qwen2.5-VL
MIN_PIXELS = 4 * 28 * 28      # 3,136
MAX_PIXELS = 16384 * 28 * 28  # 12,845,056
DEFAULT_PIXELS = 512 * 28 * 28  # ~400k pixels (good balance)


def _resolve_resample(resample: Union[str, int]) -> int:
    if isinstance(resample, int):
        return resample
    name = str(resample).lower()
    if name == "bicubic":
        return Image.BICUBIC
    if name == "lanczos":
        return Image.LANCZOS
    if name == "bilinear":
        return Image.BILINEAR
    return Image.LANCZOS


def resize_image(
    image: Image.Image,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = DEFAULT_PIXELS,
    resample: Union[str, int] = "lanczos",
) -> Image.Image:
    """
    Resize image to fit within pixel constraints.

    Maintains aspect ratio while ensuring:
    - Total pixels >= min_pixels
    - Total pixels <= max_pixels

    Args:
        image: PIL Image to resize
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels
        resample: Resampling filter

    Returns:
        Resized PIL Image
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    current_pixels = width * height

    # Check if resize is needed
    if min_pixels <= current_pixels <= max_pixels:
        return image

    # Calculate scale factor
    if current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5
    else:
        scale = (max_pixels / current_pixels) ** 0.5

    new_width = int(width * scale)
    new_height = int(height * scale)

    # Ensure dimensions are at least 28 (patch size)
    new_width = max(28, new_width)
    new_height = max(28, new_height)

    return image.resize((new_width, new_height), resample=_resolve_resample(resample))


def process_image_for_model(
    image: Union[Image.Image, str, bytes],
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = DEFAULT_PIXELS,
    resample: Union[str, int] = "lanczos",
) -> Image.Image:
    """
    Process image for model input.

    Handles various input types and ensures proper format.

    Args:
        image: PIL Image, file path, or bytes
        max_pixels: Maximum pixels after resize

    Returns:
        Processed PIL Image
    """
    # Handle different input types
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if needed
    image = resize_image(image, min_pixels=min_pixels, max_pixels=max_pixels, resample=resample)

    return image


def get_image_size_info(image: Image.Image) -> dict:
    """
    Get image size information.

    Args:
        image: PIL Image

    Returns:
        Dict with size info
    """
    width, height = image.size
    return {
        "width": width,
        "height": height,
        "pixels": width * height,
        "aspect_ratio": width / height,
        "mode": image.mode,
    }


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Strip whitespace
    text = text.strip()

    # Normalize whitespace
    text = " ".join(text.split())

    # Lowercase
    text = text.lower()

    return text


def clean_json_string(s: str) -> str:
    """
    Clean a string that should be JSON.

    Handles common issues in model outputs.

    Args:
        s: JSON string

    Returns:
        Cleaned JSON string
    """
    if not s:
        return "{}"

    s = s.strip()

    # Remove markdown code blocks
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Fix single quotes
    # Be careful with values containing apostrophes
    import re
    # Replace single quotes used as JSON delimiters (not within strings)
    s = re.sub(r"(?<=[{,:\[])\s*'", '"', s)
    s = re.sub(r"'\s*(?=[}\],:])", '"', s)

    return s


def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate text to maximum length.

    Tries to break at sentence boundaries.

    Args:
        text: Input text
        max_length: Maximum character length

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    # Try to find sentence boundary
    truncated = text[:max_length]

    # Look for last sentence ending
    for ending in [". ", "! ", "? ", "\n"]:
        last_idx = truncated.rfind(ending)
        if last_idx > max_length // 2:
            return truncated[:last_idx + 1]

    # Fall back to word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        return truncated[:last_space] + "..."

    return truncated + "..."


def batch_process_images(
    images: list,
    max_pixels: int = DEFAULT_PIXELS,
) -> list:
    """
    Process a batch of images.

    Args:
        images: List of images (PIL, paths, or bytes)
        max_pixels: Maximum pixels per image

    Returns:
        List of processed PIL Images
    """
    return [process_image_for_model(img, max_pixels) for img in images]
