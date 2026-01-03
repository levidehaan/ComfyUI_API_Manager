"""
Image Post Node

Enhanced node for posting images to API endpoints with support for:
- Multiple image formats (PNG, JPEG, WebP)
- Batch processing
- Progress tracking
- Comprehensive error handling
"""

import json
import logging
from io import BytesIO
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


class PostImageToAPI:
    """
    Posts images to an API endpoint with configurable options.

    Supports multiple image formats, authentication, and batch processing.
    """

    IMAGE_FORMATS = ["PNG", "JPEG", "WEBP"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "https://api.example.com/upload/$id"
                }),
            },
            "optional": {
                "api_object_id": ("STRING", {
                    "default": "",
                    "forceInput": True
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "forceInput": True
                }),
                "image_format": (cls.IMAGE_FORMATS, {"default": "PNG"}),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "field_name": ("STRING", {
                    "default": "file",
                    "multiline": False
                }),
                "extra_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Additional form data as JSON"
                }),
                "timeout_seconds": ("FLOAT", {
                    "default": 60.0,
                    "min": 5.0,
                    "max": 600.0,
                    "step": 5.0
                }),
            }
        }

    RETURN_TYPES = ("JSON", "BOOLEAN", "STRING", "INT")
    RETURN_NAMES = ("RESPONSES", "SUCCESS", "ERROR", "UPLOADED_COUNT")
    FUNCTION = "post_images"
    CATEGORY = "API Manager"
    OUTPUT_NODE = True

    def _image_tensor_to_pil(self, image_tensor) -> Image.Image:
        """Convert ComfyUI image tensor to PIL Image."""
        # ComfyUI images are (H, W, C) with values 0-1
        image_np = image_tensor.cpu().numpy()

        # Handle different tensor shapes
        if len(image_np.shape) == 4:
            image_np = image_np[0]  # Take first in batch

        # Convert from 0-1 float to 0-255 uint8
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(image_np)

    def _prepare_image_data(
        self,
        image: Image.Image,
        format: str,
        quality: int
    ) -> tuple[BytesIO, str, str]:
        """
        Prepare image data for upload.

        Returns:
            Tuple of (buffer, filename, mime_type)
        """
        buffer = BytesIO()

        format_upper = format.upper()
        if format_upper == "PNG":
            image.save(buffer, format="PNG", optimize=True)
            mime_type = "image/png"
            ext = "png"
        elif format_upper == "JPEG":
            # Convert to RGB if necessary (JPEG doesn't support alpha)
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            mime_type = "image/jpeg"
            ext = "jpg"
        elif format_upper == "WEBP":
            image.save(buffer, format="WEBP", quality=quality)
            mime_type = "image/webp"
            ext = "webp"
        else:
            # Default to PNG
            image.save(buffer, format="PNG")
            mime_type = "image/png"
            ext = "png"

        buffer.seek(0)
        filename = f"image.{ext}"

        return buffer, filename, mime_type

    def post_images(
        self,
        images,
        api_url: str,
        api_object_id: str = "",
        api_key: str = "",
        image_format: str = "PNG",
        quality: int = 95,
        field_name: str = "file",
        extra_data: str = "",
        timeout_seconds: float = 60.0
    ):
        """
        Post images to the API.

        Returns:
            Tuple of (responses, success, error_message, uploaded_count)
        """
        results = []
        errors = []
        uploaded_count = 0

        # Replace $id placeholder in URL
        url = api_url.replace("$id", str(api_object_id)) if api_object_id else api_url

        # Prepare headers
        headers = {}
        if api_key:
            if api_key.startswith("Bearer "):
                headers["Authorization"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"

        # Parse extra data
        extra_form_data = {}
        if extra_data and extra_data.strip():
            try:
                extra_form_data = json.loads(extra_data)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse extra_data as JSON")

        # Process each image
        for batch_idx, image_tensor in enumerate(images):
            try:
                # Convert tensor to PIL Image
                pil_image = self._image_tensor_to_pil(image_tensor)

                # Prepare image data
                buffer, filename, mime_type = self._prepare_image_data(
                    pil_image, image_format, quality
                )

                # Prepare multipart form data
                files = {
                    field_name: (filename, buffer, mime_type)
                }

                # Add extra form data
                data = dict(extra_form_data)
                if api_object_id:
                    data["object_id"] = api_object_id

                # Make request
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data if data else None,
                    timeout=timeout_seconds
                )

                if response.status_code in (200, 201):
                    try:
                        result = response.json()
                    except json.JSONDecodeError:
                        result = {"status": "success", "raw": response.text[:500]}

                    results.append({
                        "batch_index": batch_idx,
                        "success": True,
                        "response": result
                    })
                    uploaded_count += 1
                    logger.info(f"Successfully posted image {batch_idx} to {url}")
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    results.append({
                        "batch_index": batch_idx,
                        "success": False,
                        "error": error_msg
                    })
                    errors.append(f"Image {batch_idx}: {error_msg}")
                    logger.error(f"Failed to post image {batch_idx}: {error_msg}")

            except requests.exceptions.Timeout:
                error_msg = f"Timeout after {timeout_seconds}s"
                results.append({
                    "batch_index": batch_idx,
                    "success": False,
                    "error": error_msg
                })
                errors.append(f"Image {batch_idx}: {error_msg}")
                logger.error(f"Timeout posting image {batch_idx}")

            except Exception as e:
                error_msg = str(e)
                results.append({
                    "batch_index": batch_idx,
                    "success": False,
                    "error": error_msg
                })
                errors.append(f"Image {batch_idx}: {error_msg}")
                logger.error(f"Error posting image {batch_idx}: {e}")

        # Determine overall success
        all_success = len(errors) == 0
        error_summary = "; ".join(errors) if errors else ""

        return (
            {"results": results, "url": url, "total": len(images)},
            all_success,
            error_summary,
            uploaded_count
        )


class DownloadImageFromURL:
    """
    Downloads an image from a URL and converts it to ComfyUI format.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "https://example.com/image.png"
                }),
            },
            "optional": {
                "bearer_token": ("STRING", {
                    "default": "",
                    "forceInput": True
                }),
                "timeout_seconds": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 120.0,
                    "step": 5.0
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("IMAGE", "SUCCESS", "ERROR")
    FUNCTION = "download"
    CATEGORY = "API Manager"

    def download(
        self,
        url: str,
        bearer_token: str = "",
        timeout_seconds: float = 30.0
    ):
        """
        Download an image from URL.

        Returns:
            Tuple of (image_tensor, success, error_message)
        """
        import torch

        try:
            headers = {}
            if bearer_token:
                if not bearer_token.startswith("Bearer "):
                    bearer_token = f"Bearer {bearer_token}"
                headers["Authorization"] = bearer_token

            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()

            # Load image
            image = Image.open(BytesIO(response.content))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array (H, W, C)
            image_np = np.array(image).astype(np.float32) / 255.0

            # Convert to tensor (B, H, W, C)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)

            logger.info(f"Downloaded image from {url}")
            return (image_tensor, True, "")

        except requests.exceptions.Timeout:
            error_msg = f"Timeout after {timeout_seconds}s"
            logger.error(error_msg)
            # Return empty tensor
            empty = torch.zeros((1, 64, 64, 3))
            return (empty, False, error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to download image: {e}")
            empty = torch.zeros((1, 64, 64, 3))
            return (empty, False, error_msg)
