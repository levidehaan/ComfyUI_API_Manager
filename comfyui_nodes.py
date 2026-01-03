"""
ComfyUI Integration Nodes

Nodes for directly interacting with ComfyUI's API including:
- Workflow execution
- Queue management
- Image upload/download to ComfyUI
- System information
- Model listing
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get_event_loop():
    """Get or create an event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class ComfyUIExecuteWorkflowNode:
    """
    Executes a workflow on a ComfyUI server.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Paste workflow JSON (API format) here"
                }),
            },
            "optional": {
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
                "timeout": ("FLOAT", {"default": 300.0, "min": 10.0, "max": 3600.0}),
                "wait_for_completion": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("JSON", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("RESULT", "PROMPT_ID", "SUCCESS", "ERROR")
    FUNCTION = "execute"
    CATEGORY = "API Manager/ComfyUI"

    def execute(
        self,
        workflow_json: str,
        host: str = "127.0.0.1",
        port: int = 8188,
        timeout: float = 300.0,
        wait_for_completion: bool = True
    ):
        try:
            from .comfyui_client import ComfyUIClient, ComfyUIConnection, JobStatus

            workflow = json.loads(workflow_json)

            connection = ComfyUIConnection(host=host, port=port)
            client = ComfyUIClient(connection)

            loop = _get_event_loop()

            async def run():
                try:
                    await client.connect()
                    if wait_for_completion:
                        result = await client.execute_workflow(workflow, timeout=timeout)
                        return (
                            result.to_dict(),
                            result.prompt_id,
                            result.status == JobStatus.COMPLETED,
                            result.error or ""
                        )
                    else:
                        prompt_id = await client.queue_prompt(workflow)
                        return (
                            {"prompt_id": prompt_id, "status": "queued"},
                            prompt_id,
                            True,
                            ""
                        )
                finally:
                    await client.disconnect()

            return loop.run_until_complete(run())

        except json.JSONDecodeError as e:
            return ({}, "", False, f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return ({}, "", False, str(e))


class ComfyUIQueueStatusNode:
    """
    Gets the current queue status from ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = ("JSON", "INT", "INT")
    RETURN_NAMES = ("QUEUE_STATUS", "PENDING_COUNT", "RUNNING_COUNT")
    FUNCTION = "get_status"
    CATEGORY = "API Manager/ComfyUI"

    def get_status(self, host: str = "127.0.0.1", port: int = 8188):
        try:
            from .comfyui_client import ComfyUIClient, ComfyUIConnection

            connection = ComfyUIConnection(host=host, port=port)
            client = ComfyUIClient(connection)

            loop = _get_event_loop()

            async def run():
                try:
                    await client.connect()
                    queue = await client.get_queue()
                    pending = len(queue.get("queue_pending", []))
                    running = len(queue.get("queue_running", []))
                    return (queue, pending, running)
                finally:
                    await client.disconnect()

            return loop.run_until_complete(run())

        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return ({}, 0, 0)


class ComfyUISystemInfoNode:
    """
    Gets system information from ComfyUI including GPU stats.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = ("JSON", "STRING", "INT", "INT")
    RETURN_NAMES = ("SYSTEM_INFO", "DEVICE_NAME", "VRAM_TOTAL_MB", "VRAM_FREE_MB")
    FUNCTION = "get_info"
    CATEGORY = "API Manager/ComfyUI"

    def get_info(self, host: str = "127.0.0.1", port: int = 8188):
        try:
            from .comfyui_client import ComfyUIClient, ComfyUIConnection

            connection = ComfyUIConnection(host=host, port=port)
            client = ComfyUIClient(connection)

            loop = _get_event_loop()

            async def run():
                try:
                    await client.connect()
                    stats = await client.get_system_stats()

                    # Extract GPU info
                    devices = stats.get("devices", [])
                    if devices:
                        device = devices[0]
                        name = device.get("name", "Unknown")
                        vram_total = device.get("vram_total", 0) // (1024 * 1024)
                        vram_free = device.get("vram_free", 0) // (1024 * 1024)
                    else:
                        name = "No GPU"
                        vram_total = 0
                        vram_free = 0

                    return (stats, name, vram_total, vram_free)
                finally:
                    await client.disconnect()

            return loop.run_until_complete(run())

        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return ({}, "Error", 0, 0)


class ComfyUIListModelsNode:
    """
    Lists available models in ComfyUI.
    """

    MODEL_TYPES = ["checkpoints", "loras", "vae", "controlnet", "upscale_models", "embeddings"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (cls.MODEL_TYPES, {"default": "checkpoints"}),
            },
            "optional": {
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = ("JSON", "STRING", "INT")
    RETURN_NAMES = ("MODELS", "MODEL_LIST", "COUNT")
    FUNCTION = "list_models"
    CATEGORY = "API Manager/ComfyUI"

    def list_models(
        self,
        model_type: str = "checkpoints",
        host: str = "127.0.0.1",
        port: int = 8188
    ):
        try:
            from .comfyui_client import ComfyUIClient, ComfyUIConnection

            connection = ComfyUIConnection(host=host, port=port)
            client = ComfyUIClient(connection)

            loop = _get_event_loop()

            async def run():
                try:
                    await client.connect()
                    models = await client.get_models(model_type)
                    model_list = "\n".join(models) if isinstance(models, list) else str(models)
                    count = len(models) if isinstance(models, list) else 0
                    return (models, model_list, count)
                finally:
                    await client.disconnect()

            return loop.run_until_complete(run())

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return ([], "", 0)


class ComfyUIUploadImageNode:
    """
    Uploads an image to ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "filename": ("STRING", {"default": "upload.png"}),
                "subfolder": ("STRING", {"default": ""}),
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("FILENAME", "SUCCESS", "ERROR")
    FUNCTION = "upload"
    CATEGORY = "API Manager/ComfyUI"

    def upload(
        self,
        image,
        filename: str = "upload.png",
        subfolder: str = "",
        host: str = "127.0.0.1",
        port: int = 8188
    ):
        try:
            import numpy as np
            from PIL import Image as PILImage
            from io import BytesIO
            from .comfyui_client import ComfyUIClient, ComfyUIConnection

            # Convert tensor to PIL Image
            image_np = image[0].cpu().numpy()
            image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image_np)

            connection = ComfyUIConnection(host=host, port=port)
            client = ComfyUIClient(connection)

            loop = _get_event_loop()

            async def run():
                try:
                    await client.connect()
                    result = await client.upload_image(
                        pil_image,
                        filename=filename,
                        subfolder=subfolder
                    )
                    return (result.get("name", filename), True, "")
                finally:
                    await client.disconnect()

            return loop.run_until_complete(run())

        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            return ("", False, str(e))


class ComfyUIDownloadImageNode:
    """
    Downloads an image from ComfyUI.
    """

    IMAGE_TYPES = ["output", "input", "temp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": ""}),
            },
            "optional": {
                "subfolder": ("STRING", {"default": ""}),
                "image_type": (cls.IMAGE_TYPES, {"default": "output"}),
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("IMAGE", "SUCCESS", "ERROR")
    FUNCTION = "download"
    CATEGORY = "API Manager/ComfyUI"

    def download(
        self,
        filename: str,
        subfolder: str = "",
        image_type: str = "output",
        host: str = "127.0.0.1",
        port: int = 8188
    ):
        try:
            import torch
            import numpy as np
            from PIL import Image as PILImage
            from io import BytesIO
            from .comfyui_client import ComfyUIClient, ComfyUIConnection

            connection = ComfyUIConnection(host=host, port=port)
            client = ComfyUIClient(connection)

            loop = _get_event_loop()

            async def run():
                try:
                    await client.connect()
                    image_data = await client.get_image(
                        filename,
                        subfolder=subfolder,
                        image_type=image_type
                    )
                    if image_data:
                        pil_image = PILImage.open(BytesIO(image_data))
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")
                        image_np = np.array(pil_image).astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                        return (image_tensor, True, "")
                    else:
                        empty = torch.zeros((1, 64, 64, 3))
                        return (empty, False, "Image not found")
                finally:
                    await client.disconnect()

            return loop.run_until_complete(run())

        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            import torch
            empty = torch.zeros((1, 64, 64, 3))
            return (empty, False, str(e))


class ComfyUIInterruptNode:
    """
    Interrupts the currently executing workflow on ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("SUCCESS",)
    FUNCTION = "interrupt"
    CATEGORY = "API Manager/ComfyUI"
    OUTPUT_NODE = True

    def interrupt(
        self,
        trigger: bool = True,
        host: str = "127.0.0.1",
        port: int = 8188
    ):
        if not trigger:
            return (False,)

        try:
            from .comfyui_client import ComfyUIClient, ComfyUIConnection

            connection = ComfyUIConnection(host=host, port=port)
            client = ComfyUIClient(connection)

            loop = _get_event_loop()

            async def run():
                try:
                    await client.connect()
                    return await client.interrupt()
                finally:
                    await client.disconnect()

            success = loop.run_until_complete(run())
            return (success,)

        except Exception as e:
            logger.error(f"Error interrupting: {e}")
            return (False,)


class ComfyUIFreeMemoryNode:
    """
    Frees GPU memory on ComfyUI server.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "unload_models": ("BOOLEAN", {"default": True}),
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("SUCCESS",)
    FUNCTION = "free_memory"
    CATEGORY = "API Manager/ComfyUI"
    OUTPUT_NODE = True

    def free_memory(
        self,
        trigger: bool = True,
        unload_models: bool = True,
        host: str = "127.0.0.1",
        port: int = 8188
    ):
        if not trigger:
            return (False,)

        try:
            from .comfyui_client import ComfyUIClient, ComfyUIConnection

            connection = ComfyUIConnection(host=host, port=port)
            client = ComfyUIClient(connection)

            loop = _get_event_loop()

            async def run():
                try:
                    await client.connect()
                    return await client.free_memory(unload_models=unload_models)
                finally:
                    await client.disconnect()

            success = loop.run_until_complete(run())
            return (success,)

        except Exception as e:
            logger.error(f"Error freeing memory: {e}")
            return (False,)


class ComfyUIListWorkflowsNode:
    """
    Lists available workflows from ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = ("JSON", "STRING", "INT")
    RETURN_NAMES = ("WORKFLOWS", "WORKFLOW_LIST", "COUNT")
    FUNCTION = "list_workflows"
    CATEGORY = "API Manager/ComfyUI"

    def list_workflows(self, host: str = "127.0.0.1", port: int = 8188):
        try:
            from .comfyui_client import ComfyUIClient, ComfyUIConnection

            connection = ComfyUIConnection(host=host, port=port)
            client = ComfyUIClient(connection)

            loop = _get_event_loop()

            async def run():
                try:
                    await client.connect()
                    workflows = await client.list_workflows()
                    workflow_list = "\n".join(w.name for w in workflows)
                    return (
                        [w.to_dict() for w in workflows],
                        workflow_list,
                        len(workflows)
                    )
                finally:
                    await client.disconnect()

            return loop.run_until_complete(run())

        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return ([], "", 0)
