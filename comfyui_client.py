"""
ComfyUI API Client Module

A comprehensive async client for interacting with ComfyUI's REST and WebSocket APIs.
Supports workflow execution, image upload/download, queue management, and real-time
progress monitoring.

Based on ComfyUI API documentation and best practices for 2025/2026.
"""

import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a workflow execution job."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ComfyUIConnection:
    """Connection configuration for ComfyUI server."""
    host: str = "127.0.0.1"
    port: int = 8188
    use_ssl: bool = False
    client_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def base_url(self) -> str:
        protocol = "https" if self.use_ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        protocol = "wss" if self.use_ssl else "ws"
        return f"{protocol}://{self.host}:{self.port}/ws?clientId={self.client_id}"

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "use_ssl": self.use_ssl,
            "client_id": self.client_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ComfyUIConnection":
        return cls(**data)


@dataclass
class JobResult:
    """Result of a workflow execution."""
    prompt_id: str
    status: JobStatus
    outputs: dict = field(default_factory=dict)
    images: list = field(default_factory=list)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_node: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "status": self.status.value,
            "outputs": self.outputs,
            "images": [img if isinstance(img, str) else base64.b64encode(img).decode()
                       for img in self.images],
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "current_node": self.current_node
        }


@dataclass
class WorkflowInfo:
    """Information about a saved workflow."""
    name: str
    path: str
    modified: Optional[datetime] = None
    size: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "modified": self.modified.isoformat() if self.modified else None,
            "size": self.size
        }


class ComfyUIClient:
    """
    Async client for ComfyUI API.

    Provides comprehensive access to ComfyUI's REST and WebSocket APIs including:
    - Workflow execution with real-time progress monitoring
    - Image upload and download
    - Queue management
    - System information
    - Model and node information
    - User data and workflow management
    """

    def __init__(self, connection: Optional[ComfyUIConnection] = None):
        self.connection = connection or ComfyUIConnection()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._message_handlers: list[Callable] = []
        self._progress_callbacks: dict[str, Callable] = {}

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self) -> None:
        """Establish connection to ComfyUI server."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        logger.info(f"Connected to ComfyUI at {self.connection.base_url}")

    async def disconnect(self) -> None:
        """Close all connections."""
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Disconnected from ComfyUI")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure we have an active session."""
        if self._session is None or self._session.closed:
            await self.connect()
        return self._session

    # =========================================================================
    # WebSocket Methods
    # =========================================================================

    async def connect_websocket(self) -> None:
        """Connect to WebSocket for real-time updates."""
        session = await self._ensure_session()
        self._ws = await session.ws_connect(self.connection.ws_url)
        self._ws_task = asyncio.create_task(self._ws_listener())
        logger.info("WebSocket connected")

    async def _ws_listener(self) -> None:
        """Listen for WebSocket messages."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_ws_message(data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary data (usually preview images)
                    for handler in self._message_handlers:
                        await handler({"type": "binary", "data": msg.data})
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _handle_ws_message(self, data: dict) -> None:
        """Process WebSocket message."""
        msg_type = data.get("type")
        msg_data = data.get("data", {})

        # Handle progress updates
        if msg_type == "progress":
            prompt_id = msg_data.get("prompt_id")
            if prompt_id and prompt_id in self._progress_callbacks:
                progress = msg_data.get("value", 0) / max(msg_data.get("max", 1), 1)
                await self._progress_callbacks[prompt_id](progress, msg_data)

        elif msg_type == "executing":
            prompt_id = msg_data.get("prompt_id")
            node = msg_data.get("node")
            if prompt_id and prompt_id in self._progress_callbacks:
                await self._progress_callbacks[prompt_id](None, {"node": node, "type": "executing"})

        # Notify all handlers
        for handler in self._message_handlers:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")

    def add_message_handler(self, handler: Callable) -> None:
        """Add a handler for WebSocket messages."""
        self._message_handlers.append(handler)

    def remove_message_handler(self, handler: Callable) -> None:
        """Remove a message handler."""
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)

    # =========================================================================
    # Queue and Prompt Methods
    # =========================================================================

    async def queue_prompt(
        self,
        workflow: dict,
        extra_data: Optional[dict] = None
    ) -> str:
        """
        Queue a workflow for execution.

        Args:
            workflow: The workflow JSON (API format)
            extra_data: Optional extra data to include

        Returns:
            The prompt_id for tracking execution
        """
        session = await self._ensure_session()

        payload = {
            "prompt": workflow,
            "client_id": self.connection.client_id
        }
        if extra_data:
            payload["extra_data"] = extra_data

        async with session.post(
            f"{self.connection.base_url}/prompt",
            json=payload
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Failed to queue prompt: {error_text}")

            result = await resp.json()
            if "error" in result:
                raise Exception(f"Prompt error: {result['error']}")

            prompt_id = result.get("prompt_id")
            logger.info(f"Queued prompt: {prompt_id}")
            return prompt_id

    async def get_queue(self) -> dict:
        """Get current queue status."""
        session = await self._ensure_session()
        async with session.get(f"{self.connection.base_url}/queue") as resp:
            return await resp.json()

    async def get_history(self, prompt_id: Optional[str] = None, max_items: int = 100) -> dict:
        """
        Get execution history.

        Args:
            prompt_id: Specific prompt ID to get history for
            max_items: Maximum number of history items to return
        """
        session = await self._ensure_session()
        url = f"{self.connection.base_url}/history"
        if prompt_id:
            url = f"{url}/{prompt_id}"
        else:
            url = f"{url}?max_items={max_items}"

        async with session.get(url) as resp:
            return await resp.json()

    async def delete_history(self, prompt_ids: Optional[list[str]] = None, clear: bool = False) -> bool:
        """
        Delete history entries.

        Args:
            prompt_ids: Specific prompt IDs to delete
            clear: If True, clear all history
        """
        session = await self._ensure_session()
        payload = {}
        if clear:
            payload["clear"] = True
        elif prompt_ids:
            payload["delete"] = prompt_ids

        async with session.post(
            f"{self.connection.base_url}/history",
            json=payload
        ) as resp:
            return resp.status == 200

    async def cancel_queue_item(self, prompt_id: str) -> bool:
        """Cancel a queued item."""
        session = await self._ensure_session()
        async with session.post(
            f"{self.connection.base_url}/queue",
            json={"delete": [prompt_id]}
        ) as resp:
            return resp.status == 200

    async def clear_queue(self) -> bool:
        """Clear all queued items."""
        session = await self._ensure_session()
        async with session.post(
            f"{self.connection.base_url}/queue",
            json={"clear": True}
        ) as resp:
            return resp.status == 200

    async def interrupt(self) -> bool:
        """Interrupt currently executing workflow."""
        session = await self._ensure_session()
        async with session.post(f"{self.connection.base_url}/interrupt") as resp:
            return resp.status == 200

    # =========================================================================
    # Workflow Execution with Tracking
    # =========================================================================

    async def execute_workflow(
        self,
        workflow: dict,
        progress_callback: Optional[Callable] = None,
        timeout: float = 600.0
    ) -> JobResult:
        """
        Execute a workflow and wait for completion.

        Args:
            workflow: The workflow JSON (API format)
            progress_callback: Optional callback for progress updates
            timeout: Maximum time to wait for completion

        Returns:
            JobResult with outputs and images
        """
        # Ensure WebSocket is connected
        if not self._ws:
            await self.connect_websocket()

        result = JobResult(
            prompt_id="",
            status=JobStatus.PENDING,
            started_at=datetime.now()
        )

        completion_event = asyncio.Event()

        async def handle_completion(data: dict):
            if data.get("type") == "executing":
                node = data.get("data", {}).get("node")
                if data.get("data", {}).get("prompt_id") == result.prompt_id:
                    result.current_node = node
                    if node is None:
                        # Execution complete
                        result.status = JobStatus.COMPLETED
                        result.completed_at = datetime.now()
                        completion_event.set()

            elif data.get("type") == "execution_error":
                if data.get("data", {}).get("prompt_id") == result.prompt_id:
                    result.status = JobStatus.FAILED
                    result.error = str(data.get("data", {}).get("exception_message", "Unknown error"))
                    result.completed_at = datetime.now()
                    completion_event.set()

        self.add_message_handler(handle_completion)

        if progress_callback:
            async def wrapped_progress(progress, data):
                result.progress = progress if progress is not None else result.progress
                await progress_callback(result)
            self._progress_callbacks[result.prompt_id] = wrapped_progress

        try:
            # Queue the prompt
            result.prompt_id = await self.queue_prompt(workflow)
            result.status = JobStatus.QUEUED

            if progress_callback:
                self._progress_callbacks[result.prompt_id] = wrapped_progress

            # Wait for completion
            try:
                await asyncio.wait_for(completion_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                result.status = JobStatus.FAILED
                result.error = "Execution timeout"
                result.completed_at = datetime.now()

            # Get outputs from history
            if result.status == JobStatus.COMPLETED:
                history = await self.get_history(result.prompt_id)
                if result.prompt_id in history:
                    outputs = history[result.prompt_id].get("outputs", {})
                    result.outputs = outputs

                    # Extract images
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for img_info in node_output["images"]:
                                img_data = await self.get_image(
                                    img_info["filename"],
                                    img_info.get("subfolder", ""),
                                    img_info.get("type", "output")
                                )
                                if img_data:
                                    result.images.append(img_data)

            return result

        finally:
            self.remove_message_handler(handle_completion)
            if result.prompt_id in self._progress_callbacks:
                del self._progress_callbacks[result.prompt_id]

    # =========================================================================
    # Image Methods
    # =========================================================================

    async def upload_image(
        self,
        image: bytes | Image.Image | str | Path,
        filename: Optional[str] = None,
        subfolder: str = "",
        image_type: str = "input",
        overwrite: bool = False
    ) -> dict:
        """
        Upload an image to ComfyUI.

        Args:
            image: Image data (bytes, PIL Image, or file path)
            filename: Name for the uploaded file
            subfolder: Subfolder to upload to
            image_type: Type of image ('input', 'temp', 'output')
            overwrite: Whether to overwrite existing file

        Returns:
            Dict with upload result including filename
        """
        session = await self._ensure_session()

        # Convert image to bytes
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_bytes = f.read()
            if filename is None:
                filename = Path(image).name
        elif isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            if filename is None:
                filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        else:
            image_bytes = image
            if filename is None:
                filename = f"upload_{uuid.uuid4().hex[:8]}.png"

        # Prepare form data
        data = aiohttp.FormData()
        data.add_field("image", image_bytes, filename=filename, content_type="image/png")
        data.add_field("subfolder", subfolder)
        data.add_field("type", image_type)
        data.add_field("overwrite", str(overwrite).lower())

        async with session.post(
            f"{self.connection.base_url}/upload/image",
            data=data
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to upload image: {await resp.text()}")
            return await resp.json()

    async def upload_mask(
        self,
        mask: bytes | Image.Image | str | Path,
        filename: Optional[str] = None,
        subfolder: str = "",
        overwrite: bool = False,
        original_ref: Optional[dict] = None
    ) -> dict:
        """Upload a mask image."""
        session = await self._ensure_session()

        # Convert mask to bytes
        if isinstance(mask, (str, Path)):
            with open(mask, "rb") as f:
                mask_bytes = f.read()
            if filename is None:
                filename = Path(mask).name
        elif isinstance(mask, Image.Image):
            buffer = BytesIO()
            mask.save(buffer, format="PNG")
            mask_bytes = buffer.getvalue()
            if filename is None:
                filename = f"mask_{uuid.uuid4().hex[:8]}.png"
        else:
            mask_bytes = mask
            if filename is None:
                filename = f"mask_{uuid.uuid4().hex[:8]}.png"

        data = aiohttp.FormData()
        data.add_field("image", mask_bytes, filename=filename, content_type="image/png")
        data.add_field("subfolder", subfolder)
        data.add_field("type", "input")
        data.add_field("overwrite", str(overwrite).lower())
        if original_ref:
            data.add_field("original_ref", json.dumps(original_ref))

        async with session.post(
            f"{self.connection.base_url}/upload/mask",
            data=data
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to upload mask: {await resp.text()}")
            return await resp.json()

    async def get_image(
        self,
        filename: str,
        subfolder: str = "",
        image_type: str = "output",
        preview: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Get an image from ComfyUI.

        Args:
            filename: Name of the image file
            subfolder: Subfolder containing the image
            image_type: Type ('input', 'output', 'temp')
            preview: Preview format if needed

        Returns:
            Image data as bytes
        """
        session = await self._ensure_session()

        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": image_type
        }
        if preview:
            params["preview"] = preview

        async with session.get(
            f"{self.connection.base_url}/view",
            params=params
        ) as resp:
            if resp.status != 200:
                logger.error(f"Failed to get image: {await resp.text()}")
                return None
            return await resp.read()

    # =========================================================================
    # System Information
    # =========================================================================

    async def get_system_stats(self) -> dict:
        """Get system statistics including GPU info."""
        session = await self._ensure_session()
        async with session.get(f"{self.connection.base_url}/system_stats") as resp:
            return await resp.json()

    async def get_object_info(self, node_class: Optional[str] = None) -> dict:
        """
        Get information about available nodes.

        Args:
            node_class: Specific node class to get info for
        """
        session = await self._ensure_session()
        url = f"{self.connection.base_url}/object_info"
        if node_class:
            url = f"{url}/{node_class}"

        async with session.get(url) as resp:
            return await resp.json()

    async def get_embeddings(self) -> list[str]:
        """Get list of available embeddings."""
        session = await self._ensure_session()
        async with session.get(f"{self.connection.base_url}/embeddings") as resp:
            return await resp.json()

    async def get_extensions(self) -> list[dict]:
        """Get list of installed extensions."""
        session = await self._ensure_session()
        async with session.get(f"{self.connection.base_url}/extensions") as resp:
            return await resp.json()

    async def get_models(self, folder: Optional[str] = None) -> dict | list:
        """
        Get available models.

        Args:
            folder: Specific model folder to list
        """
        session = await self._ensure_session()
        if folder:
            async with session.get(f"{self.connection.base_url}/models/{folder}") as resp:
                return await resp.json()
        else:
            async with session.get(f"{self.connection.base_url}/models") as resp:
                return await resp.json()

    async def get_features(self) -> dict:
        """Get server features and capabilities."""
        session = await self._ensure_session()
        async with session.get(f"{self.connection.base_url}/features") as resp:
            return await resp.json()

    async def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> bool:
        """
        Free GPU memory.

        Args:
            unload_models: Whether to unload models
            free_memory: Whether to free CUDA memory
        """
        session = await self._ensure_session()
        async with session.post(
            f"{self.connection.base_url}/free",
            json={"unload_models": unload_models, "free_memory": free_memory}
        ) as resp:
            return resp.status == 200

    # =========================================================================
    # User Data and Workflows
    # =========================================================================

    async def get_user_data_files(self, directory: str = "") -> list[str]:
        """
        Get list of user data files.

        Args:
            directory: Subdirectory to list
        """
        session = await self._ensure_session()
        params = {"dir": directory} if directory else {}
        async with session.get(
            f"{self.connection.base_url}/userdata",
            params=params
        ) as resp:
            if resp.status != 200:
                return []
            return await resp.json()

    async def get_workflow_templates(self) -> list[dict]:
        """Get available workflow templates."""
        session = await self._ensure_session()
        async with session.get(f"{self.connection.base_url}/workflow_templates") as resp:
            if resp.status != 200:
                return []
            return await resp.json()

    async def list_workflows(self) -> list[WorkflowInfo]:
        """
        List saved workflows from user data.

        Returns:
            List of WorkflowInfo objects
        """
        workflows = []

        # Get files from userdata/workflows directory
        try:
            files = await self.get_user_data_files("workflows")
            for file in files:
                if file.endswith(".json"):
                    workflows.append(WorkflowInfo(
                        name=file.replace(".json", ""),
                        path=f"workflows/{file}"
                    ))
        except Exception as e:
            logger.warning(f"Could not list user workflows: {e}")

        # Also get workflow templates
        try:
            templates = await self.get_workflow_templates()
            for template in templates:
                workflows.append(WorkflowInfo(
                    name=template.get("name", "Unknown"),
                    path=template.get("path", ""),
                ))
        except Exception as e:
            logger.warning(f"Could not list workflow templates: {e}")

        return workflows

    async def is_connected(self) -> bool:
        """Check if server is reachable."""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.connection.base_url}/system_stats",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except Exception:
            return False


# Synchronous wrapper for non-async contexts
class ComfyUIClientSync:
    """Synchronous wrapper for ComfyUIClient."""

    def __init__(self, connection: Optional[ComfyUIConnection] = None):
        self._async_client = ComfyUIClient(connection)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _run(self, coro):
        return self._get_loop().run_until_complete(coro)

    def connect(self):
        return self._run(self._async_client.connect())

    def disconnect(self):
        return self._run(self._async_client.disconnect())

    def queue_prompt(self, workflow: dict) -> str:
        return self._run(self._async_client.queue_prompt(workflow))

    def get_queue(self) -> dict:
        return self._run(self._async_client.get_queue())

    def get_history(self, prompt_id: Optional[str] = None) -> dict:
        return self._run(self._async_client.get_history(prompt_id))

    def execute_workflow(self, workflow: dict, timeout: float = 600.0) -> JobResult:
        return self._run(self._async_client.execute_workflow(workflow, timeout=timeout))

    def upload_image(self, image, filename: Optional[str] = None, **kwargs) -> dict:
        return self._run(self._async_client.upload_image(image, filename, **kwargs))

    def get_image(self, filename: str, **kwargs) -> Optional[bytes]:
        return self._run(self._async_client.get_image(filename, **kwargs))

    def get_system_stats(self) -> dict:
        return self._run(self._async_client.get_system_stats())

    def get_object_info(self, node_class: Optional[str] = None) -> dict:
        return self._run(self._async_client.get_object_info(node_class))

    def get_models(self, folder: Optional[str] = None) -> dict | list:
        return self._run(self._async_client.get_models(folder))

    def list_workflows(self) -> list[WorkflowInfo]:
        return self._run(self._async_client.list_workflows())

    def is_connected(self) -> bool:
        return self._run(self._async_client.is_connected())

    def interrupt(self) -> bool:
        return self._run(self._async_client.interrupt())

    def free_memory(self) -> bool:
        return self._run(self._async_client.free_memory())
