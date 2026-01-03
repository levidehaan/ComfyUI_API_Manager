"""
MCP Server for ComfyUI API Manager

Exposes ComfyUI functionality through the Model Context Protocol (MCP),
allowing AI systems to interact with ComfyUI for image generation workflows.

This server provides:
- Tools for executing workflows, managing queues, uploading images
- Resources for accessing workflows, models, and job history
- Comprehensive error handling and progress reporting
"""

import asyncio
import base64
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.types import TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None
    Context = None

from .comfyui_client import ComfyUIClient, ComfyUIConnection, JobResult, JobStatus
from .workflow_manager import WorkflowManager, Workflow
from .job_tracker import JobTracker, Job, JobStatus as TrackerJobStatus, get_job_tracker
from .settings_manager import get_settings_manager, SettingsManager

logger = logging.getLogger(__name__)


class ComfyUIMCPServer:
    """
    MCP Server that exposes ComfyUI functionality to AI systems.

    Provides tools for:
    - Workflow execution and management
    - Image generation and upload
    - Queue and job management
    - Model and system information
    """

    def __init__(
        self,
        settings_manager: Optional[SettingsManager] = None,
        job_tracker: Optional[JobTracker] = None
    ):
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP SDK not installed. Install with: pip install mcp>=1.25.0"
            )

        self.settings = settings_manager or get_settings_manager()
        self.job_tracker = job_tracker or get_job_tracker()
        self.workflow_manager = WorkflowManager()

        # Client cache for different connections
        self._clients: dict[str, ComfyUIClient] = {}

        # Create MCP server
        self.mcp = FastMCP(
            name="ComfyUI API Manager",
            version="2.0.0",
            description="Comprehensive API interface for ComfyUI image generation"
        )

        # Register tools and resources
        self._register_tools()
        self._register_resources()

    async def _get_client(self, connection_name: Optional[str] = None) -> ComfyUIClient:
        """Get or create a ComfyUI client for a connection."""
        name = connection_name or self.settings.settings.default_connection
        connection = self.settings.get_connection(name)

        if not connection:
            raise ValueError(f"Unknown connection: {name}")

        if name not in self._clients:
            self._clients[name] = ComfyUIClient(connection)
            await self._clients[name].connect()

        return self._clients[name]

    def _register_tools(self):
        """Register all MCP tools."""

        # =====================================================================
        # Workflow Execution Tools
        # =====================================================================

        @self.mcp.tool()
        async def execute_workflow(
            workflow_json: str,
            inputs: Optional[str] = None,
            connection: Optional[str] = None,
            wait_for_completion: bool = True,
            timeout: float = 600.0
        ) -> str:
            """
            Execute a ComfyUI workflow.

            Args:
                workflow_json: The workflow in API JSON format
                inputs: Optional JSON string of input overrides (e.g., {"3.inputs.text": "a cat"})
                connection: Name of the ComfyUI connection to use
                wait_for_completion: If True, wait for workflow to complete
                timeout: Maximum time to wait in seconds

            Returns:
                JSON string with job ID and results
            """
            try:
                workflow = json.loads(workflow_json)
                input_overrides = json.loads(inputs) if inputs else {}

                # Apply input overrides
                if input_overrides:
                    for key, value in input_overrides.items():
                        parts = key.split(".")
                        if len(parts) >= 2:
                            node_id = parts[0]
                            path = parts[1:]
                            if node_id in workflow:
                                target = workflow[node_id]
                                for p in path[:-1]:
                                    target = target.setdefault(p, {})
                                target[path[-1]] = value

                client = await self._get_client(connection)
                job_id = str(uuid.uuid4())

                if wait_for_completion:
                    result = await client.execute_workflow(workflow, timeout=timeout)

                    # Track job
                    job = self.job_tracker.create_job(
                        job_id=job_id,
                        prompt_id=result.prompt_id,
                        workflow_name="custom",
                        connection_name=connection or "default",
                        inputs=input_overrides
                    )
                    self.job_tracker.update_status(
                        job_id,
                        TrackerJobStatus.COMPLETED if result.status == JobStatus.COMPLETED
                        else TrackerJobStatus.FAILED,
                        error=result.error
                    )

                    return json.dumps({
                        "success": result.status == JobStatus.COMPLETED,
                        "job_id": job_id,
                        "prompt_id": result.prompt_id,
                        "status": result.status.value,
                        "outputs": result.outputs,
                        "image_count": len(result.images),
                        "error": result.error
                    })
                else:
                    prompt_id = await client.queue_prompt(workflow)
                    job = self.job_tracker.create_job(
                        job_id=job_id,
                        prompt_id=prompt_id,
                        workflow_name="custom",
                        connection_name=connection or "default",
                        inputs=input_overrides
                    )
                    return json.dumps({
                        "success": True,
                        "job_id": job_id,
                        "prompt_id": prompt_id,
                        "status": "queued"
                    })

            except Exception as e:
                logger.error(f"Error executing workflow: {e}")
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def generate_image(
            prompt: str,
            negative_prompt: str = "",
            width: int = 512,
            height: int = 512,
            steps: int = 20,
            cfg: float = 7.0,
            seed: int = -1,
            checkpoint: Optional[str] = None,
            connection: Optional[str] = None
        ) -> str:
            """
            Generate an image using a simple text-to-image workflow.

            Args:
                prompt: The positive text prompt describing the image
                negative_prompt: Negative prompt (things to avoid)
                width: Image width in pixels
                height: Image height in pixels
                steps: Number of sampling steps
                cfg: CFG scale (guidance strength)
                seed: Random seed (-1 for random)
                checkpoint: Model checkpoint to use (optional)
                connection: ComfyUI connection name

            Returns:
                JSON with generation results and image info
            """
            # Build a simple txt2img workflow
            workflow = self._build_txt2img_workflow(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed if seed >= 0 else None,
                checkpoint=checkpoint
            )

            return await execute_workflow(
                workflow_json=json.dumps(workflow),
                connection=connection,
                wait_for_completion=True
            )

        @self.mcp.tool()
        async def img2img(
            input_image_path: str,
            prompt: str,
            negative_prompt: str = "",
            denoise: float = 0.75,
            steps: int = 20,
            cfg: float = 7.0,
            seed: int = -1,
            connection: Optional[str] = None
        ) -> str:
            """
            Generate an image from an input image (img2img).

            Args:
                input_image_path: Path to the input image file
                prompt: The positive text prompt
                negative_prompt: Negative prompt
                denoise: Denoising strength (0.0-1.0)
                steps: Number of sampling steps
                cfg: CFG scale
                seed: Random seed (-1 for random)
                connection: ComfyUI connection name

            Returns:
                JSON with generation results
            """
            try:
                client = await self._get_client(connection)

                # Upload the input image
                upload_result = await client.upload_image(
                    input_image_path,
                    image_type="input"
                )
                image_name = upload_result.get("name", "")

                # Build img2img workflow
                workflow = self._build_img2img_workflow(
                    image_name=image_name,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    denoise=denoise,
                    steps=steps,
                    cfg=cfg,
                    seed=seed if seed >= 0 else None
                )

                return await execute_workflow(
                    workflow_json=json.dumps(workflow),
                    connection=connection,
                    wait_for_completion=True
                )
            except Exception as e:
                logger.error(f"Error in img2img: {e}")
                return json.dumps({"success": False, "error": str(e)})

        # =====================================================================
        # Image Management Tools
        # =====================================================================

        @self.mcp.tool()
        async def upload_image(
            image_path: str,
            subfolder: str = "",
            image_type: str = "input",
            connection: Optional[str] = None
        ) -> str:
            """
            Upload an image to ComfyUI.

            Args:
                image_path: Path to the image file
                subfolder: Subfolder in ComfyUI to upload to
                image_type: Type of image ('input', 'temp', 'output')
                connection: ComfyUI connection name

            Returns:
                JSON with upload result including filename
            """
            try:
                client = await self._get_client(connection)
                result = await client.upload_image(
                    image_path,
                    subfolder=subfolder,
                    image_type=image_type
                )
                return json.dumps({"success": True, **result})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def get_image(
            filename: str,
            subfolder: str = "",
            image_type: str = "output",
            connection: Optional[str] = None
        ) -> str:
            """
            Get an image from ComfyUI as base64.

            Args:
                filename: Name of the image file
                subfolder: Subfolder containing the image
                image_type: Type ('input', 'output', 'temp')
                connection: ComfyUI connection name

            Returns:
                JSON with base64-encoded image data
            """
            try:
                client = await self._get_client(connection)
                image_data = await client.get_image(
                    filename,
                    subfolder=subfolder,
                    image_type=image_type
                )
                if image_data:
                    return json.dumps({
                        "success": True,
                        "filename": filename,
                        "data": base64.b64encode(image_data).decode(),
                        "size": len(image_data)
                    })
                return json.dumps({"success": False, "error": "Image not found"})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        # =====================================================================
        # Queue Management Tools
        # =====================================================================

        @self.mcp.tool()
        async def get_queue_status(connection: Optional[str] = None) -> str:
            """
            Get the current queue status.

            Args:
                connection: ComfyUI connection name

            Returns:
                JSON with queue information
            """
            try:
                client = await self._get_client(connection)
                queue = await client.get_queue()
                return json.dumps({"success": True, "queue": queue})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def cancel_job(
            prompt_id: str,
            connection: Optional[str] = None
        ) -> str:
            """
            Cancel a queued or running job.

            Args:
                prompt_id: The prompt ID to cancel
                connection: ComfyUI connection name

            Returns:
                JSON with cancellation result
            """
            try:
                client = await self._get_client(connection)
                success = await client.cancel_queue_item(prompt_id)

                # Update job tracker
                job = self.job_tracker.get_job_by_prompt(prompt_id)
                if job:
                    self.job_tracker.update_status(job.job_id, TrackerJobStatus.CANCELLED)

                return json.dumps({"success": success, "prompt_id": prompt_id})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def clear_queue(connection: Optional[str] = None) -> str:
            """
            Clear all queued jobs.

            Args:
                connection: ComfyUI connection name

            Returns:
                JSON with result
            """
            try:
                client = await self._get_client(connection)
                success = await client.clear_queue()
                return json.dumps({"success": success})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def interrupt_execution(connection: Optional[str] = None) -> str:
            """
            Interrupt the currently executing workflow.

            Args:
                connection: ComfyUI connection name

            Returns:
                JSON with result
            """
            try:
                client = await self._get_client(connection)
                success = await client.interrupt()
                return json.dumps({"success": success})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        # =====================================================================
        # Job History Tools
        # =====================================================================

        @self.mcp.tool()
        async def get_job_history(
            status: Optional[str] = None,
            workflow_name: Optional[str] = None,
            limit: int = 50
        ) -> str:
            """
            Get job execution history.

            Args:
                status: Filter by status (pending, running, completed, failed)
                workflow_name: Filter by workflow name
                limit: Maximum number of jobs to return

            Returns:
                JSON with job history
            """
            try:
                status_enum = TrackerJobStatus(status) if status else None
                jobs = self.job_tracker.list_jobs(
                    status=status_enum,
                    workflow_name=workflow_name,
                    limit=limit
                )
                return json.dumps({
                    "success": True,
                    "count": len(jobs),
                    "jobs": [j.to_dict() for j in jobs]
                })
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def get_job_status(job_id: str) -> str:
            """
            Get the status of a specific job.

            Args:
                job_id: The job ID to check

            Returns:
                JSON with job details
            """
            try:
                job = self.job_tracker.get_job(job_id)
                if job:
                    return json.dumps({"success": True, "job": job.to_dict()})
                return json.dumps({"success": False, "error": "Job not found"})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def get_job_statistics() -> str:
            """
            Get job execution statistics.

            Returns:
                JSON with statistics
            """
            try:
                stats = self.job_tracker.get_statistics()
                return json.dumps({"success": True, "statistics": stats})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        # =====================================================================
        # System Information Tools
        # =====================================================================

        @self.mcp.tool()
        async def get_system_info(connection: Optional[str] = None) -> str:
            """
            Get ComfyUI system information including GPU stats.

            Args:
                connection: ComfyUI connection name

            Returns:
                JSON with system information
            """
            try:
                client = await self._get_client(connection)
                stats = await client.get_system_stats()
                return json.dumps({"success": True, "system": stats})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def list_models(
            model_type: Optional[str] = None,
            connection: Optional[str] = None
        ) -> str:
            """
            List available models in ComfyUI.

            Args:
                model_type: Type of models to list (checkpoints, loras, vae, etc.)
                connection: ComfyUI connection name

            Returns:
                JSON with model list
            """
            try:
                client = await self._get_client(connection)
                models = await client.get_models(model_type)
                return json.dumps({"success": True, "models": models})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def list_nodes(
            node_class: Optional[str] = None,
            connection: Optional[str] = None
        ) -> str:
            """
            List available node types in ComfyUI.

            Args:
                node_class: Specific node class to get info for
                connection: ComfyUI connection name

            Returns:
                JSON with node information
            """
            try:
                client = await self._get_client(connection)
                nodes = await client.get_object_info(node_class)
                return json.dumps({"success": True, "nodes": nodes})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def free_memory(connection: Optional[str] = None) -> str:
            """
            Free GPU memory by unloading models.

            Args:
                connection: ComfyUI connection name

            Returns:
                JSON with result
            """
            try:
                client = await self._get_client(connection)
                success = await client.free_memory()
                return json.dumps({"success": success})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        # =====================================================================
        # Connection Management Tools
        # =====================================================================

        @self.mcp.tool()
        async def list_connections() -> str:
            """
            List all configured ComfyUI connections.

            Returns:
                JSON with connection list
            """
            try:
                connections = self.settings.list_connections()
                return json.dumps({
                    "success": True,
                    "default": self.settings.settings.default_connection,
                    "connections": {
                        name: conn.to_dict()
                        for name, conn in connections.items()
                    }
                })
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def test_connection(connection: Optional[str] = None) -> str:
            """
            Test a ComfyUI connection.

            Args:
                connection: Connection name to test

            Returns:
                JSON with connection status
            """
            try:
                client = await self._get_client(connection)
                is_connected = await client.is_connected()
                if is_connected:
                    stats = await client.get_system_stats()
                    return json.dumps({
                        "success": True,
                        "connected": True,
                        "system": stats
                    })
                return json.dumps({
                    "success": True,
                    "connected": False
                })
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "connected": False,
                    "error": str(e)
                })

        # =====================================================================
        # Workflow Management Tools
        # =====================================================================

        @self.mcp.tool()
        async def list_workflows(connection: Optional[str] = None) -> str:
            """
            List available workflows.

            Args:
                connection: ComfyUI connection name

            Returns:
                JSON with workflow list
            """
            try:
                client = await self._get_client(connection)
                workflows = await client.list_workflows()
                return json.dumps({
                    "success": True,
                    "workflows": [w.to_dict() for w in workflows]
                })
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def get_workflow_templates(connection: Optional[str] = None) -> str:
            """
            Get available workflow templates.

            Args:
                connection: ComfyUI connection name

            Returns:
                JSON with template list
            """
            try:
                client = await self._get_client(connection)
                templates = await client.get_workflow_templates()
                return json.dumps({"success": True, "templates": templates})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def validate_workflow(workflow_json: str) -> str:
            """
            Validate a workflow structure.

            Args:
                workflow_json: The workflow JSON to validate

            Returns:
                JSON with validation result
            """
            try:
                workflow = json.loads(workflow_json)
                is_valid, errors = self.workflow_manager.validate_workflow(workflow)
                return json.dumps({
                    "success": True,
                    "valid": is_valid,
                    "errors": errors
                })
            except json.JSONDecodeError as e:
                return json.dumps({
                    "success": False,
                    "valid": False,
                    "errors": [f"Invalid JSON: {e}"]
                })
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        @self.mcp.tool()
        async def analyze_workflow(workflow_json: str) -> str:
            """
            Analyze a workflow to find inputs and outputs.

            Args:
                workflow_json: The workflow JSON to analyze

            Returns:
                JSON with workflow analysis
            """
            try:
                workflow_data = json.loads(workflow_json)
                workflow = self.workflow_manager.parse_workflow(workflow_data)
                return json.dumps({
                    "success": True,
                    "name": workflow.name,
                    "inputs": [i.to_dict() for i in workflow.inputs],
                    "outputs": [o.to_dict() for o in workflow.outputs],
                    "node_count": len(workflow_data)
                })
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

    def _register_resources(self):
        """Register MCP resources."""

        @self.mcp.resource("comfyui://connections")
        async def get_connections_resource() -> str:
            """Get all configured connections."""
            connections = self.settings.list_connections()
            return json.dumps({
                "default": self.settings.settings.default_connection,
                "connections": {
                    name: conn.to_dict()
                    for name, conn in connections.items()
                }
            })

        @self.mcp.resource("comfyui://jobs/recent")
        async def get_recent_jobs_resource() -> str:
            """Get recent job history."""
            jobs = self.job_tracker.list_jobs(limit=50)
            return json.dumps({
                "count": len(jobs),
                "jobs": [j.to_dict() for j in jobs]
            })

        @self.mcp.resource("comfyui://jobs/statistics")
        async def get_statistics_resource() -> str:
            """Get job statistics."""
            return json.dumps(self.job_tracker.get_statistics())

        @self.mcp.resource("comfyui://settings")
        async def get_settings_resource() -> str:
            """Get current plugin settings."""
            return json.dumps(self.settings.export_settings())

    def _build_txt2img_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        checkpoint: Optional[str] = None
    ) -> dict:
        """Build a simple txt2img workflow."""
        import random
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": seed,
                    "steps": steps
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": checkpoint or "v1-5-pruned-emaonly.safetensors"
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": height,
                    "width": width
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": prompt
                },
                "_meta": {"title": "Positive Prompt"}
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": negative_prompt
                },
                "_meta": {"title": "Negative Prompt"}
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                }
            }
        }

    def _build_img2img_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "",
        denoise: float = 0.75,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        checkpoint: Optional[str] = None
    ) -> dict:
        """Build a simple img2img workflow."""
        import random
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_name
                }
            },
            "2": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["4", 2]
                }
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": denoise,
                    "latent_image": ["2", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": seed,
                    "steps": steps
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": checkpoint or "v1-5-pruned-emaonly.safetensors"
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": prompt
                },
                "_meta": {"title": "Positive Prompt"}
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": negative_prompt
                },
                "_meta": {"title": "Negative Prompt"}
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                }
            }
        }

    async def cleanup(self):
        """Clean up resources."""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()

    def run(self, transport: str = "stdio"):
        """Run the MCP server."""
        self.mcp.run(transport=transport)


def create_mcp_server(
    settings_manager: Optional[SettingsManager] = None
) -> Optional[ComfyUIMCPServer]:
    """Create an MCP server if MCP is available and enabled."""
    if not MCP_AVAILABLE:
        logger.warning("MCP SDK not available. Install with: pip install mcp>=1.25.0")
        return None

    settings = settings_manager or get_settings_manager()
    if not settings.is_mcp_enabled():
        logger.info("MCP server is disabled in settings")
        return None

    return ComfyUIMCPServer(settings_manager=settings)


def main():
    """Entry point for running the MCP server standalone."""
    if not MCP_AVAILABLE:
        print("Error: MCP SDK not installed. Install with: pip install mcp>=1.25.0")
        sys.exit(1)

    # Force enable MCP for standalone execution
    settings = get_settings_manager()
    settings.set_mcp_enabled(True)

    server = ComfyUIMCPServer(settings_manager=settings)

    transport = settings.get_mcp_settings().transport
    print(f"Starting ComfyUI MCP Server (transport: {transport})")

    try:
        server.run(transport=transport)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        asyncio.get_event_loop().run_until_complete(server.cleanup())


if __name__ == "__main__":
    main()
