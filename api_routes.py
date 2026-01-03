"""
ComfyUI API Manager - Server Routes

Adds custom API endpoints to ComfyUI's PromptServer for:
- Listing available workflows
- Analyzing workflow inputs/outputs
- Executing workflows with inputs (including images)
- Managing job queue and history

These routes are registered with ComfyUI's aiohttp server on plugin load.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from aiohttp import web
from PIL import Image

logger = logging.getLogger("comfyui_api_manager")


def find_workflow_path(name: str) -> Optional[Path]:
    """
    Find a workflow file by name, checking multiple locations.

    ComfyUI stores workflows in user/default/workflows/ (new)
    or user/workflows/ (legacy).

    Returns the Path if found, None otherwise.
    """
    try:
        import folder_paths
        user_dir = folder_paths.get_user_directory()

        # Possible workflow directories
        possible_dirs = [
            Path(user_dir) / "default" / "workflows",  # New ComfyUI location
            Path(user_dir) / "workflows",               # Legacy location
        ]

        # Try each directory
        for workflows_dir in possible_dirs:
            if not workflows_dir.exists():
                continue

            # Try exact name with .json extension
            workflow_path = workflows_dir / f"{name}.json"
            if workflow_path.exists():
                return workflow_path

            # Try without extension (in case name already has .json)
            if name.endswith(".json"):
                workflow_path = workflows_dir / name
                if workflow_path.exists():
                    return workflow_path

        return None
    except Exception as e:
        logger.error(f"Error finding workflow {name}: {e}")
        return None


# Global state
_plugin_settings = None
_job_tracker = None
_mcp_server_process = None


def get_plugin_settings():
    """Get plugin settings, initializing if needed."""
    global _plugin_settings
    if _plugin_settings is None:
        from .settings_manager import get_settings_manager
        _plugin_settings = get_settings_manager()
    return _plugin_settings


def get_job_tracker():
    """Get job tracker, initializing if needed."""
    global _job_tracker
    if _job_tracker is None:
        from .job_tracker import get_job_tracker as get_tracker
        settings = get_plugin_settings()
        history_file = Path(settings.base_dir) / "job_history.json"
        _job_tracker = get_tracker(history_file)
    return _job_tracker


class WorkflowAnalyzer:
    """
    Analyzes ComfyUI workflows to detect inputs and outputs.
    """

    # Node types that typically accept user inputs
    INPUT_NODES = {
        # Text inputs
        "CLIPTextEncode": {
            "inputs": ["text"],
            "type": "text",
            "description": "Text prompt for image generation"
        },
        "CLIPTextEncodeSDXL": {
            "inputs": ["text_g", "text_l"],
            "type": "text",
            "description": "SDXL text prompts"
        },

        # Image inputs
        "LoadImage": {
            "inputs": ["image"],
            "type": "image",
            "description": "Input image file"
        },
        "LoadImageMask": {
            "inputs": ["image"],
            "type": "image",
            "description": "Input mask image"
        },

        # Numeric inputs
        "KSampler": {
            "inputs": ["seed", "steps", "cfg", "denoise"],
            "type": "number",
            "description": "Sampler parameters"
        },
        "KSamplerAdvanced": {
            "inputs": ["seed", "steps", "cfg", "start_at_step", "end_at_step"],
            "type": "number",
            "description": "Advanced sampler parameters"
        },
        "EmptyLatentImage": {
            "inputs": ["width", "height", "batch_size"],
            "type": "number",
            "description": "Output image dimensions"
        },

        # Model selection
        "CheckpointLoaderSimple": {
            "inputs": ["ckpt_name"],
            "type": "model",
            "description": "Checkpoint model selection"
        },
        "LoraLoader": {
            "inputs": ["lora_name", "strength_model", "strength_clip"],
            "type": "model",
            "description": "LoRA model and strength"
        },
        "VAELoader": {
            "inputs": ["vae_name"],
            "type": "model",
            "description": "VAE model selection"
        },
        "ControlNetLoader": {
            "inputs": ["control_net_name"],
            "type": "model",
            "description": "ControlNet model selection"
        },
    }

    # Node types that produce outputs
    OUTPUT_NODES = {
        "SaveImage": {"type": "image", "description": "Saves generated image"},
        "PreviewImage": {"type": "image", "description": "Preview generated image"},
        "SaveImageWebsocket": {"type": "image", "description": "Streams image via WebSocket"},
    }

    @classmethod
    def analyze(cls, workflow: dict) -> dict:
        """
        Analyze a workflow to extract its inputs and outputs.

        Args:
            workflow: ComfyUI workflow in API format

        Returns:
            Dict with inputs, outputs, and metadata
        """
        inputs = []
        outputs = []
        node_info = {}

        for node_id, node_data in workflow.items():
            class_type = node_data.get("class_type", "")
            node_inputs = node_data.get("inputs", {})
            meta = node_data.get("_meta", {})
            title = meta.get("title", class_type)

            node_info[node_id] = {
                "class_type": class_type,
                "title": title
            }

            # Check for input nodes
            if class_type in cls.INPUT_NODES:
                node_config = cls.INPUT_NODES[class_type]
                for input_name in node_config["inputs"]:
                    if input_name in node_inputs:
                        value = node_inputs[input_name]
                        # Skip if it's a connection to another node
                        if isinstance(value, list):
                            continue

                        inputs.append({
                            "node_id": node_id,
                            "node_title": title,
                            "node_type": class_type,
                            "input_name": input_name,
                            "input_type": cls._get_input_type(class_type, input_name, value),
                            "current_value": value,
                            "description": node_config["description"],
                            "required": True
                        })

            # Check for output nodes
            if class_type in cls.OUTPUT_NODES:
                output_config = cls.OUTPUT_NODES[class_type]
                outputs.append({
                    "node_id": node_id,
                    "node_title": title,
                    "node_type": class_type,
                    "output_type": output_config["type"],
                    "description": output_config["description"]
                })

        return {
            "inputs": inputs,
            "outputs": outputs,
            "node_count": len(workflow),
            "has_image_input": any(i["input_type"] == "image" for i in inputs),
            "has_text_input": any(i["input_type"] == "text" for i in inputs),
            "produces_image": any(o["output_type"] == "image" for o in outputs)
        }

    @classmethod
    def _get_input_type(cls, class_type: str, input_name: str, value: Any) -> str:
        """Determine the type of an input based on context."""
        # Image inputs
        if class_type in ("LoadImage", "LoadImageMask"):
            return "image"

        # Text inputs
        if input_name in ("text", "text_g", "text_l", "prompt", "negative_prompt"):
            return "text"

        # Model inputs
        if input_name.endswith("_name") and "name" in input_name:
            return "model"

        # Numeric inputs
        if isinstance(value, (int, float)):
            if input_name == "seed":
                return "seed"
            return "number"

        # Boolean
        if isinstance(value, bool):
            return "boolean"

        # Default to string
        return "string"


def register_routes(server):
    """
    Register API routes with ComfyUI's PromptServer.

    Args:
        server: The PromptServer instance
    """
    routes = server.routes

    # ==========================================================================
    # API Manager Info
    # ==========================================================================

    @routes.get("/api-manager/info")
    async def get_info(request):
        """Get API Manager plugin information."""
        import folder_paths

        settings = get_plugin_settings()
        user_dir = folder_paths.get_user_directory()

        # Show workflow directories being searched
        workflow_dirs = []
        for subdir in ["default/workflows", "workflows"]:
            path = Path(user_dir) / subdir
            workflow_dirs.append({
                "path": str(path),
                "exists": path.exists(),
                "files": len(list(path.glob("*.json"))) if path.exists() else 0
            })

        return web.json_response({
            "name": "ComfyUI API Manager",
            "version": "2.0.0",
            "mcp_enabled": settings.is_mcp_enabled(),
            "mcp_port": settings.get_mcp_settings().port if settings.is_mcp_enabled() else None,
            "user_directory": str(user_dir),
            "workflow_directories": workflow_dirs,
            "endpoints": [
                "/api-manager/info",
                "/api-manager/settings",
                "/api-manager/workflows",
                "/api-manager/workflows/{name}",
                "/api-manager/workflows/{name}/analyze",
                "/api-manager/workflows/{name}/execute",
                "/api-manager/jobs",
                "/api-manager/jobs/{job_id}",
                "/api-manager/mcp/start",
                "/api-manager/mcp/stop",
            ]
        })

    # ==========================================================================
    # Settings Endpoints
    # ==========================================================================

    @routes.get("/api-manager/settings")
    async def get_settings(request):
        """Get current plugin settings."""
        settings = get_plugin_settings()
        return web.json_response(settings.export_settings())

    @routes.post("/api-manager/settings")
    async def update_settings(request):
        """Update plugin settings."""
        try:
            data = await request.json()
            settings = get_plugin_settings()

            if "mcp" in data:
                mcp = data["mcp"]
                settings.update_mcp_settings(
                    enabled=mcp.get("enabled"),
                    host=mcp.get("host"),
                    port=mcp.get("port"),
                    transport=mcp.get("transport")
                )

            if "logging" in data:
                log = data["logging"]
                settings.update_logging_settings(
                    level=log.get("level"),
                    log_to_file=log.get("log_to_file"),
                    log_file=log.get("log_file")
                )

            return web.json_response({
                "success": True,
                "settings": settings.export_settings()
            })

        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=400)

    # ==========================================================================
    # Workflow Endpoints
    # ==========================================================================

    @routes.get("/api-manager/workflows")
    async def list_workflows(request):
        """
        List all available workflows.

        Returns workflows from:
        - ComfyUI's user/workflows directory
        - Workflow templates
        """
        try:
            import folder_paths

            workflows = []

            # Get user workflows
            # ComfyUI stores workflows in user/default/workflows/
            user_dir = folder_paths.get_user_directory()

            # Check multiple possible locations
            possible_dirs = [
                Path(user_dir) / "default" / "workflows",  # New ComfyUI location
                Path(user_dir) / "workflows",               # Legacy location
            ]

            for workflows_dir in possible_dirs:
                if workflows_dir.exists():
                    logger.debug(f"Scanning workflows in: {workflows_dir}")
                    for file in workflows_dir.glob("*.json"):
                        try:
                            stat = file.stat()
                            workflows.append({
                                "name": file.stem,
                                "path": str(file),
                                "source": "user",
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                            })
                        except Exception as e:
                            logger.warning(f"Error reading workflow {file}: {e}")

            # Get default workflows
            default_dir = Path(folder_paths.base_path) / "web" / "scripts" / "defaultWorkflows"
            if default_dir.exists():
                for file in default_dir.glob("*.json"):
                    try:
                        workflows.append({
                            "name": file.stem,
                            "path": str(file),
                            "source": "default",
                            "size": file.stat().st_size
                        })
                    except Exception:
                        pass

            return web.json_response({
                "success": True,
                "count": len(workflows),
                "workflows": workflows
            })

        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.get("/api-manager/workflows/{name}")
    async def get_workflow(request):
        """Get a specific workflow by name."""
        try:
            name = request.match_info["name"]

            # Find workflow using helper
            workflow_path = find_workflow_path(name)
            if not workflow_path:
                return web.json_response(
                    {"success": False, "error": f"Workflow not found: {name}"},
                    status=404
                )

            with open(workflow_path, "r") as f:
                workflow = json.load(f)

            return web.json_response({
                "success": True,
                "name": name,
                "path": str(workflow_path),
                "workflow": workflow
            })

        except Exception as e:
            logger.error(f"Error getting workflow: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.get("/api-manager/workflows/{name}/analyze")
    async def analyze_workflow(request):
        """
        Analyze a workflow to determine its inputs and outputs.

        Returns information about:
        - Required inputs (text prompts, images, numbers)
        - Expected outputs (images, etc.)
        - Input types and current values
        """
        try:
            name = request.match_info["name"]

            # Find workflow using helper
            workflow_path = find_workflow_path(name)
            if not workflow_path:
                return web.json_response(
                    {"success": False, "error": f"Workflow not found: {name}"},
                    status=404
                )

            with open(workflow_path, "r") as f:
                workflow = json.load(f)

            # Handle both API format and graph format
            if "nodes" in workflow:
                # This is graph format, need to convert to API format
                return web.json_response({
                    "success": False,
                    "error": "Workflow is in graph format. Please save in API format."
                }, status=400)

            analysis = WorkflowAnalyzer.analyze(workflow)

            return web.json_response({
                "success": True,
                "name": name,
                "analysis": analysis
            })

        except Exception as e:
            logger.error(f"Error analyzing workflow: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.post("/api-manager/workflows/{name}/analyze")
    async def analyze_workflow_post(request):
        """
        Analyze a workflow provided in the request body.

        Body should contain:
        - workflow: The workflow JSON (API format)
        """
        try:
            data = await request.json()
            workflow = data.get("workflow", {})

            if not workflow:
                return web.json_response(
                    {"success": False, "error": "No workflow provided"},
                    status=400
                )

            analysis = WorkflowAnalyzer.analyze(workflow)

            return web.json_response({
                "success": True,
                "analysis": analysis
            })

        except Exception as e:
            logger.error(f"Error analyzing workflow: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ==========================================================================
    # Workflow Execution
    # ==========================================================================

    @routes.post("/api-manager/workflows/{name}/execute")
    async def execute_workflow(request):
        """
        Execute a workflow with provided inputs.

        Body should contain:
        - inputs: Dict of input overrides (node_id.input_name: value)
        - images: Dict of base64-encoded images (node_id: base64_data)
        - wait: Whether to wait for completion (default: true)
        - timeout: Execution timeout in seconds (default: 300)
        """
        try:
            import execution

            name = request.match_info["name"]
            data = await request.json()

            # Find workflow using helper
            workflow_path = find_workflow_path(name)
            if not workflow_path:
                return web.json_response(
                    {"success": False, "error": f"Workflow not found: {name}"},
                    status=404
                )

            with open(workflow_path, "r") as f:
                workflow = json.load(f)

            # Apply input overrides
            inputs = data.get("inputs", {})
            for key, value in inputs.items():
                if "." in key:
                    node_id, input_name = key.split(".", 1)
                    if node_id in workflow and "inputs" in workflow[node_id]:
                        workflow[node_id]["inputs"][input_name] = value

            # Handle image inputs
            images = data.get("images", {})
            for node_id, image_data in images.items():
                if node_id in workflow:
                    # Decode and save image
                    if isinstance(image_data, str) and image_data.startswith("data:"):
                        # Handle data URL
                        _, image_data = image_data.split(",", 1)

                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes))

                    # Save to input folder
                    input_dir = folder_paths.get_input_directory()
                    filename = f"api_input_{uuid.uuid4().hex[:8]}.png"
                    filepath = Path(input_dir) / filename
                    image.save(filepath, "PNG")

                    # Update workflow to use this image
                    if workflow[node_id].get("class_type") == "LoadImage":
                        workflow[node_id]["inputs"]["image"] = filename

            # Create job
            job_id = str(uuid.uuid4())
            tracker = get_job_tracker()
            from .job_tracker import JobStatus
            tracker.create_job(
                job_id=job_id,
                prompt_id="",
                workflow_name=name,
                inputs=inputs
            )

            # Queue the prompt
            prompt_queue = server.prompt_queue

            # Validate the prompt first
            valid = execution.validate_prompt(workflow)
            if valid[0]:
                # Queue it
                prompt_id = str(uuid.uuid4())
                outputs_to_execute = valid[2]

                prompt_queue.put((
                    0,  # priority
                    prompt_id,
                    workflow,
                    {},  # extra_data
                    outputs_to_execute
                ))

                tracker.update_status(job_id, JobStatus.QUEUED)
                tracker._jobs[job_id].prompt_id = prompt_id

                wait = data.get("wait", True)
                timeout = data.get("timeout", 300)

                if wait:
                    # Wait for execution to complete
                    start_time = asyncio.get_event_loop().time()
                    while True:
                        await asyncio.sleep(0.5)

                        # Check history for result
                        history = server.prompt_queue.get_history(prompt_id=prompt_id)
                        if prompt_id in history:
                            result = history[prompt_id]
                            tracker.update_status(job_id, JobStatus.COMPLETED)

                            # Get output images
                            output_images = []
                            outputs = result.get("outputs", {})
                            for node_id, node_output in outputs.items():
                                if "images" in node_output:
                                    for img_info in node_output["images"]:
                                        filename = img_info["filename"]
                                        subfolder = img_info.get("subfolder", "")
                                        output_images.append({
                                            "filename": filename,
                                            "subfolder": subfolder,
                                            "url": f"/view?filename={filename}&subfolder={subfolder}&type=output"
                                        })

                            return web.json_response({
                                "success": True,
                                "job_id": job_id,
                                "prompt_id": prompt_id,
                                "status": "completed",
                                "outputs": outputs,
                                "images": output_images
                            })

                        # Check timeout
                        if asyncio.get_event_loop().time() - start_time > timeout:
                            tracker.update_status(job_id, JobStatus.FAILED, error="Execution timeout")
                            return web.json_response({
                                "success": False,
                                "job_id": job_id,
                                "prompt_id": prompt_id,
                                "error": "Execution timeout"
                            }, status=408)
                else:
                    return web.json_response({
                        "success": True,
                        "job_id": job_id,
                        "prompt_id": prompt_id,
                        "status": "queued",
                        "message": "Workflow queued for execution"
                    })

            else:
                # Validation failed
                errors = valid[1] if len(valid) > 1 else []
                tracker.update_status(job_id, JobStatus.FAILED, error=str(errors))
                return web.json_response({
                    "success": False,
                    "job_id": job_id,
                    "error": "Workflow validation failed",
                    "validation_errors": errors
                }, status=400)

        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.post("/api-manager/execute")
    async def execute_raw_workflow(request):
        """
        Execute a raw workflow provided in the request body.

        Body should contain:
        - workflow: The workflow JSON (API format)
        - inputs: Optional input overrides
        - images: Optional base64-encoded images
        - wait: Whether to wait for completion
        - timeout: Execution timeout
        """
        try:
            import execution

            data = await request.json()
            workflow = data.get("workflow")

            if not workflow:
                return web.json_response(
                    {"success": False, "error": "No workflow provided"},
                    status=400
                )

            # Apply input overrides
            inputs = data.get("inputs", {})
            for key, value in inputs.items():
                if "." in key:
                    node_id, input_name = key.split(".", 1)
                    if node_id in workflow and "inputs" in workflow[node_id]:
                        workflow[node_id]["inputs"][input_name] = value

            # Handle image inputs (same as above)
            images = data.get("images", {})
            for node_id, image_data in images.items():
                if node_id in workflow:
                    import folder_paths
                    if isinstance(image_data, str) and image_data.startswith("data:"):
                        _, image_data = image_data.split(",", 1)

                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes))

                    input_dir = folder_paths.get_input_directory()
                    filename = f"api_input_{uuid.uuid4().hex[:8]}.png"
                    filepath = Path(input_dir) / filename
                    image.save(filepath, "PNG")

                    if workflow[node_id].get("class_type") == "LoadImage":
                        workflow[node_id]["inputs"]["image"] = filename

            # Create job and execute
            job_id = str(uuid.uuid4())
            tracker = get_job_tracker()
            from .job_tracker import JobStatus
            tracker.create_job(
                job_id=job_id,
                prompt_id="",
                workflow_name="api_workflow",
                inputs=inputs
            )

            # Validate and queue
            valid = execution.validate_prompt(workflow)
            if valid[0]:
                prompt_id = str(uuid.uuid4())
                outputs_to_execute = valid[2]

                server.prompt_queue.put((
                    0,
                    prompt_id,
                    workflow,
                    {},
                    outputs_to_execute
                ))

                tracker.update_status(job_id, JobStatus.QUEUED)
                tracker._jobs[job_id].prompt_id = prompt_id

                wait = data.get("wait", False)
                timeout = data.get("timeout", 300)

                if wait:
                    start_time = asyncio.get_event_loop().time()
                    while True:
                        await asyncio.sleep(0.5)
                        history = server.prompt_queue.get_history(prompt_id=prompt_id)

                        if prompt_id in history:
                            result = history[prompt_id]
                            tracker.update_status(job_id, JobStatus.COMPLETED)

                            output_images = []
                            outputs = result.get("outputs", {})
                            for node_id, node_output in outputs.items():
                                if "images" in node_output:
                                    for img_info in node_output["images"]:
                                        output_images.append({
                                            "filename": img_info["filename"],
                                            "subfolder": img_info.get("subfolder", ""),
                                            "url": f"/view?filename={img_info['filename']}&type=output"
                                        })

                            return web.json_response({
                                "success": True,
                                "job_id": job_id,
                                "prompt_id": prompt_id,
                                "status": "completed",
                                "outputs": outputs,
                                "images": output_images
                            })

                        if asyncio.get_event_loop().time() - start_time > timeout:
                            tracker.update_status(job_id, JobStatus.FAILED, error="Timeout")
                            return web.json_response({
                                "success": False,
                                "error": "Execution timeout"
                            }, status=408)

                return web.json_response({
                    "success": True,
                    "job_id": job_id,
                    "prompt_id": prompt_id,
                    "status": "queued"
                })

            else:
                return web.json_response({
                    "success": False,
                    "error": "Validation failed",
                    "details": valid[1] if len(valid) > 1 else []
                }, status=400)

        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ==========================================================================
    # Job Management
    # ==========================================================================

    @routes.get("/api-manager/jobs")
    async def list_jobs(request):
        """List job history."""
        try:
            tracker = get_job_tracker()

            status_filter = request.query.get("status")
            limit = int(request.query.get("limit", "50"))

            from .job_tracker import JobStatus
            status = JobStatus(status_filter) if status_filter else None

            jobs = tracker.list_jobs(status=status, limit=limit)

            return web.json_response({
                "success": True,
                "count": len(jobs),
                "jobs": [j.to_dict() for j in jobs]
            })

        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.get("/api-manager/jobs/{job_id}")
    async def get_job(request):
        """Get a specific job by ID."""
        try:
            job_id = request.match_info["job_id"]
            tracker = get_job_tracker()
            job = tracker.get_job(job_id)

            if job:
                return web.json_response({
                    "success": True,
                    "job": job.to_dict()
                })
            else:
                return web.json_response(
                    {"success": False, "error": "Job not found"},
                    status=404
                )

        except Exception as e:
            logger.error(f"Error getting job: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.get("/api-manager/jobs/stats")
    async def get_job_stats(request):
        """Get job statistics."""
        try:
            tracker = get_job_tracker()
            stats = tracker.get_statistics()

            return web.json_response({
                "success": True,
                "statistics": stats
            })

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.delete("/api-manager/jobs")
    async def clear_jobs(request):
        """Clear job history."""
        try:
            tracker = get_job_tracker()
            keep_active = request.query.get("keep_active", "true").lower() == "true"
            removed = tracker.clear_history(keep_active=keep_active)

            return web.json_response({
                "success": True,
                "removed": removed
            })

        except Exception as e:
            logger.error(f"Error clearing jobs: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ==========================================================================
    # MCP Server Control
    # ==========================================================================

    @routes.post("/api-manager/mcp/start")
    async def start_mcp_server(request):
        """Start the MCP server."""
        global _mcp_server_process

        try:
            settings = get_plugin_settings()
            mcp_settings = settings.get_mcp_settings()

            if not mcp_settings.enabled:
                return web.json_response({
                    "success": False,
                    "error": "MCP server is disabled in settings"
                }, status=400)

            if _mcp_server_process is not None:
                return web.json_response({
                    "success": False,
                    "error": "MCP server is already running"
                }, status=400)

            # Start MCP server in background
            import subprocess
            import sys

            script_path = Path(__file__).parent / "mcp_server.py"
            _mcp_server_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            return web.json_response({
                "success": True,
                "message": f"MCP server started on port {mcp_settings.port}",
                "pid": _mcp_server_process.pid
            })

        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.post("/api-manager/mcp/stop")
    async def stop_mcp_server(request):
        """Stop the MCP server."""
        global _mcp_server_process

        try:
            if _mcp_server_process is None:
                return web.json_response({
                    "success": False,
                    "error": "MCP server is not running"
                }, status=400)

            _mcp_server_process.terminate()
            _mcp_server_process.wait(timeout=5)
            _mcp_server_process = None

            return web.json_response({
                "success": True,
                "message": "MCP server stopped"
            })

        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.get("/api-manager/mcp/status")
    async def mcp_status(request):
        """Get MCP server status."""
        global _mcp_server_process

        settings = get_plugin_settings()
        mcp_settings = settings.get_mcp_settings()

        running = _mcp_server_process is not None and _mcp_server_process.poll() is None

        return web.json_response({
            "success": True,
            "enabled": mcp_settings.enabled,
            "running": running,
            "port": mcp_settings.port,
            "transport": mcp_settings.transport
        })

    logger.info("API Manager routes registered")
