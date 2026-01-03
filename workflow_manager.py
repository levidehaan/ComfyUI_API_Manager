"""
Workflow Manager Module

Handles loading, saving, validating, and managing ComfyUI workflows.
Supports workflow templates, input parameter injection, and workflow validation.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowInput:
    """Represents an input parameter for a workflow."""
    node_id: str
    input_name: str
    input_type: str
    current_value: Any
    description: str = ""
    required: bool = True
    options: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "input_name": self.input_name,
            "input_type": self.input_type,
            "current_value": self.current_value,
            "description": self.description,
            "required": self.required,
            "options": self.options
        }


@dataclass
class WorkflowOutput:
    """Represents an output from a workflow."""
    node_id: str
    output_type: str
    output_name: str = ""

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "output_type": self.output_type,
            "output_name": self.output_name
        }


@dataclass
class Workflow:
    """Represents a ComfyUI workflow."""
    name: str
    workflow_json: dict
    inputs: list[WorkflowInput] = field(default_factory=list)
    outputs: list[WorkflowOutput] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"
    source_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "workflow": self.workflow_json,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "description": self.description,
            "version": self.version,
            "source_path": self.source_path
        }

    def get_api_format(self) -> dict:
        """Get workflow in API format for execution."""
        return self.workflow_json


class WorkflowManager:
    """
    Manages ComfyUI workflows including loading, validation, and execution preparation.
    """

    # Common node types that accept user inputs
    INPUT_NODE_TYPES = {
        "CLIPTextEncode": ["text"],
        "KSampler": ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
        "KSamplerAdvanced": ["seed", "steps", "cfg", "sampler_name", "scheduler", "start_at_step", "end_at_step"],
        "EmptyLatentImage": ["width", "height", "batch_size"],
        "LoadImage": ["image"],
        "CheckpointLoaderSimple": ["ckpt_name"],
        "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
        "VAELoader": ["vae_name"],
        "ControlNetLoader": ["control_net_name"],
        "LoadImageMask": ["image", "channel"],
    }

    # Output node types
    OUTPUT_NODE_TYPES = {
        "SaveImage": "IMAGE",
        "PreviewImage": "IMAGE",
        "SaveImageWebsocket": "IMAGE",
    }

    def __init__(self, workflows_dir: Optional[Path] = None):
        self.workflows_dir = workflows_dir or Path("./workflows")
        self._workflows_cache: dict[str, Workflow] = {}

    def parse_workflow(self, workflow_json: dict, name: str = "Untitled") -> Workflow:
        """
        Parse a workflow JSON and extract inputs and outputs.

        Args:
            workflow_json: The workflow in API format
            name: Name for the workflow

        Returns:
            Parsed Workflow object
        """
        inputs = []
        outputs = []

        for node_id, node_data in workflow_json.items():
            class_type = node_data.get("class_type", "")

            # Check for input nodes
            if class_type in self.INPUT_NODE_TYPES:
                input_names = self.INPUT_NODE_TYPES[class_type]
                node_inputs = node_data.get("inputs", {})

                for input_name in input_names:
                    if input_name in node_inputs:
                        value = node_inputs[input_name]
                        # Skip if it's a connection to another node
                        if not isinstance(value, list):
                            inputs.append(WorkflowInput(
                                node_id=node_id,
                                input_name=input_name,
                                input_type=self._infer_type(value),
                                current_value=value,
                                description=f"{class_type}.{input_name}"
                            ))

            # Check for output nodes
            if class_type in self.OUTPUT_NODE_TYPES:
                outputs.append(WorkflowOutput(
                    node_id=node_id,
                    output_type=self.OUTPUT_NODE_TYPES[class_type],
                    output_name=node_data.get("_meta", {}).get("title", class_type)
                ))

        return Workflow(
            name=name,
            workflow_json=workflow_json,
            inputs=inputs,
            outputs=outputs
        )

    def _infer_type(self, value: Any) -> str:
        """Infer the type of a value."""
        if isinstance(value, bool):
            return "BOOLEAN"
        elif isinstance(value, int):
            return "INT"
        elif isinstance(value, float):
            return "FLOAT"
        elif isinstance(value, str):
            return "STRING"
        elif isinstance(value, list):
            return "ARRAY"
        elif isinstance(value, dict):
            return "OBJECT"
        return "UNKNOWN"

    def load_workflow(self, path: str | Path) -> Workflow:
        """
        Load a workflow from a file.

        Args:
            path: Path to the workflow JSON file

        Returns:
            Loaded Workflow object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        name = path.stem
        workflow = self.parse_workflow(data, name)
        workflow.source_path = str(path)

        self._workflows_cache[name] = workflow
        logger.info(f"Loaded workflow: {name} from {path}")

        return workflow

    def save_workflow(self, workflow: Workflow, path: Optional[str | Path] = None) -> Path:
        """
        Save a workflow to a file.

        Args:
            workflow: The workflow to save
            path: Optional path (uses workflows_dir if not specified)

        Returns:
            Path where workflow was saved
        """
        if path is None:
            path = self.workflows_dir / f"{workflow.name}.json"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(workflow.workflow_json, f, indent=2)

        logger.info(f"Saved workflow: {workflow.name} to {path}")
        return path

    def apply_inputs(self, workflow: Workflow, inputs: dict[str, Any]) -> dict:
        """
        Apply input values to a workflow.

        Args:
            workflow: The workflow to modify
            inputs: Dict mapping input keys to values
                    Keys can be "node_id.input_name" or just "input_name"

        Returns:
            Modified workflow JSON ready for execution
        """
        workflow_copy = json.loads(json.dumps(workflow.workflow_json))

        for key, value in inputs.items():
            applied = False

            # Check if key is "node_id.input_name" format
            if "." in key:
                node_id, input_name = key.split(".", 1)
                if node_id in workflow_copy:
                    if "inputs" in workflow_copy[node_id]:
                        workflow_copy[node_id]["inputs"][input_name] = value
                        applied = True
                        logger.debug(f"Applied input {key}={value}")

            # Otherwise, search for matching input name
            if not applied:
                for node_id, node_data in workflow_copy.items():
                    if "inputs" in node_data and key in node_data["inputs"]:
                        # Only replace if it's not a node connection
                        current = node_data["inputs"][key]
                        if not isinstance(current, list):
                            node_data["inputs"][key] = value
                            applied = True
                            logger.debug(f"Applied input {node_id}.{key}={value}")
                            break

            if not applied:
                logger.warning(f"Could not apply input: {key}")

        return workflow_copy

    def set_text_prompt(
        self,
        workflow: dict,
        prompt: str,
        node_id: Optional[str] = None,
        node_title: Optional[str] = None
    ) -> dict:
        """
        Set a text prompt in a CLIPTextEncode node.

        Args:
            workflow: The workflow JSON
            prompt: The text prompt to set
            node_id: Specific node ID to set
            node_title: Find node by title (from _meta)

        Returns:
            Modified workflow
        """
        workflow_copy = json.loads(json.dumps(workflow))

        for nid, node_data in workflow_copy.items():
            if node_data.get("class_type") == "CLIPTextEncode":
                # Match by node_id
                if node_id and nid == node_id:
                    node_data["inputs"]["text"] = prompt
                    return workflow_copy

                # Match by title
                if node_title:
                    title = node_data.get("_meta", {}).get("title", "")
                    if node_title.lower() in title.lower():
                        node_data["inputs"]["text"] = prompt
                        return workflow_copy

                # If no specific target, set in first positive prompt
                if not node_id and not node_title:
                    title = node_data.get("_meta", {}).get("title", "").lower()
                    if "positive" in title or "prompt" in title:
                        node_data["inputs"]["text"] = prompt
                        return workflow_copy

        # Fallback: set in first CLIPTextEncode
        for nid, node_data in workflow_copy.items():
            if node_data.get("class_type") == "CLIPTextEncode":
                node_data["inputs"]["text"] = prompt
                break

        return workflow_copy

    def set_image_input(
        self,
        workflow: dict,
        image_path: str,
        node_id: Optional[str] = None
    ) -> dict:
        """
        Set an input image in a LoadImage node.

        Args:
            workflow: The workflow JSON
            image_path: Path/name of the uploaded image
            node_id: Specific node ID to set

        Returns:
            Modified workflow
        """
        workflow_copy = json.loads(json.dumps(workflow))

        for nid, node_data in workflow_copy.items():
            if node_data.get("class_type") == "LoadImage":
                if node_id is None or nid == node_id:
                    node_data["inputs"]["image"] = image_path
                    if node_id:
                        break

        return workflow_copy

    def set_sampler_settings(
        self,
        workflow: dict,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        sampler_name: Optional[str] = None,
        scheduler: Optional[str] = None,
        denoise: Optional[float] = None,
        node_id: Optional[str] = None
    ) -> dict:
        """
        Set sampler settings in KSampler nodes.

        Args:
            workflow: The workflow JSON
            seed: Random seed (-1 for random)
            steps: Number of sampling steps
            cfg: CFG scale
            sampler_name: Sampler algorithm name
            scheduler: Scheduler name
            denoise: Denoising strength
            node_id: Specific node ID to set

        Returns:
            Modified workflow
        """
        workflow_copy = json.loads(json.dumps(workflow))

        for nid, node_data in workflow_copy.items():
            class_type = node_data.get("class_type", "")
            if class_type in ("KSampler", "KSamplerAdvanced"):
                if node_id is None or nid == node_id:
                    inputs = node_data.get("inputs", {})
                    if seed is not None:
                        inputs["seed"] = seed
                    if steps is not None:
                        inputs["steps"] = steps
                    if cfg is not None:
                        inputs["cfg"] = cfg
                    if sampler_name is not None:
                        inputs["sampler_name"] = sampler_name
                    if scheduler is not None:
                        inputs["scheduler"] = scheduler
                    if denoise is not None:
                        inputs["denoise"] = denoise

                    if node_id:
                        break

        return workflow_copy

    def set_image_size(
        self,
        workflow: dict,
        width: int,
        height: int,
        batch_size: int = 1,
        node_id: Optional[str] = None
    ) -> dict:
        """
        Set image dimensions in EmptyLatentImage nodes.

        Args:
            workflow: The workflow JSON
            width: Image width
            height: Image height
            batch_size: Number of images to generate
            node_id: Specific node ID to set

        Returns:
            Modified workflow
        """
        workflow_copy = json.loads(json.dumps(workflow))

        for nid, node_data in workflow_copy.items():
            if node_data.get("class_type") == "EmptyLatentImage":
                if node_id is None or nid == node_id:
                    node_data["inputs"]["width"] = width
                    node_data["inputs"]["height"] = height
                    node_data["inputs"]["batch_size"] = batch_size
                    if node_id:
                        break

        return workflow_copy

    def validate_workflow(self, workflow: dict) -> tuple[bool, list[str]]:
        """
        Validate a workflow structure.

        Args:
            workflow: The workflow JSON to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if not isinstance(workflow, dict):
            errors.append("Workflow must be a dictionary")
            return False, errors

        if len(workflow) == 0:
            errors.append("Workflow is empty")
            return False, errors

        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                errors.append(f"Node {node_id} must be a dictionary")
                continue

            if "class_type" not in node_data:
                errors.append(f"Node {node_id} missing 'class_type'")

            if "inputs" not in node_data:
                errors.append(f"Node {node_id} missing 'inputs'")

        # Check for at least one output node
        has_output = any(
            node.get("class_type") in self.OUTPUT_NODE_TYPES
            for node in workflow.values()
        )
        if not has_output:
            errors.append("Workflow has no output nodes (SaveImage, PreviewImage, etc.)")

        return len(errors) == 0, errors

    def get_node_connections(self, workflow: dict) -> dict[str, list[dict]]:
        """
        Analyze node connections in a workflow.

        Args:
            workflow: The workflow JSON

        Returns:
            Dict mapping node IDs to their connections
        """
        connections = {}

        for node_id, node_data in workflow.items():
            connections[node_id] = {
                "inputs_from": [],
                "outputs_to": []
            }

            inputs = node_data.get("inputs", {})
            for input_name, value in inputs.items():
                if isinstance(value, list) and len(value) >= 2:
                    source_node = str(value[0])
                    source_slot = value[1]
                    connections[node_id]["inputs_from"].append({
                        "input_name": input_name,
                        "source_node": source_node,
                        "source_slot": source_slot
                    })

        # Build outputs_to based on inputs_from
        for node_id, conn in connections.items():
            for inp in conn["inputs_from"]:
                source = inp["source_node"]
                if source in connections:
                    connections[source]["outputs_to"].append({
                        "target_node": node_id,
                        "target_input": inp["input_name"],
                        "output_slot": inp["source_slot"]
                    })

        return connections

    def find_nodes_by_type(self, workflow: dict, class_type: str) -> list[str]:
        """Find all nodes of a specific type."""
        return [
            node_id for node_id, node_data in workflow.items()
            if node_data.get("class_type") == class_type
        ]

    def find_nodes_by_title(self, workflow: dict, title_pattern: str) -> list[str]:
        """Find nodes whose title matches a pattern."""
        pattern = re.compile(title_pattern, re.IGNORECASE)
        results = []

        for node_id, node_data in workflow.items():
            title = node_data.get("_meta", {}).get("title", "")
            if pattern.search(title):
                results.append(node_id)

        return results
