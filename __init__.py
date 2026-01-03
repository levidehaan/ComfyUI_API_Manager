"""
ComfyUI API Manager

A comprehensive API manager for ComfyUI with:
- Full ComfyUI API integration (REST + WebSocket)
- MCP (Model Context Protocol) server for AI integration
- Workflow management and execution
- Job tracking with history
- Multiple connection support
- Robust error handling

Version: 2.0.0
"""

import logging
import sys
from pathlib import Path

__version__ = "2.0.0"
__author__ = "ComfyUI API Manager Contributors"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Import all node classes
# ============================================================================

# API Request nodes
from .api_request import APIRequestNode, APIRequestNodeSimple

# Image handling nodes
from .image_post_node import PostImageToAPI, DownloadImageFromURL

# Text and JSON processing nodes
from .text_prompt_combiner_node import (
    TextPromptCombinerNode,
    TextTemplateNode,
    JSONExtractNode
)
from .json_array_iterator import (
    JSONArrayIteratorNode,
    JSONArrayFilterNode,
    JSONArraySliceNode,
    JSONArrayMapNode,
    JSONArrayStatsNode,
    JSONMergeNode
)

# ComfyUI integration nodes
from .comfyui_nodes import (
    ComfyUIExecuteWorkflowNode,
    ComfyUIQueueStatusNode,
    ComfyUISystemInfoNode,
    ComfyUIListModelsNode,
    ComfyUIUploadImageNode,
    ComfyUIDownloadImageNode,
    ComfyUIInterruptNode,
    ComfyUIFreeMemoryNode,
    ComfyUIListWorkflowsNode
)

# Settings and management nodes
from .settings_nodes import (
    APIManagerSettingsNode,
    ConnectionManagerNode,
    JobHistoryNode,
    JobStatisticsNode,
    ClearJobHistoryNode,
    MCPServerControlNode,
    LoggingSettingsNode,
    ViewRecentJobsNode
)

# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    # API Request Nodes
    "APIRequestNode": APIRequestNode,
    "APIRequestNodeSimple": APIRequestNodeSimple,

    # Image Nodes
    "PostImageToAPI": PostImageToAPI,
    "DownloadImageFromURL": DownloadImageFromURL,

    # Text/JSON Processing Nodes
    "TextPromptCombinerNode": TextPromptCombinerNode,
    "TextTemplateNode": TextTemplateNode,
    "JSONExtractNode": JSONExtractNode,
    "JSONArrayIteratorNode": JSONArrayIteratorNode,
    "JSONArrayFilterNode": JSONArrayFilterNode,
    "JSONArraySliceNode": JSONArraySliceNode,
    "JSONArrayMapNode": JSONArrayMapNode,
    "JSONArrayStatsNode": JSONArrayStatsNode,
    "JSONMergeNode": JSONMergeNode,

    # ComfyUI Integration Nodes
    "ComfyUIExecuteWorkflow": ComfyUIExecuteWorkflowNode,
    "ComfyUIQueueStatus": ComfyUIQueueStatusNode,
    "ComfyUISystemInfo": ComfyUISystemInfoNode,
    "ComfyUIListModels": ComfyUIListModelsNode,
    "ComfyUIUploadImage": ComfyUIUploadImageNode,
    "ComfyUIDownloadImage": ComfyUIDownloadImageNode,
    "ComfyUIInterrupt": ComfyUIInterruptNode,
    "ComfyUIFreeMemory": ComfyUIFreeMemoryNode,
    "ComfyUIListWorkflows": ComfyUIListWorkflowsNode,

    # Settings Nodes
    "APIManagerSettings": APIManagerSettingsNode,
    "ConnectionManager": ConnectionManagerNode,
    "JobHistory": JobHistoryNode,
    "JobStatistics": JobStatisticsNode,
    "ClearJobHistory": ClearJobHistoryNode,
    "MCPServerControl": MCPServerControlNode,
    "LoggingSettings": LoggingSettingsNode,
    "ViewRecentJobs": ViewRecentJobsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # API Request Nodes
    "APIRequestNode": "API Request",
    "APIRequestNodeSimple": "Simple API Request",

    # Image Nodes
    "PostImageToAPI": "Post Image to API",
    "DownloadImageFromURL": "Download Image from URL",

    # Text/JSON Processing Nodes
    "TextPromptCombinerNode": "Text Prompt Combiner",
    "TextTemplateNode": "Text Template",
    "JSONExtractNode": "JSON Extract",
    "JSONArrayIteratorNode": "JSON Array Iterator",
    "JSONArrayFilterNode": "JSON Array Filter",
    "JSONArraySliceNode": "JSON Array Slice",
    "JSONArrayMapNode": "JSON Array Map",
    "JSONArrayStatsNode": "JSON Array Stats",
    "JSONMergeNode": "JSON Merge",

    # ComfyUI Integration Nodes
    "ComfyUIExecuteWorkflow": "Execute Workflow (ComfyUI)",
    "ComfyUIQueueStatus": "Queue Status (ComfyUI)",
    "ComfyUISystemInfo": "System Info (ComfyUI)",
    "ComfyUIListModels": "List Models (ComfyUI)",
    "ComfyUIUploadImage": "Upload Image (ComfyUI)",
    "ComfyUIDownloadImage": "Download Image (ComfyUI)",
    "ComfyUIInterrupt": "Interrupt (ComfyUI)",
    "ComfyUIFreeMemory": "Free Memory (ComfyUI)",
    "ComfyUIListWorkflows": "List Workflows (ComfyUI)",

    # Settings Nodes
    "APIManagerSettings": "API Manager Settings",
    "ConnectionManager": "Connection Manager",
    "JobHistory": "Job History",
    "JobStatistics": "Job Statistics",
    "ClearJobHistory": "Clear Job History",
    "MCPServerControl": "MCP Server Control",
    "LoggingSettings": "Logging Settings",
    "ViewRecentJobs": "View Recent Jobs",
}

# ============================================================================
# MCP Server Initialization
# ============================================================================

def _initialize_mcp_server():
    """Initialize MCP server if enabled in settings."""
    try:
        from .settings_manager import get_settings_manager
        from .mcp_server import create_mcp_server, MCP_AVAILABLE

        if not MCP_AVAILABLE:
            logger.info("MCP SDK not installed. MCP server disabled.")
            return None

        settings = get_settings_manager()

        if settings.is_mcp_enabled():
            logger.info("MCP server is enabled in settings")
            # Note: Actual server startup is handled separately
            # This just validates the configuration
            return True
        else:
            logger.info("MCP server is disabled in settings. Enable via settings node.")
            return False

    except ImportError as e:
        logger.debug(f"MCP initialization skipped: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error initializing MCP server: {e}")
        return None


# Initialize on module load
_mcp_status = _initialize_mcp_server()

# ============================================================================
# Web Directory (for ComfyUI web extensions)
# ============================================================================

WEB_DIRECTORY = "./web"

# ============================================================================
# Export all
# ============================================================================

__all__ = [
    # Version info
    "__version__",

    # Node mappings
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",

    # Node classes - API
    "APIRequestNode",
    "APIRequestNodeSimple",

    # Node classes - Image
    "PostImageToAPI",
    "DownloadImageFromURL",

    # Node classes - Text/JSON
    "TextPromptCombinerNode",
    "TextTemplateNode",
    "JSONExtractNode",
    "JSONArrayIteratorNode",
    "JSONArrayFilterNode",
    "JSONArraySliceNode",
    "JSONArrayMapNode",
    "JSONArrayStatsNode",
    "JSONMergeNode",

    # Node classes - ComfyUI
    "ComfyUIExecuteWorkflowNode",
    "ComfyUIQueueStatusNode",
    "ComfyUISystemInfoNode",
    "ComfyUIListModelsNode",
    "ComfyUIUploadImageNode",
    "ComfyUIDownloadImageNode",
    "ComfyUIInterruptNode",
    "ComfyUIFreeMemoryNode",
    "ComfyUIListWorkflowsNode",

    # Node classes - Settings
    "APIManagerSettingsNode",
    "ConnectionManagerNode",
    "JobHistoryNode",
    "JobStatisticsNode",
    "ClearJobHistoryNode",
    "MCPServerControlNode",
    "LoggingSettingsNode",
    "ViewRecentJobsNode",
]

logger.info(f"ComfyUI API Manager v{__version__} loaded - {len(NODE_CLASS_MAPPINGS)} nodes registered")
