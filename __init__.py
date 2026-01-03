"""
ComfyUI API Manager

A server-side plugin for ComfyUI that provides:
- REST API endpoints for workflow management and execution
- Workflow analysis to detect inputs (images, text, numbers)
- MCP (Model Context Protocol) server for AI integration
- Job tracking with history
- Settings panel in ComfyUI UI

This is a server-only plugin - no workflow nodes are added.
All functionality is exposed via API endpoints.

Version: 2.0.0
"""

import logging
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
# Server-Only Plugin - No Node Mappings
# ============================================================================

# Empty mappings - this plugin adds no workflow nodes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============================================================================
# Web Directory (for ComfyUI settings panel extension)
# ============================================================================

WEB_DIRECTORY = "./web"

# ============================================================================
# Server Route Registration
# ============================================================================

def _register_api_routes():
    """Register API routes with ComfyUI's PromptServer."""
    try:
        from server import PromptServer
        from .api_routes import register_routes

        server = PromptServer.instance
        register_routes(server)
        logger.info("API Manager routes registered successfully")
        return True

    except ImportError as e:
        logger.warning(f"Could not import PromptServer: {e}")
        logger.info("API routes will not be available (running outside ComfyUI?)")
        return False
    except Exception as e:
        logger.error(f"Error registering API routes: {e}")
        return False

# ============================================================================
# MCP Server Initialization
# ============================================================================

def _initialize_mcp_server():
    """Check MCP server status from settings."""
    try:
        from .settings_manager import get_settings_manager
        from .mcp_server import MCP_AVAILABLE

        if not MCP_AVAILABLE:
            logger.info("MCP SDK not installed. Install with: pip install mcp")
            return None

        settings = get_settings_manager()

        if settings.is_mcp_enabled():
            logger.info(f"MCP server enabled on port {settings.settings.get('mcp', {}).get('port', 8765)}")
            logger.info("MCP server can be started via API: POST /api-manager/mcp/start")
            return True
        else:
            logger.info("MCP server disabled. Enable in ComfyUI Settings > API Manager")
            return False

    except ImportError as e:
        logger.debug(f"MCP initialization skipped: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error checking MCP settings: {e}")
        return None

# ============================================================================
# Plugin Initialization
# ============================================================================

def _initialize_plugin():
    """Initialize the API Manager plugin."""
    logger.info(f"ComfyUI API Manager v{__version__} initializing...")

    # Register API routes with ComfyUI server
    routes_registered = _register_api_routes()

    # Check MCP server status
    mcp_status = _initialize_mcp_server()

    # Log initialization summary
    if routes_registered:
        logger.info("API Manager initialized successfully")
        logger.info("API endpoints available at /api-manager/*")
        logger.info("Settings available in ComfyUI Settings panel")
    else:
        logger.warning("API Manager partially initialized (routes not registered)")

    return routes_registered

# Initialize on module load
_plugin_initialized = _initialize_plugin()

# ============================================================================
# Export
# ============================================================================

__all__ = [
    "__version__",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
