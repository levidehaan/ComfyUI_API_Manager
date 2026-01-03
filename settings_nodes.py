"""
Settings and Job History Nodes

Provides nodes for:
- Managing API Manager settings
- Viewing job history
- Connection management
- MCP server control
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class APIManagerSettingsNode:
    """
    Displays and manages API Manager settings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "action": (["view", "enable_mcp", "disable_mcp", "reset"], {"default": "view"}),
            }
        }

    RETURN_TYPES = ("JSON", "BOOLEAN", "STRING")
    RETURN_NAMES = ("SETTINGS", "MCP_ENABLED", "STATUS")
    FUNCTION = "execute"
    CATEGORY = "API Manager/Settings"

    def execute(self, action: str = "view"):
        try:
            from .settings_manager import get_settings_manager

            manager = get_settings_manager()

            if action == "enable_mcp":
                manager.set_mcp_enabled(True)
                status = "MCP server enabled"
            elif action == "disable_mcp":
                manager.set_mcp_enabled(False)
                status = "MCP server disabled"
            elif action == "reset":
                manager.reset_to_defaults()
                status = "Settings reset to defaults"
            else:
                status = "Current settings"

            settings = manager.export_settings()
            mcp_enabled = manager.is_mcp_enabled()

            return (settings, mcp_enabled, status)

        except Exception as e:
            logger.error(f"Error managing settings: {e}")
            return ({}, False, f"Error: {e}")


class ConnectionManagerNode:
    """
    Manages ComfyUI connections.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["list", "add", "remove", "set_default", "test"], {"default": "list"}),
            },
            "optional": {
                "connection_name": ("STRING", {"default": ""}),
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
                "use_ssl": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("JSON", "BOOLEAN", "STRING")
    RETURN_NAMES = ("CONNECTIONS", "SUCCESS", "MESSAGE")
    FUNCTION = "execute"
    CATEGORY = "API Manager/Settings"

    def execute(
        self,
        action: str = "list",
        connection_name: str = "",
        host: str = "127.0.0.1",
        port: int = 8188,
        use_ssl: bool = False
    ):
        try:
            from .settings_manager import get_settings_manager

            manager = get_settings_manager()

            if action == "add":
                if not connection_name:
                    return ({}, False, "Connection name required")
                manager.add_connection(connection_name, host, port, use_ssl)
                message = f"Added connection: {connection_name}"

            elif action == "remove":
                if not connection_name:
                    return ({}, False, "Connection name required")
                success = manager.remove_connection(connection_name)
                message = f"Removed: {connection_name}" if success else "Cannot remove (might be default)"

            elif action == "set_default":
                if not connection_name:
                    return ({}, False, "Connection name required")
                success = manager.set_default_connection(connection_name)
                message = f"Default set to: {connection_name}" if success else "Connection not found"

            elif action == "test":
                name = connection_name or manager.settings.default_connection
                conn = manager.get_connection(name)
                if conn:
                    # Test connection
                    from .comfyui_client import ComfyUIClient
                    import asyncio

                    async def test():
                        client = ComfyUIClient(conn)
                        try:
                            await client.connect()
                            is_connected = await client.is_connected()
                            return is_connected
                        finally:
                            await client.disconnect()

                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    is_connected = loop.run_until_complete(test())
                    message = f"Connection {name}: {'OK' if is_connected else 'FAILED'}"
                    return (manager.list_connections(), is_connected, message)
                else:
                    message = f"Connection not found: {name}"
                    return ({}, False, message)
            else:
                message = "Current connections"

            connections = {
                name: conn.to_dict()
                for name, conn in manager.list_connections().items()
            }
            connections["_default"] = manager.settings.default_connection

            return (connections, True, message)

        except Exception as e:
            logger.error(f"Error managing connections: {e}")
            return ({}, False, f"Error: {e}")


class JobHistoryNode:
    """
    Displays job execution history.
    """

    STATUS_FILTERS = ["all", "pending", "running", "completed", "failed", "cancelled"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "status_filter": (cls.STATUS_FILTERS, {"default": "all"}),
                "limit": ("INT", {"default": 50, "min": 1, "max": 500}),
                "workflow_filter": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("JSON", "INT", "STRING")
    RETURN_NAMES = ("JOBS", "COUNT", "SUMMARY")
    FUNCTION = "get_history"
    CATEGORY = "API Manager/Settings"

    def get_history(
        self,
        status_filter: str = "all",
        limit: int = 50,
        workflow_filter: str = ""
    ):
        try:
            from .job_tracker import get_job_tracker, JobStatus

            tracker = get_job_tracker()

            # Convert status filter
            status = None
            if status_filter != "all":
                try:
                    status = JobStatus(status_filter)
                except ValueError:
                    pass

            # Get jobs
            jobs = tracker.list_jobs(
                status=status,
                workflow_name=workflow_filter if workflow_filter else None,
                limit=limit
            )

            # Create summary
            stats = tracker.get_statistics()
            summary = (
                f"Total: {stats['total_jobs']} | "
                f"Active: {stats['active_jobs']} | "
                f"Success Rate: {stats['success_rate']:.1%}"
            )

            return (
                [j.to_dict() for j in jobs],
                len(jobs),
                summary
            )

        except Exception as e:
            logger.error(f"Error getting job history: {e}")
            return ([], 0, f"Error: {e}")


class JobStatisticsNode:
    """
    Displays job execution statistics.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("JSON", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("STATISTICS", "SUCCESS_RATE", "AVG_DURATION", "TOTAL_JOBS")
    FUNCTION = "get_stats"
    CATEGORY = "API Manager/Settings"

    def get_stats(self):
        try:
            from .job_tracker import get_job_tracker

            tracker = get_job_tracker()
            stats = tracker.get_statistics()

            return (
                stats,
                stats.get("success_rate", 0.0),
                stats.get("average_duration_seconds", 0.0),
                stats.get("total_jobs", 0)
            )

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return ({}, 0.0, 0.0, 0)


class ClearJobHistoryNode:
    """
    Clears job history.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "confirm": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "keep_active": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("REMOVED_COUNT", "MESSAGE")
    FUNCTION = "clear"
    CATEGORY = "API Manager/Settings"
    OUTPUT_NODE = True

    def clear(self, confirm: bool = False, keep_active: bool = True):
        if not confirm:
            return (0, "Set confirm=True to clear history")

        try:
            from .job_tracker import get_job_tracker

            tracker = get_job_tracker()
            removed = tracker.clear_history(keep_active=keep_active)

            message = f"Removed {removed} jobs"
            if keep_active:
                message += " (kept active jobs)"

            return (removed, message)

        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return (0, f"Error: {e}")


class MCPServerControlNode:
    """
    Controls the MCP server.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["status", "enable", "disable", "configure"], {"default": "status"}),
            },
            "optional": {
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8765, "min": 1, "max": 65535}),
                "transport": (["stdio", "streamable-http"], {"default": "stdio"}),
            }
        }

    RETURN_TYPES = ("JSON", "BOOLEAN", "STRING")
    RETURN_NAMES = ("MCP_CONFIG", "ENABLED", "STATUS")
    FUNCTION = "control"
    CATEGORY = "API Manager/Settings"
    OUTPUT_NODE = True

    def control(
        self,
        action: str = "status",
        host: str = "127.0.0.1",
        port: int = 8765,
        transport: str = "stdio"
    ):
        try:
            from .settings_manager import get_settings_manager

            manager = get_settings_manager()

            if action == "enable":
                manager.set_mcp_enabled(True)
                status = "MCP server enabled. Restart ComfyUI to apply."
            elif action == "disable":
                manager.set_mcp_enabled(False)
                status = "MCP server disabled."
            elif action == "configure":
                manager.update_mcp_settings(
                    host=host,
                    port=port,
                    transport=transport
                )
                status = f"MCP configured: {transport} on {host}:{port}"
            else:
                status = "Current MCP configuration"

            mcp_settings = manager.get_mcp_settings()
            enabled = mcp_settings.enabled

            return (mcp_settings.to_dict(), enabled, status)

        except Exception as e:
            logger.error(f"Error controlling MCP: {e}")
            return ({}, False, f"Error: {e}")


class LoggingSettingsNode:
    """
    Configures logging settings.
    """

    LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "level": (cls.LOG_LEVELS, {"default": "INFO"}),
            },
            "optional": {
                "log_to_file": ("BOOLEAN", {"default": False}),
                "log_file": ("STRING", {"default": "comfyui_api_manager.log"}),
            }
        }

    RETURN_TYPES = ("JSON", "STRING")
    RETURN_NAMES = ("LOGGING_CONFIG", "STATUS")
    FUNCTION = "configure"
    CATEGORY = "API Manager/Settings"
    OUTPUT_NODE = True

    def configure(
        self,
        level: str = "INFO",
        log_to_file: bool = False,
        log_file: str = "comfyui_api_manager.log"
    ):
        try:
            from .settings_manager import get_settings_manager

            manager = get_settings_manager()
            settings = manager.update_logging_settings(
                level=level,
                log_to_file=log_to_file,
                log_file=log_file
            )

            status = f"Logging set to {level}"
            if log_to_file:
                status += f", writing to {log_file}"

            return (settings.to_dict(), status)

        except Exception as e:
            logger.error(f"Error configuring logging: {e}")
            return ({}, f"Error: {e}")


class ViewRecentJobsNode:
    """
    Quick view of recent jobs with success/failure summary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "count": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("JOB_SUMMARY", "SUCCESS_COUNT", "FAILURE_COUNT")
    FUNCTION = "view"
    CATEGORY = "API Manager/Settings"

    def view(self, count: int = 10):
        try:
            from .job_tracker import get_job_tracker, JobStatus

            tracker = get_job_tracker()

            # Get recent jobs
            jobs = tracker.list_jobs(limit=count)

            success_count = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
            failure_count = sum(1 for j in jobs if j.status == JobStatus.FAILED)

            # Build summary
            lines = []
            for job in jobs[:count]:
                status_icon = {
                    JobStatus.COMPLETED: "[OK]",
                    JobStatus.FAILED: "[FAIL]",
                    JobStatus.RUNNING: "[RUN]",
                    JobStatus.PENDING: "[WAIT]",
                    JobStatus.CANCELLED: "[STOP]",
                    JobStatus.QUEUED: "[QUEUE]"
                }.get(job.status, "[?]")

                duration = f"{job.duration:.1f}s" if job.duration else "-"
                lines.append(
                    f"{status_icon} {job.workflow_name[:20]:<20} {duration:>8} {job.created_at.strftime('%H:%M:%S')}"
                )

            summary = "\n".join(lines) if lines else "No jobs in history"

            return (summary, success_count, failure_count)

        except Exception as e:
            logger.error(f"Error viewing jobs: {e}")
            return (f"Error: {e}", 0, 0)
