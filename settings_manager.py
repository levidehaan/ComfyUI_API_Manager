"""
Settings Manager Module

Manages plugin settings including:
- MCP server enable/disable
- ComfyUI connections
- Job history configuration
- Logging settings
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .comfyui_client import ComfyUIConnection

logger = logging.getLogger(__name__)


@dataclass
class MCPSettings:
    """MCP server settings."""
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    transport: str = "stdio"  # "stdio" or "streamable-http"
    require_auth: bool = False
    auth_token: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "transport": self.transport,
            "require_auth": self.require_auth,
            "auth_token": self.auth_token
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MCPSettings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LoggingSettings:
    """Logging settings."""
    level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "comfyui_api_manager.log"
    max_log_size_mb: int = 10
    backup_count: int = 3

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "log_to_file": self.log_to_file,
            "log_file": self.log_file,
            "max_log_size_mb": self.max_log_size_mb,
            "backup_count": self.backup_count
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoggingSettings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class JobHistorySettings:
    """Job history settings."""
    enabled: bool = True
    max_jobs: int = 1000
    auto_cleanup_days: int = 30
    store_outputs: bool = True
    store_images: bool = False  # Images can be large

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "max_jobs": self.max_jobs,
            "auto_cleanup_days": self.auto_cleanup_days,
            "store_outputs": self.store_outputs,
            "store_images": self.store_images
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobHistorySettings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PluginSettings:
    """Complete plugin settings."""
    version: str = "2.0.0"
    mcp: MCPSettings = field(default_factory=MCPSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    job_history: JobHistorySettings = field(default_factory=JobHistorySettings)
    connections: dict[str, ComfyUIConnection] = field(default_factory=lambda: {
        "default": ComfyUIConnection()
    })
    default_connection: str = "default"
    workflows_dir: str = "./workflows"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "mcp": self.mcp.to_dict(),
            "logging": self.logging.to_dict(),
            "job_history": self.job_history.to_dict(),
            "connections": {
                name: conn.to_dict()
                for name, conn in self.connections.items()
            },
            "default_connection": self.default_connection,
            "workflows_dir": self.workflows_dir
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PluginSettings":
        settings = cls()
        settings.version = data.get("version", "2.0.0")

        if "mcp" in data:
            settings.mcp = MCPSettings.from_dict(data["mcp"])

        if "logging" in data:
            settings.logging = LoggingSettings.from_dict(data["logging"])

        if "job_history" in data:
            settings.job_history = JobHistorySettings.from_dict(data["job_history"])

        if "connections" in data:
            settings.connections = {
                name: ComfyUIConnection.from_dict(conn_data)
                for name, conn_data in data["connections"].items()
            }

        settings.default_connection = data.get("default_connection", "default")
        settings.workflows_dir = data.get("workflows_dir", "./workflows")

        return settings


class SettingsManager:
    """
    Manages plugin settings with persistence and validation.

    Settings are stored in a JSON file and can be modified at runtime.
    Changes can be observed via callbacks.
    """

    DEFAULT_SETTINGS_FILE = "api_manager_settings.json"

    def __init__(self, settings_file: Optional[Path] = None, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.settings_file = settings_file or self.base_dir / self.DEFAULT_SETTINGS_FILE
        self._settings = PluginSettings()
        self._lock = threading.RLock()
        self._callbacks: list[callable] = []

        self._load()

    def _load(self) -> None:
        """Load settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r") as f:
                    data = json.load(f)
                self._settings = PluginSettings.from_dict(data)
                logger.info(f"Loaded settings from {self.settings_file}")
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
                self._settings = PluginSettings()
        else:
            logger.info("No settings file found, using defaults")
            self._save()

    def _save(self) -> None:
        """Save settings to file."""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_file, "w") as f:
                json.dump(self._settings.to_dict(), f, indent=2)
            logger.debug(f"Saved settings to {self.settings_file}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def _notify_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify callbacks of a settings change."""
        for callback in self._callbacks:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Error in settings callback: {e}")

    @property
    def settings(self) -> PluginSettings:
        """Get current settings."""
        with self._lock:
            return self._settings

    def add_change_callback(self, callback: callable) -> None:
        """Add a callback for settings changes."""
        self._callbacks.append(callback)

    def remove_change_callback(self, callback: callable) -> None:
        """Remove a settings change callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    # MCP Settings
    def is_mcp_enabled(self) -> bool:
        """Check if MCP server is enabled."""
        with self._lock:
            return self._settings.mcp.enabled

    def set_mcp_enabled(self, enabled: bool) -> None:
        """Enable or disable MCP server."""
        with self._lock:
            old = self._settings.mcp.enabled
            self._settings.mcp.enabled = enabled
            self._save()
            self._notify_change("mcp.enabled", old, enabled)

    def get_mcp_settings(self) -> MCPSettings:
        """Get MCP settings."""
        with self._lock:
            return self._settings.mcp

    def update_mcp_settings(
        self,
        enabled: Optional[bool] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        transport: Optional[str] = None,
        require_auth: Optional[bool] = None,
        auth_token: Optional[str] = None
    ) -> MCPSettings:
        """Update MCP settings."""
        with self._lock:
            if enabled is not None:
                self._settings.mcp.enabled = enabled
            if host is not None:
                self._settings.mcp.host = host
            if port is not None:
                self._settings.mcp.port = port
            if transport is not None:
                self._settings.mcp.transport = transport
            if require_auth is not None:
                self._settings.mcp.require_auth = require_auth
            if auth_token is not None:
                self._settings.mcp.auth_token = auth_token

            self._save()
            return self._settings.mcp

    # Connection Management
    def get_connection(self, name: Optional[str] = None) -> Optional[ComfyUIConnection]:
        """Get a connection by name."""
        with self._lock:
            name = name or self._settings.default_connection
            return self._settings.connections.get(name)

    def get_default_connection(self) -> ComfyUIConnection:
        """Get the default connection."""
        with self._lock:
            return self._settings.connections.get(
                self._settings.default_connection,
                ComfyUIConnection()
            )

    def list_connections(self) -> dict[str, ComfyUIConnection]:
        """Get all connections."""
        with self._lock:
            return dict(self._settings.connections)

    def add_connection(
        self,
        name: str,
        host: str = "127.0.0.1",
        port: int = 8188,
        use_ssl: bool = False
    ) -> ComfyUIConnection:
        """Add a new connection."""
        with self._lock:
            connection = ComfyUIConnection(host=host, port=port, use_ssl=use_ssl)
            self._settings.connections[name] = connection
            self._save()
            logger.info(f"Added connection: {name} ({host}:{port})")
            return connection

    def update_connection(
        self,
        name: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_ssl: Optional[bool] = None
    ) -> Optional[ComfyUIConnection]:
        """Update an existing connection."""
        with self._lock:
            if name not in self._settings.connections:
                return None

            conn = self._settings.connections[name]
            if host is not None:
                conn.host = host
            if port is not None:
                conn.port = port
            if use_ssl is not None:
                conn.use_ssl = use_ssl

            self._save()
            return conn

    def remove_connection(self, name: str) -> bool:
        """Remove a connection."""
        with self._lock:
            if name not in self._settings.connections:
                return False

            if name == self._settings.default_connection:
                logger.warning("Cannot remove default connection")
                return False

            del self._settings.connections[name]
            self._save()
            logger.info(f"Removed connection: {name}")
            return True

    def set_default_connection(self, name: str) -> bool:
        """Set the default connection."""
        with self._lock:
            if name not in self._settings.connections:
                return False

            self._settings.default_connection = name
            self._save()
            return True

    # Job History Settings
    def get_job_history_settings(self) -> JobHistorySettings:
        """Get job history settings."""
        with self._lock:
            return self._settings.job_history

    def update_job_history_settings(
        self,
        enabled: Optional[bool] = None,
        max_jobs: Optional[int] = None,
        auto_cleanup_days: Optional[int] = None,
        store_outputs: Optional[bool] = None,
        store_images: Optional[bool] = None
    ) -> JobHistorySettings:
        """Update job history settings."""
        with self._lock:
            if enabled is not None:
                self._settings.job_history.enabled = enabled
            if max_jobs is not None:
                self._settings.job_history.max_jobs = max_jobs
            if auto_cleanup_days is not None:
                self._settings.job_history.auto_cleanup_days = auto_cleanup_days
            if store_outputs is not None:
                self._settings.job_history.store_outputs = store_outputs
            if store_images is not None:
                self._settings.job_history.store_images = store_images

            self._save()
            return self._settings.job_history

    # Logging Settings
    def get_logging_settings(self) -> LoggingSettings:
        """Get logging settings."""
        with self._lock:
            return self._settings.logging

    def update_logging_settings(
        self,
        level: Optional[str] = None,
        log_to_file: Optional[bool] = None,
        log_file: Optional[str] = None
    ) -> LoggingSettings:
        """Update logging settings."""
        with self._lock:
            if level is not None:
                self._settings.logging.level = level
            if log_to_file is not None:
                self._settings.logging.log_to_file = log_to_file
            if log_file is not None:
                self._settings.logging.log_file = log_file

            self._save()
            self._apply_logging_settings()
            return self._settings.logging

    def _apply_logging_settings(self) -> None:
        """Apply logging settings."""
        log_settings = self._settings.logging
        log_level = getattr(logging, log_settings.level.upper(), logging.INFO)

        # Configure root logger for this package
        pkg_logger = logging.getLogger("comfyui_api_manager")
        pkg_logger.setLevel(log_level)

        # Remove existing handlers
        for handler in pkg_logger.handlers[:]:
            pkg_logger.removeHandler(handler)

        # Add console handler
        console = logging.StreamHandler()
        console.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console.setFormatter(formatter)
        pkg_logger.addHandler(console)

        # Add file handler if enabled
        if log_settings.log_to_file:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_settings.log_file,
                maxBytes=log_settings.max_log_size_mb * 1024 * 1024,
                backupCount=log_settings.backup_count
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            pkg_logger.addHandler(file_handler)

    # Workflows Directory
    def get_workflows_dir(self) -> Path:
        """Get workflows directory."""
        with self._lock:
            return Path(self._settings.workflows_dir)

    def set_workflows_dir(self, path: str) -> None:
        """Set workflows directory."""
        with self._lock:
            self._settings.workflows_dir = path
            self._save()

    # Reset
    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        with self._lock:
            self._settings = PluginSettings()
            self._save()
            logger.info("Reset settings to defaults")

    def export_settings(self) -> dict:
        """Export settings as dictionary."""
        with self._lock:
            return self._settings.to_dict()

    def import_settings(self, data: dict) -> bool:
        """Import settings from dictionary."""
        try:
            with self._lock:
                self._settings = PluginSettings.from_dict(data)
                self._save()
                logger.info("Imported settings")
                return True
        except Exception as e:
            logger.error(f"Failed to import settings: {e}")
            return False


# Global settings manager instance
_global_manager: Optional[SettingsManager] = None


def get_settings_manager(settings_file: Optional[Path] = None) -> SettingsManager:
    """Get or create the global settings manager."""
    global _global_manager

    if _global_manager is None:
        _global_manager = SettingsManager(settings_file=settings_file)

    return _global_manager


def reset_settings_manager() -> None:
    """Reset the global settings manager."""
    global _global_manager
    _global_manager = None
