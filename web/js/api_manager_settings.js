/**
 * ComfyUI API Manager - Settings Panel
 *
 * Adds settings to ComfyUI's settings panel for configuring:
 * - MCP Server (enable/disable, port, transport)
 * - API Manager options
 * - Workflow visibility settings
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// API Manager extension
const API_MANAGER_EXTENSION = {
  name: "ComfyUI.APIManager",

  settings: [
    // MCP Server Settings
    {
      id: "APIManager.MCP.enabled",
      name: "Enable MCP Server",
      type: "boolean",
      defaultValue: false,
      tooltip: "Enable the Model Context Protocol server for AI integration",
      onChange: async (newVal, oldVal) => {
        if (newVal !== oldVal) {
          await updateMCPSetting("enabled", newVal);
        }
      },
    },
    {
      id: "APIManager.MCP.port",
      name: "MCP Server Port",
      type: "number",
      defaultValue: 8765,
      attrs: {
        min: 1024,
        max: 65535,
        step: 1,
      },
      tooltip: "Port number for the MCP server (requires restart)",
      onChange: async (newVal, oldVal) => {
        if (newVal !== oldVal) {
          await updateMCPSetting("port", newVal);
        }
      },
    },
    {
      id: "APIManager.MCP.transport",
      name: "MCP Transport",
      type: "combo",
      defaultValue: "stdio",
      options: ["stdio", "streamable-http"],
      tooltip: "Transport protocol for MCP server",
      onChange: async (newVal, oldVal) => {
        if (newVal !== oldVal) {
          await updateMCPSetting("transport", newVal);
        }
      },
    },

    // API Settings
    {
      id: "APIManager.API.enableWorkflowList",
      name: "Enable Workflow Listing API",
      type: "boolean",
      defaultValue: true,
      tooltip: "Allow listing workflows via API endpoint",
    },
    {
      id: "APIManager.API.enableExecution",
      name: "Enable Workflow Execution API",
      type: "boolean",
      defaultValue: true,
      tooltip: "Allow executing workflows via API endpoint",
    },
    {
      id: "APIManager.API.maxConcurrentJobs",
      name: "Max Concurrent Jobs",
      type: "number",
      defaultValue: 3,
      attrs: {
        min: 1,
        max: 10,
        step: 1,
      },
      tooltip: "Maximum number of concurrent workflow executions",
    },

    // Job History Settings
    {
      id: "APIManager.JobHistory.enabled",
      name: "Enable Job History",
      type: "boolean",
      defaultValue: true,
      tooltip: "Track execution history for API-triggered jobs",
    },
    {
      id: "APIManager.JobHistory.maxJobs",
      name: "Max History Size",
      type: "number",
      defaultValue: 1000,
      attrs: {
        min: 100,
        max: 10000,
        step: 100,
      },
      tooltip: "Maximum number of jobs to keep in history",
    },

    // Logging Settings
    {
      id: "APIManager.Logging.level",
      name: "Log Level",
      type: "combo",
      defaultValue: "INFO",
      options: ["DEBUG", "INFO", "WARNING", "ERROR"],
      tooltip: "Logging verbosity level",
    },
  ],

  async setup() {
    console.log("[API Manager] Extension loaded");

    // Load current settings from server
    await this.loadServerSettings();

    // Add API Manager status to UI
    this.addStatusIndicator();
  },

  async loadServerSettings() {
    try {
      const response = await api.fetchApi("/api-manager/settings");
      if (response.ok) {
        const data = await response.json();
        console.log("[API Manager] Loaded settings:", data);

        // Sync settings with ComfyUI settings manager
        if (data.mcp) {
          app.extensionManager.setting.set(
            "APIManager.MCP.enabled",
            data.mcp.enabled
          );
          app.extensionManager.setting.set(
            "APIManager.MCP.port",
            data.mcp.port
          );
          app.extensionManager.setting.set(
            "APIManager.MCP.transport",
            data.mcp.transport
          );
        }
      }
    } catch (error) {
      console.warn("[API Manager] Could not load settings:", error);
    }
  },

  addStatusIndicator() {
    // Add a small indicator to show API Manager status
    const statusDiv = document.createElement("div");
    statusDiv.id = "api-manager-status";
    statusDiv.style.cssText = `
      position: fixed;
      bottom: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: #fff;
      padding: 5px 10px;
      border-radius: 4px;
      font-size: 11px;
      z-index: 1000;
      display: none;
    `;
    statusDiv.innerHTML = "API Manager: Active";
    document.body.appendChild(statusDiv);

    // Check if MCP is enabled and show indicator
    this.updateStatusIndicator();
  },

  async updateStatusIndicator() {
    const statusDiv = document.getElementById("api-manager-status");
    if (!statusDiv) return;

    try {
      const response = await api.fetchApi("/api-manager/mcp/status");
      if (response.ok) {
        const data = await response.json();
        if (data.enabled) {
          statusDiv.style.display = "block";
          statusDiv.innerHTML = data.running
            ? `MCP: Running (port ${data.port})`
            : `MCP: Enabled (port ${data.port})`;
          statusDiv.style.background = data.running
            ? "rgba(0, 128, 0, 0.7)"
            : "rgba(128, 128, 0, 0.7)";
        } else {
          statusDiv.style.display = "none";
        }
      }
    } catch (error) {
      // Silently ignore
    }
  },
};

// Helper function to update MCP settings on server
async function updateMCPSetting(key, value) {
  try {
    const response = await api.fetchApi("/api-manager/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mcp: { [key]: value },
      }),
    });

    if (response.ok) {
      console.log(`[API Manager] Updated MCP ${key}:`, value);

      // If enabling/disabling MCP, update status indicator
      if (key === "enabled") {
        const ext = app.extensions.find((e) => e.name === "ComfyUI.APIManager");
        if (ext) {
          ext.updateStatusIndicator();
        }
      }
    } else {
      console.error("[API Manager] Failed to update setting");
    }
  } catch (error) {
    console.error("[API Manager] Error updating setting:", error);
  }
}

// Register the extension
app.registerExtension(API_MANAGER_EXTENSION);

// Add menu items for API Manager
const originalSetup = API_MANAGER_EXTENSION.setup;
API_MANAGER_EXTENSION.setup = async function () {
  await originalSetup.call(this);

  // Add to the top menu (if ComfyUI supports it)
  if (app.ui && app.ui.menu) {
    app.ui.menu.addEntry({
      name: "API Manager",
      items: [
        {
          name: "View API Info",
          callback: async () => {
            const response = await api.fetchApi("/api-manager/info");
            if (response.ok) {
              const data = await response.json();
              alert(
                `ComfyUI API Manager v${data.version}\n\nEndpoints:\n${data.endpoints.join("\n")}`
              );
            }
          },
        },
        {
          name: "View Job History",
          callback: async () => {
            const response = await api.fetchApi("/api-manager/jobs?limit=10");
            if (response.ok) {
              const data = await response.json();
              console.log("[API Manager] Recent Jobs:", data.jobs);
              alert(
                `Recent Jobs (${data.count}):\n\n${data.jobs
                  .map(
                    (j) =>
                      `${j.workflow_name}: ${j.status} (${j.created_at.split("T")[0]})`
                  )
                  .join("\n")}`
              );
            }
          },
        },
        {
          name: "Start MCP Server",
          callback: async () => {
            const response = await api.fetchApi("/api-manager/mcp/start", {
              method: "POST",
            });
            const data = await response.json();
            alert(data.success ? data.message : `Error: ${data.error}`);
          },
        },
        {
          name: "Stop MCP Server",
          callback: async () => {
            const response = await api.fetchApi("/api-manager/mcp/stop", {
              method: "POST",
            });
            const data = await response.json();
            alert(data.success ? data.message : `Error: ${data.error}`);
          },
        },
      ],
    });
  }
};

console.log("[API Manager] Settings extension registered");
