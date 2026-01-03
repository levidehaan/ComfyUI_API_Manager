# ComfyUI API Manager v2.0.0

A server-side plugin for ComfyUI that provides REST API endpoints for external workflow control, workflow analysis, and MCP (Model Context Protocol) integration for AI systems.

**This is a server-only plugin** - no workflow nodes are added. All functionality is exposed via API endpoints and configured through ComfyUI's Settings panel.

## Features

- **REST API Endpoints**: List, analyze, and execute workflows via HTTP
- **Workflow Analysis**: Automatically detect workflow inputs (images, text prompts, numbers, models)
- **Image Input Support**: Send base64-encoded images to workflows with image inputs
- **MCP Server**: Expose ComfyUI to AI systems via Model Context Protocol
- **Job Tracking**: Persistent history of API-triggered job executions
- **Settings Panel**: Configure plugin options in ComfyUI's Settings menu
- **Real-time Updates**: WebSocket support for execution progress

## Installation

### Using ComfyUI Manager (Recommended)

Search for "ComfyUI API Manager" in ComfyUI Manager and install.

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/levidehaan/ComfyUI_API_Manager.git
cd ComfyUI_API_Manager
pip install -r requirements.txt
```

### Dependencies

```
aiohttp>=3.9.0
aiofiles>=23.2.1
websockets>=12.0
Pillow>=10.0.0
pydantic>=2.5.0
mcp>=1.25.0  # Optional, for MCP server support
```

## API Endpoints

All endpoints are prefixed with `/api-manager/`.

### Plugin Information

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api-manager/info` | Get plugin version and available endpoints |
| GET | `/api-manager/settings` | Get current settings |
| POST | `/api-manager/settings` | Update settings |

### Workflow Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api-manager/workflows` | List all available workflows |
| GET | `/api-manager/workflows/{name}` | Get workflow JSON by name |
| GET | `/api-manager/workflows/{name}/analyze` | Analyze workflow inputs/outputs |
| POST | `/api-manager/workflows/{name}/execute` | Execute workflow with inputs |

### Job Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api-manager/jobs` | List job history |
| GET | `/api-manager/jobs/{job_id}` | Get job details |
| GET | `/api-manager/jobs/{job_id}/status` | Get job status |
| DELETE | `/api-manager/jobs/{job_id}` | Cancel a job |

### MCP Server Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api-manager/mcp/status` | Get MCP server status |
| POST | `/api-manager/mcp/start` | Start MCP server |
| POST | `/api-manager/mcp/stop` | Stop MCP server |

## Usage Examples

### List Available Workflows

```bash
curl http://localhost:8188/api-manager/workflows
```

Response:
```json
{
  "workflows": [
    {
      "name": "txt2img_basic",
      "path": "txt2img_basic.json",
      "size": 4523,
      "modified": "2024-12-15T10:30:00"
    }
  ],
  "count": 1
}
```

### Analyze Workflow Inputs

```bash
curl http://localhost:8188/api-manager/workflows/txt2img_basic/analyze
```

Response:
```json
{
  "workflow_name": "txt2img_basic",
  "inputs": {
    "images": [
      {"node_id": "3", "node_type": "LoadImage", "name": "image"}
    ],
    "text": [
      {"node_id": "6", "node_type": "CLIPTextEncode", "name": "positive_prompt"},
      {"node_id": "7", "node_type": "CLIPTextEncode", "name": "negative_prompt"}
    ],
    "numbers": [
      {"node_id": "10", "input_name": "steps", "default": 20},
      {"node_id": "10", "input_name": "cfg", "default": 7.0},
      {"node_id": "10", "input_name": "seed", "default": 0}
    ],
    "models": [
      {"node_id": "4", "node_type": "CheckpointLoaderSimple", "input_name": "ckpt_name"}
    ]
  },
  "outputs": [
    {"node_id": "9", "node_type": "SaveImage", "output_type": "images"}
  ]
}
```

### Execute Workflow with Inputs

```bash
curl -X POST http://localhost:8188/api-manager/workflows/txt2img_basic/execute \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "positive_prompt": "A beautiful sunset over mountains",
      "negative_prompt": "blurry, low quality",
      "steps": 25,
      "cfg": 7.5,
      "seed": 12345
    }
  }'
```

### Execute with Image Input

```bash
curl -X POST http://localhost:8188/api-manager/workflows/img2img/execute \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "image": "data:image/png;base64,iVBORw0KGgo...",
      "denoise": 0.7,
      "prompt": "oil painting style"
    }
  }'
```

Response:
```json
{
  "success": true,
  "job_id": "abc123-def456",
  "prompt_id": "xyz789",
  "status": "queued",
  "message": "Workflow queued for execution"
}
```

### Check Job Status

```bash
curl http://localhost:8188/api-manager/jobs/abc123-def456/status
```

## Settings Panel

After installing, open ComfyUI's Settings (gear icon) and find the **API Manager** section:

### MCP Server Settings
- **Enable MCP Server**: Toggle MCP server on/off
- **MCP Server Port**: Port number (default: 8765)
- **MCP Transport**: Protocol (stdio or streamable-http)

### API Settings
- **Enable Workflow Listing API**: Allow listing workflows
- **Enable Workflow Execution API**: Allow executing workflows
- **Max Concurrent Jobs**: Limit simultaneous executions

### Job History Settings
- **Enable Job History**: Track execution history
- **Max History Size**: Maximum jobs to retain

### Logging Settings
- **Log Level**: DEBUG, INFO, WARNING, or ERROR

## MCP Server

The MCP (Model Context Protocol) server allows AI systems like Claude to interact with ComfyUI.

### Enabling MCP Server

1. Open ComfyUI Settings
2. Navigate to API Manager section
3. Enable "MCP Server"
4. Set desired port (default: 8765)
5. Restart ComfyUI or use API to start

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `list_workflows` | List available workflows |
| `analyze_workflow` | Detect workflow inputs/outputs |
| `execute_workflow` | Run workflow with inputs |
| `generate_image` | Simple text-to-image generation |
| `img2img` | Image-to-image generation |
| `upload_image` | Upload an image to ComfyUI |
| `get_image` | Download generated images |
| `get_queue_status` | Check execution queue |
| `cancel_job` | Cancel a running job |
| `get_job_history` | View job execution history |
| `get_system_info` | Get GPU/system information |
| `list_models` | List available models |
| `free_memory` | Free GPU memory |

### Connecting AI to MCP Server

For Claude Code or other MCP-compatible AI systems:

```json
{
  "mcpServers": {
    "comfyui": {
      "command": "python",
      "args": ["-m", "comfyui_api_manager.mcp_server"],
      "env": {
        "COMFYUI_HOST": "127.0.0.1",
        "COMFYUI_PORT": "8188"
      }
    }
  }
}
```

## Configuration File

Settings are persisted in `api_manager_settings.json`:

```json
{
  "version": "2.0.0",
  "mcp": {
    "enabled": false,
    "host": "127.0.0.1",
    "port": 8765,
    "transport": "stdio"
  },
  "api": {
    "enable_workflow_list": true,
    "enable_execution": true,
    "max_concurrent_jobs": 3
  },
  "logging": {
    "level": "INFO"
  },
  "job_history": {
    "enabled": true,
    "max_jobs": 1000
  }
}
```

## Python API

### ComfyUI Client

```python
from comfyui_api_manager.comfyui_client import ComfyUIClient, ComfyUIConnection

connection = ComfyUIConnection(host="127.0.0.1", port=8188)
client = ComfyUIClient(connection)

async with client:
    # Execute workflow
    result = await client.execute_workflow(workflow_json)
    print(f"Status: {result.status}")
    print(f"Images: {len(result.images)}")

    # Upload image
    await client.upload_image(image_data, "input.png")

    # Get system info
    info = await client.get_system_stats()
```

### Workflow Manager

```python
from comfyui_api_manager.workflow_manager import WorkflowManager

manager = WorkflowManager()

# Load and modify workflow
workflow = manager.load_workflow("my_workflow.json")
modified = manager.set_text_prompt(workflow.workflow_json, "A sunset")
modified = manager.set_sampler_settings(modified, steps=30, cfg=7.5)
```

### Job Tracker

```python
from comfyui_api_manager.job_tracker import get_job_tracker

tracker = get_job_tracker()

# Get statistics
stats = tracker.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")

# List recent jobs
jobs = tracker.get_jobs(limit=10, status="completed")
```

## Troubleshooting

### API Endpoints Not Available

1. Ensure plugin is installed in `custom_nodes/` directory
2. Restart ComfyUI after installation
3. Check ComfyUI console for initialization errors

### MCP Server Not Starting

1. Install MCP SDK: `pip install mcp>=1.25.0`
2. Enable in Settings: API Manager > Enable MCP Server
3. Check port is not in use: `netstat -an | grep 8765`

### Workflow Execution Fails

1. Verify workflow JSON is valid
2. Check required models are installed
3. Ensure input types match workflow requirements
4. Check ComfyUI console for error details

### Image Upload Issues

1. Use base64-encoded image data with proper prefix
2. Supported formats: PNG, JPEG, WebP
3. Check image size limits

## Contributing

Contributions are welcome! Please submit issues and pull requests to the GitHub repository.

## License

MIT License - see LICENSE file for details.

## Changelog

### v2.0.0
- Complete rewrite as server-only plugin
- Removed all workflow nodes (pure API approach)
- Added REST API endpoints for workflow management
- Added workflow input/output analysis
- Added MCP server for AI integration
- Added Settings panel in ComfyUI UI
- Added job tracking with persistent history
- Updated for ComfyUI 2025/2026 compatibility

### v1.0.0
- Initial release with basic API nodes
