# ComfyUI API Manager v2.0.0

A comprehensive API manager for ComfyUI with full API integration, MCP (Model Context Protocol) server support, workflow management, and job tracking.

## Features

- **Full ComfyUI API Integration**: REST and WebSocket support for real-time workflow execution
- **MCP Server**: Expose ComfyUI functionality to AI systems via Model Context Protocol
- **Workflow Management**: List, load, validate, and execute workflows programmatically
- **Job Tracking**: Persistent history of job executions with status, outputs, and statistics
- **Multiple Connections**: Manage connections to multiple ComfyUI instances
- **Comprehensive Nodes**: 26+ nodes for API requests, image handling, JSON processing, and more
- **Settings Management**: Configure MCP server, logging, and job history from within ComfyUI

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
requests>=2.31.0
Pillow>=10.0.0
numpy>=1.24.0
pydantic>=2.5.0
mcp>=1.25.0  # Optional, for MCP server support
```

## Node Categories

### API Manager
- **API Request**: Full-featured HTTP client with auth, retry, and response parsing
- **Simple API Request**: Quick GET requests with bearer token support
- **Post Image to API**: Upload images to external APIs
- **Download Image from URL**: Fetch images from URLs

### API Manager/ComfyUI
- **Execute Workflow (ComfyUI)**: Run workflows on remote ComfyUI instances
- **Queue Status (ComfyUI)**: View current queue
- **System Info (ComfyUI)**: Get GPU and system information
- **List Models (ComfyUI)**: List checkpoints, LoRAs, VAEs, etc.
- **Upload Image (ComfyUI)**: Upload images to ComfyUI
- **Download Image (ComfyUI)**: Download generated images
- **Interrupt (ComfyUI)**: Stop current execution
- **Free Memory (ComfyUI)**: Unload models and free VRAM
- **List Workflows (ComfyUI)**: Browse available workflows

### API Manager/Settings
- **API Manager Settings**: View and modify plugin settings
- **Connection Manager**: Add, remove, and test connections
- **Job History**: Browse execution history
- **Job Statistics**: View success rates and timing
- **Clear Job History**: Clean up old jobs
- **MCP Server Control**: Enable/disable MCP server
- **Logging Settings**: Configure log level and file output
- **View Recent Jobs**: Quick job summary

### JSON/Text Processing
- **Text Prompt Combiner**: Template substitution with CLIP encoding
- **Text Template**: Simple string templates
- **JSON Extract**: Path-based value extraction
- **JSON Array Iterator**: Select items from arrays
- **JSON Array Filter**: Filter arrays by conditions
- **JSON Array Slice**: Slice arrays by index
- **JSON Array Map**: Extract fields from array items
- **JSON Array Stats**: Calculate array statistics
- **JSON Merge**: Combine objects/arrays

## MCP Server

The MCP (Model Context Protocol) server allows AI systems like Claude to interact with ComfyUI programmatically.

### Enabling MCP Server

1. Use the **MCP Server Control** node in ComfyUI
2. Set action to "enable"
3. Restart ComfyUI

Or edit `api_manager_settings.json`:
```json
{
  "mcp": {
    "enabled": true,
    "transport": "stdio",
    "host": "127.0.0.1",
    "port": 8765
  }
}
```

### Running MCP Server Standalone

```bash
python -m comfyui_api_manager.mcp_server
```

### MCP Tools Available

| Tool | Description |
|------|-------------|
| `execute_workflow` | Execute a ComfyUI workflow with inputs |
| `generate_image` | Simple text-to-image generation |
| `img2img` | Image-to-image generation |
| `upload_image` | Upload an image to ComfyUI |
| `get_image` | Download an image from ComfyUI |
| `get_queue_status` | Check execution queue |
| `cancel_job` | Cancel a running job |
| `clear_queue` | Clear all queued jobs |
| `interrupt_execution` | Stop current workflow |
| `get_job_history` | View job execution history |
| `get_job_status` | Check specific job status |
| `get_job_statistics` | View overall statistics |
| `get_system_info` | Get GPU/system information |
| `list_models` | List available models |
| `list_nodes` | List available node types |
| `free_memory` | Free GPU memory |
| `list_connections` | View configured connections |
| `test_connection` | Test ComfyUI connection |
| `list_workflows` | List available workflows |
| `get_workflow_templates` | Get workflow templates |
| `validate_workflow` | Validate workflow JSON |
| `analyze_workflow` | Analyze workflow inputs/outputs |

### MCP Resources

| Resource | Description |
|----------|-------------|
| `comfyui://connections` | All configured connections |
| `comfyui://jobs/recent` | Recent job history |
| `comfyui://jobs/statistics` | Job statistics |
| `comfyui://settings` | Current settings |

## Usage Examples

### Basic API Request

```
APIRequestNode:
  api_url: "https://api.example.com/data"
  method: "GET"
  auth_type: "bearer"
  auth_credentials: {"token": "your-token-here"}
  response_path: "data.items"
```

### Execute Remote Workflow

```
ComfyUIExecuteWorkflow:
  workflow_json: <your workflow JSON>
  host: "192.168.1.100"
  port: 8188
  timeout: 300
  wait_for_completion: true
```

### Dynamic Prompt Generation

```
APIRequestNode -> TextPromptCombinerNode -> KSampler

Template: "A $style portrait of $subject in $setting"
API Response: {"style": "watercolor", "subject": "cat", "setting": "garden"}
Result: "A watercolor portrait of cat in garden"
```

### Job Tracking

```
JobHistory:
  status_filter: "failed"
  limit: 20

Output: List of failed jobs with error messages
```

## Configuration

Settings are stored in `api_manager_settings.json`:

```json
{
  "version": "2.0.0",
  "mcp": {
    "enabled": false,
    "host": "127.0.0.1",
    "port": 8765,
    "transport": "stdio"
  },
  "logging": {
    "level": "INFO",
    "log_to_file": false,
    "log_file": "comfyui_api_manager.log"
  },
  "job_history": {
    "enabled": true,
    "max_jobs": 1000,
    "auto_cleanup_days": 30
  },
  "connections": {
    "default": {
      "host": "127.0.0.1",
      "port": 8188,
      "use_ssl": false
    }
  },
  "default_connection": "default"
}
```

## API Reference

### ComfyUI Client

```python
from comfyui_api_manager.comfyui_client import ComfyUIClient, ComfyUIConnection

# Create connection
connection = ComfyUIConnection(host="127.0.0.1", port=8188)
client = ComfyUIClient(connection)

# Execute workflow
async with client:
    result = await client.execute_workflow(workflow_json)
    print(f"Status: {result.status}")
    print(f"Images: {len(result.images)}")
```

### Workflow Manager

```python
from comfyui_api_manager.workflow_manager import WorkflowManager

manager = WorkflowManager()

# Load and modify workflow
workflow = manager.load_workflow("my_workflow.json")
modified = manager.set_text_prompt(workflow.workflow_json, "A beautiful sunset")
modified = manager.set_sampler_settings(modified, steps=30, cfg=7.5)
```

### Job Tracker

```python
from comfyui_api_manager.job_tracker import get_job_tracker

tracker = get_job_tracker()

# Get statistics
stats = tracker.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")

# List recent failures
failures = tracker.get_recent_failures(limit=10)
```

## Troubleshooting

### MCP Server Not Starting

1. Ensure MCP SDK is installed: `pip install mcp>=1.25.0`
2. Check settings: `api_manager_settings.json` -> `mcp.enabled: true`
3. Restart ComfyUI after enabling

### Connection Refused

1. Verify ComfyUI is running on the specified host/port
2. Check firewall settings
3. Use **Connection Manager** node to test connection

### Workflow Execution Timeout

1. Increase `timeout` parameter
2. Check GPU memory with **System Info** node
3. Use **Free Memory** node before large workflows

### Missing Nodes

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please submit issues and pull requests to the GitHub repository.

## License

MIT License - see LICENSE file for details.

## Changelog

### v2.0.0
- Complete rewrite with async API client
- Added MCP server for AI integration
- Added job tracking with persistent history
- Added multiple connection support
- Added 26+ new nodes
- Added comprehensive settings management
- Improved error handling throughout
- Updated for ComfyUI 2025/2026 compatibility

### v1.0.0
- Initial release with basic API nodes
