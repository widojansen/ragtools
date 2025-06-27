# MCP Client Discovery Process Documentation

## Overview

The MCP (Model Context Protocol) client discovery process is responsible for finding, connecting to, and discovering the capabilities of configured MCP servers. This process determines what tools and resources are available for the chatbot to use.

This document provides a detailed walkthrough of how the MCP client discovers servers, establishes connections, and maps available capabilities.

## 🔍 MCP Client Discovery Process

### 1. Initialization & Configuration Loading

#### Step 1: MCPClient Initialization

```python
class MCPClient:
    def __init__(self, config_path: Optional[str] = None):
        self.servers: Dict[str, MCPServerConfig] = {}           # Server configurations
        self.connections: Dict[str, MCPConnection] = {}         # Active connections
        self.config_path = config_path or "mcp_config.json"    # Config file path
        self._load_config()                                     # Load and parse config
```

**What happens:**
- Initializes empty dictionaries for servers and connections
- Sets default configuration file path (`mcp_config.json`)
- Immediately loads configuration from the file

#### Step 2: Configuration File Discovery

```python
def _load_config(self):
    """Load MCP server configurations from file"""
    config_file = Path(self.config_path)
    logger.info(f"Looking for MCP config at: {config_file.absolute()}")
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                logger.info(f"Loaded config data: {config_data}")
                
                for server_name, server_config in config_data.get('servers', {}).items():
                    self.servers[server_name] = MCPServerConfig(
                        name=server_name,
                        **server_config
                    )
                    logger.info(f"Server {server_name} is {'enabled' if server_config.get('enabled') else 'disabled'}")
                    
            logger.info(f"Loaded {len(self.servers)} MCP server configurations")
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
    else:
        logger.warning(f"Config file not found at {config_file.absolute()}, creating default config")
        self._create_default_config()
```

**Process:**
1. **File Location**: Looks for `mcp_config.json` in current working directory
2. **Path Logging**: Logs the absolute path being checked for debugging
3. **JSON Parsing**: Loads and parses the configuration file
4. **Server Creation**: Creates `MCPServerConfig` objects for each server definition
5. **Status Tracking**: Logs enabled/disabled status for each server

#### Example Configuration Structure

```json
{
  "servers": {
    "Terminal Server": {
      "command": "/Users/widojansen/.asdf/shims/uv",
      "args": ["run", "/Users/widojansen/Projects/Agents/MCP/shellserver/server.py"],
      "env": {},
      "description": "Terminal command execution on local computer",
      "enabled": true
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/widojansen"],
      "env": {},
      "description": "Access filesystem resources",
      "enabled": false
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-brave-api-key"
      },
      "description": "Web search capabilities",
      "enabled": false
    }
  }
}
```

#### Step 3: Server Configuration Parsing

```python
for server_name, server_config in config_data.get('servers', {}).items():
    self.servers[server_name] = MCPServerConfig(
        name=server_name,
        command=server_config['command'],           # Executable command
        args=server_config['args'],                # Command arguments
        env=server_config.get('env', {}),          # Environment variables
        description=server_config.get('description', ''),  # Human description
        enabled=server_config.get('enabled', False)        # Enable/disable flag
    )
    logger.info(f"Server {server_name} is {'enabled' if server_config.get('enabled') else 'disabled'}")
```

**MCPServerConfig Structure:**
- **name**: Server identifier (e.g., "Terminal Server")
- **command**: Path to executable (e.g., "/usr/bin/python")
- **args**: Command line arguments (e.g., ["server.py", "--port", "8080"])
- **env**: Environment variables (e.g., {"API_KEY": "secret"})
- **description**: Human-readable description
- **enabled**: Whether to connect to this server

### 2. Server Connection Process

#### Step 1: Connection Initiation

```python
async def connect_all_servers(self):
    """Connect to all enabled MCP servers"""
    logger.info(f"Total servers configured: {len(self.servers)}")
    
    enabled_servers = [name for name, config in self.servers.items() if config.enabled]
    disabled_servers = [name for name, config in self.servers.items() if not config.enabled]
    
    logger.info(f"Enabled servers: {enabled_servers}")
    logger.info(f"Disabled servers: {disabled_servers}")
    
    tasks = []
    for server_name, server_config in self.servers.items():
        if server_config.enabled:
            logger.info(f"Attempting to connect to enabled server: {server_name}")
            tasks.append(self.connect_server(server_name))
        else:
            logger.info(f"Server {server_name} is disabled")

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        connected = sum(1 for r in results if r is True)
        logger.info(f"Connected to {connected}/{len(tasks)} MCP servers")
    else:
        logger.warning("No enabled servers found to connect to")
```

**Process:**
1. **Server Enumeration**: Scans all configured servers
2. **Filter by Status**: Separates enabled from disabled servers
3. **Parallel Connection**: Creates async tasks for all enabled servers
4. **Result Aggregation**: Waits for all connections and reports success rate

#### Step 2: Individual Server Connection

```python
async def connect_server(self, server_name: str) -> bool:
    """Connect to a specific MCP server"""
    if server_name not in self.servers:
        logger.error(f"Server {server_name} not configured")
        return False

    server_config = self.servers[server_name]
    
    if not server_config.enabled:
        logger.info(f"Server {server_name} is disabled")
        return False

    # Disconnect if already connected
    if server_name in self.connections:
        await self.disconnect_server(server_name)

    connection = MCPConnection(server_config)
    success = await connection.connect()
    
    if success:
        self.connections[server_name] = connection
        logger.info(f"Successfully connected to MCP server: {server_name}")
        return True
    else:
        logger.error(f"Failed to connect to MCP server: {server_name}")
        return False
```

**What happens:**
1. **Configuration Lookup**: Finds server config by name
2. **Status Check**: Verifies server is enabled
3. **Cleanup**: Disconnects existing connection if present
4. **Connection Creation**: Creates new `MCPConnection` instance
5. **Connection Attempt**: Attempts to establish connection
6. **Result Storage**: Stores successful connection for future use

#### Step 3: MCPConnection.connect() Process

```python
async def connect(self) -> bool:
    """Connect to the MCP server"""
    try:
        logger.info(f"Starting MCP server: {self.config.name}")
        logger.info(f"Command: {self.config.command} {' '.join(self.config.args)}")
        
        # Prepare environment
        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)
        
        # Start the server process
        self.process = subprocess.Popen(
            [self.config.command] + self.config.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            env=env
        )
        
        # Wait for process to start
        await asyncio.sleep(0.5)
        
        # Check if process is running
        if self.process.poll() is not None:
            stderr_output = self.process.stderr.read() if self.process.stderr else "No stderr"
            stdout_output = self.process.stdout.read() if self.process.stdout else "No stdout"
            logger.error(f"MCP server {self.config.name} died immediately")
            logger.error(f"Command: {self.config.command} {' '.join(self.config.args)}")
            logger.error(f"Stderr: {stderr_output}")
            logger.error(f"Stdout: {stdout_output}")
            logger.error(f"Return code: {self.process.returncode}")
            return False
        
        logger.info(f"MCP server {self.config.name} process started")
        
        # Send initialization request
        success = await self._initialize()
        if success:
            # Get server capabilities
            await self._get_capabilities()
            self.connected = True
            logger.info(f"Successfully connected to MCP server: {self.config.name}")
            return True
        else:
            logger.error(f"Failed to initialize MCP server: {self.config.name}")
            await self.disconnect()
            return False
            
    except Exception as e:
        logger.error(f"Error connecting to MCP server {self.config.name}: {e}")
        await self.disconnect()
        return False
```

**Detailed Process:**
1. **Environment Setup**: Merges custom environment variables with system environment
2. **Process Launch**: Starts MCP server as subprocess with stdin/stdout/stderr pipes
3. **Health Check**: Waits briefly then verifies process didn't crash immediately
4. **Error Logging**: Captures and logs detailed error information if process fails
5. **Protocol Initialization**: Sends MCP initialization handshake
6. **Capability Discovery**: Requests server's available tools and resources
7. **Connection Finalization**: Marks connection as established and ready

### 3. MCP Protocol Initialization

#### Step 1: Initialize Request

```python
async def _initialize(self) -> bool:
    """Send initialization request to server"""
    init_request = {
        "jsonrpc": "2.0",
        "id": self._get_next_id(),
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",           # MCP protocol version
            "capabilities": {
                "roots": {"listChanged": True},        # Client can handle root changes
                "sampling": {}                         # Client supports sampling
            },
            "clientInfo": {
                "name": "ragtools-mcp-client",         # Client identification
                "version": "1.0.0"                     # Client version
            }
        }
    }
    
    response = await self._send_request(init_request)
    return response is not None and "result" in response
```

**Protocol Handshake:**
1. **JSON-RPC 2.0**: Uses standard JSON-RPC protocol
2. **Version Declaration**: Declares supported MCP protocol version
3. **Capability Advertisement**: Tells server what client capabilities are supported
4. **Client Identification**: Provides client name and version for logging
5. **Response Validation**: Waits for successful acknowledgment from server

#### Step 2: JSON-RPC Communication

```python
async def _send_request(self, request: Dict, timeout: float = 10.0) -> Optional[Dict]:
    """Send a JSON-RPC request to the server"""
    if not self.process or self.process.poll() is not None:
        logger.error(f"Process not running for {self.config.name}")
        return None
    
    try:
        # Send request
        request_json = json.dumps(request) + '\n'
        self.process.stdin.write(request_json)
        self.process.stdin.flush()
        
        # Wait for response with timeout
        response_line = await asyncio.wait_for(
            asyncio.to_thread(self.process.stdout.readline),
            timeout=timeout
        )
        
        if not response_line:
            logger.error(f"No response from {self.config.name}")
            return None
        
        response = json.loads(response_line.strip())
        
        # Check for JSON-RPC errors
        if "error" in response:
            logger.error(f"JSON-RPC error from {self.config.name}: {response['error']}")
            return None
            
        return response
        
    except asyncio.TimeoutError:
        logger.error(f"Request timeout for {self.config.name}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from {self.config.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Request failed for {self.config.name}: {e}")
        return None
```

**Communication Process:**
1. **Process Health Check**: Verifies subprocess is still running
2. **Request Serialization**: Converts request dict to JSON with newline
3. **Transmission**: Writes to server's stdin pipe
4. **Response Waiting**: Waits for response on stdout with timeout
5. **Response Parsing**: Parses JSON response and checks for errors
6. **Error Handling**: Comprehensive error handling for all failure modes

### 4. Capability Discovery

#### Step 1: Tools Discovery

```python
async def _get_capabilities(self):
    """Get server capabilities (tools and resources)"""
    # Get tools
    tools_request = {
        "jsonrpc": "2.0",
        "id": self._get_next_id(),
        "method": "tools/list"
    }
    
    tools_response = await self._send_request(tools_request)
    if tools_response and "result" in tools_response:
        self.capabilities["tools"] = tools_response["result"].get("tools", [])
        logger.info(f"Server {self.config.name} reported {len(self.capabilities['tools'])} tools")
        
        # Log each tool for debugging
        for tool in self.capabilities["tools"]:
            logger.info(f"  Tool: {tool.get('name')} - {tool.get('description', 'No description')}")
    else:
        logger.warning(f"Failed to get tools from {self.config.name}")
        self.capabilities["tools"] = []
```

**Tools Discovery Process:**
1. **Tools List Request**: Sends `tools/list` JSON-RPC method
2. **Response Processing**: Extracts tools array from response
3. **Capability Storage**: Stores tools in connection's capabilities
4. **Detailed Logging**: Logs each discovered tool with name and description
5. **Fallback Handling**: Sets empty list if discovery fails

#### Example Tools Response

```json
{
  "jsonrpc": "2.0",
  "id": 123,
  "result": {
    "tools": [
      {
        "name": "terminal_cmd",
        "description": "Execute a terminal command asynchronously and return its output",
        "inputSchema": {
          "type": "object",
          "properties": {
            "command": {
              "type": "string",
              "description": "The terminal command to execute"
            }
          },
          "required": ["command"],
          "additionalProperties": false
        }
      },
      {
        "name": "read_file",
        "description": "Read any file from the filesystem",
        "inputSchema": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "Path to the file to read (can be relative or absolute)"
            }
          },
          "required": ["file_path"],
          "additionalProperties": false
        }
      },
      {
        "name": "create_test_file",
        "description": "Create a test file on desktop to verify resource functionality",
        "inputSchema": {
          "type": "object",
          "properties": {},
          "additionalProperties": false
        }
      },
      {
        "name": "test_tool",
        "description": "Download content from a URL using curl and return the downloaded data as a string",
        "inputSchema": {
          "type": "object",
          "properties": {
            "url": {
              "type": "string",
              "description": "The URL to download",
              "default": "https://gist.github.com/widojansen/dcfc4da1109a8d95d8971992508a7975"
            }
          },
          "required": [],
          "additionalProperties": false
        }
      }
    ]
  }
}
```

#### Step 2: Resources Discovery

```python
# Get resources
resources_request = {
    "jsonrpc": "2.0",
    "id": self._get_next_id(),
    "method": "resources/list"
}

resources_response = await self._send_request(resources_request)
if resources_response and "result" in resources_response:
    self.capabilities["resources"] = resources_response["result"].get("resources", [])
    logger.info(f"Server {self.config.name} reported {len(self.capabilities['resources'])} resources")
    
    # Log each resource for debugging
    for resource in self.capabilities["resources"]:
        logger.info(f"  Resource: {resource.get('name')} - {resource.get('uri')}")
else:
    logger.warning(f"Failed to get resources from {self.config.name}")
    self.capabilities["resources"] = []
```

#### Example Resources Response

```json
{
  "jsonrpc": "2.0", 
  "id": 124,
  "result": {
    "resources": [
      {
        "uri": "file:///desktop/mcpreadme.md",
        "name": "MCP Readme",
        "description": "Contents of ~/Desktop/mcpreadme.md",
        "mimeType": "text/markdown"
      }
    ]
  }
}
```

### 5. Capability Aggregation & Access

#### Step 1: Client-Level Aggregation

```python
def get_available_tools(self) -> Dict[str, List[Dict]]:
    """Get all available tools from connected servers"""
    return {
        server_name: connection.get_tools()
        for server_name, connection in self.connections.items()
        if connection.connected
    }

def get_available_resources(self) -> Dict[str, List[Dict]]:
    """Get all available resources from connected servers"""
    return {
        server_name: connection.get_resources()
        for server_name, connection in self.connections.items()
        if connection.connected
    }
```

**Aggregation Process:**
1. **Connection Filtering**: Only includes connected servers
2. **Tool Collection**: Gathers tools from all active connections
3. **Server Grouping**: Organizes tools by server name
4. **Live Status**: Respects current connection status

#### Step 2: Connection-Level Access

```python
def get_tools(self) -> List[Dict[str, Any]]:
    """Get available tools from this connection"""
    return self.capabilities.get("tools", [])

def get_resources(self) -> List[Dict[str, Any]]:
    """Get available resources from this connection"""
    return self.capabilities.get("resources", [])
```

#### Aggregated Result Structure

```python
# Example of get_available_tools() output:
{
    "Terminal Server": [
        {
            "name": "terminal_cmd",
            "description": "Execute a terminal command asynchronously and return its output",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        },
        {
            "name": "read_file", 
            "description": "Read any file from the filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"]
            }
        }
    ],
    "filesystem": [
        {
            "name": "list_directory",
            "description": "Get detailed listing of files and directories",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    ]
}
```

### 6. Debug & Discovery Information

The system provides comprehensive discovery debugging through multiple interfaces:

#### Connection Status Tracking

```python
def get_server_status(self) -> Dict[str, str]:
    """Get connection status of all servers"""
    return {
        name: "connected" if name in self.connections and self.connections[name].connected else "disconnected"
        for name in self.servers.keys()
    }
```

#### Capability Summary in UI

```python
# In chatbot_mcp.py debug output:
mcp_client = st.session_state.mcp_client

st.write("**Debug Info:**")
st.write(f"- Total configured servers: {len(mcp_client.servers)}")
st.write(f"- Server configs: {list(mcp_client.servers.keys())}")
st.write(f"- Active connections: {list(mcp_client.connections.keys())}")
st.write(f"- Connected servers: {[name for name, conn in mcp_client.connections.items() if conn.connected]}")

available_tools = mcp_client.get_available_tools()
st.write(f"**Available Tools:** {available_tools}")

for server_name, tools in available_tools.items():
    st.write(f"**{server_name}:** ({len(tools)} tools)")
    if not tools:
        st.write("  - No tools found!")
    else:
        for tool in tools:
            st.write(f"  - {tool.get('name')} ({tool.get('description', 'No description')})")
```

### 7. Error Handling & Troubleshooting

#### Common Discovery Issues

##### 1. Server Process Fails to Start

```python
if self.process.poll() is not None:
    stderr_output = self.process.stderr.read() if self.process.stderr else "No stderr"
    stdout_output = self.process.stdout.read() if self.process.stdout else "No stdout"
    logger.error(f"MCP server {self.config.name} died immediately")
    logger.error(f"Command: {self.config.command} {' '.join(self.config.args)}")
    logger.error(f"Stderr: {stderr_output}")
    logger.error(f"Stdout: {stdout_output}")
    logger.error(f"Return code: {self.process.returncode}")
    return False
```

**Common Causes:**
- Invalid command path
- Missing dependencies
- Permission issues
- Port conflicts
- Environment variable problems

##### 2. Tools/List Returns Empty

```python
if not tools_response or "result" not in tools_response:
    logger.warning(f"Server {self.config.name} failed to list tools")
    self.capabilities["tools"] = []
```

**Common Causes:**
- Server doesn't implement `tools/list` method
- Server started but not ready yet
- JSON-RPC communication issues
- Server internal errors

##### 3. Connection Timeout

```python
try:
    response_line = await asyncio.wait_for(
        asyncio.to_thread(self.process.stdout.readline),
        timeout=timeout
    )
except asyncio.TimeoutError:
    logger.error(f"Request timeout for {self.config.name}")
    return None
```

**Common Causes:**
- Server overloaded or slow
- Network issues (for remote servers)
- Server hanging or blocked
- Resource contention

#### Debug Information Available

```python
# Raw connection debugging in UI test:
st.write("**Debug Info:**")
st.write(f"Connected servers: {list(mcp_client.connections.keys())}")
st.write(f"Server status: {[name for name, conn in mcp_client.connections.items() if conn.connected]}")

# Raw tools data
st.write("**All Available Tools:**")
st.write(f"Raw tools data: {available_tools}")

# Direct connection inspection
if server_name in mcp_client.connections:
    connection = mcp_client.connections[server_name]
    st.write(f"  - Connection connected: {connection.connected}")
    st.write(f"  - Connection capabilities: {getattr(connection, 'capabilities', 'None')}")
    
    # Direct tools call test
    try:
        direct_tools = connection.get_tools()
        st.write(f"  - Direct tools call result: {direct_tools}")
        
        # Raw JSON-RPC test
        async def test_tools_list():
            request = {
                "jsonrpc": "2.0",
                "id": 999,
                "method": "tools/list"
            }
            response = await connection._send_request(request)
            return response
        
        raw_response = asyncio.run(test_tools_list())
        st.write(f"  - Raw tools/list response: {raw_response}")
        
    except Exception as e:
        st.write(f"  - Direct tools call failed: {e}")
```

### 8. Fallback & Recovery Mechanisms

#### Tool Discovery Fallback

```python
# In enhance_query_with_mcp:
if 'Terminal Server' in available_tools and not available_tools['Terminal Server']:
    logger.info("Terminal Server found but no tools listed - adding fallback tools")
    available_tools['Terminal Server'] = [
        {
            'name': 'terminal_cmd', 
            'description': 'Execute terminal commands', 
            'inputSchema': {
                'properties': {
                    'command': {'type': 'string'}
                }
            }
        }
    ]
```

#### Manual Tool Testing

```python
# Test known tool names even when discovery fails
if not command_tool and 'Terminal Server' in available_tools:
    st.write("**Trying known tool names for Terminal Server...**")
    known_tools = ['terminal_cmd', 'execute_command', 'run_command', 'shell_exec']
    
    for tool_name in known_tools:
        try:
            st.write(f"Testing tool: {tool_name}")
            result = asyncio.run(mcp_client.call_tool(
                'Terminal Server', 
                tool_name, 
                {'command': 'pwd'}
            ))
            if result:
                st.success(f"✅ Found working tool: {tool_name}")
                st.json(result)
                command_server = 'Terminal Server'
                command_tool = tool_name
                break
        except Exception as e:
            st.write(f"  - {tool_name} failed: {e}")
```

#### Connection Recovery

```python
async def reconnect_server(self, server_name: str) -> bool:
    """Attempt to reconnect to a server"""
    logger.info(f"Attempting to reconnect to {server_name}")
    
    # Disconnect if connected
    if server_name in self.connections:
        await self.disconnect_server(server_name)
    
    # Wait briefly before reconnecting
    await asyncio.sleep(1.0)
    
    # Attempt reconnection
    return await self.connect_server(server_name)
```

## 🔄 Complete Discovery Flow Example

### Your Terminal Server Discovery Trace:

#### 1. Configuration Load
```
[INFO] Looking for MCP config at: /Users/widojansen/Projects/Agents/RAG/BraceFaceRag/mcp_config.json
[INFO] Loaded config data: {"servers": {"Terminal Server": {...}}}
[INFO] Server Terminal Server is enabled
[INFO] Loaded 1 MCP server configurations
```

#### 2. Connection Initiation
```
[INFO] Total servers configured: 1
[INFO] Enabled servers: ['Terminal Server']
[INFO] Disabled servers: []
[INFO] Attempting to connect to enabled server: Terminal Server
```

#### 3. Process Launch
```
[INFO] Starting MCP server: Terminal Server
[INFO] Command: /Users/widojansen/.asdf/shims/uv run /Users/widojansen/Projects/Agents/MCP/shellserver/server.py
[INFO] MCP server Terminal Server process started
```

#### 4. Protocol Initialization
```
[INFO] Sending initialize request to Terminal Server
Request: {
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {"roots": {"listChanged": true}, "sampling": {}},
    "clientInfo": {"name": "ragtools-mcp-client", "version": "1.0.0"}
  }
}
Response: {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {...}}}
[INFO] Successfully initialized Terminal Server
```

#### 5. Capability Discovery
```
[INFO] Requesting tools from Terminal Server
Request: {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
Response: {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}  ← Issue here!
[WARNING] Server Terminal Server reported 0 tools

[INFO] Requesting resources from Terminal Server  
Request: {"jsonrpc": "2.0", "id": 3, "method": "resources/list"}
Response: {"jsonrpc": "2.0", "id": 3, "result": {"resources": [...]}}
[INFO] Server Terminal Server reported 1 resources
```

#### 6. Connection Finalization
```
[INFO] Successfully connected to MCP server: Terminal Server
[INFO] Connected to 1/1 MCP servers
```

#### 7. Fallback Mechanism Activation
```
[INFO] Terminal Server found but no tools listed - adding fallback tools
[INFO] Added fallback tool: terminal_cmd
```

#### 8. Manual Tool Testing
```
[INFO] Testing tool: terminal_cmd
[INFO] Call: Terminal Server.terminal_cmd({"command": "pwd"})
[INFO] Result: {"stdout": "/Users/widojansen/Projects/Agents/RAG/BraceFaceRag", "stderr": "", "return_code": 0}
[SUCCESS] ✅ Found working tool: terminal_cmd
```

## 🎯 Key Discovery Insights

### What Works
- **Configuration Loading**: ✅ Successfully finds and parses `mcp_config.json`
- **Process Launch**: ✅ Terminal Server subprocess starts correctly
- **Protocol Handshake**: ✅ MCP initialization succeeds
- **Manual Tool Testing**: ✅ `terminal_cmd` tool works when called directly

### What Needs Investigation
- **Tools Discovery**: ⚠️ `tools/list` returns empty array despite server having tools
- **Resource Discovery**: ✅ `resources/list` works correctly

### Possible Root Causes
1. **Server Implementation**: Your Terminal Server may not properly implement `tools/list`
2. **Timing Issues**: Server may not be ready when `tools/list` is called
3. **FastMCP Framework**: The FastMCP framework might have a bug in tool registration
4. **Protocol Version**: Version mismatch between client and server

### Recovery Strategies
1. **Fallback Tool Injection**: Manually add known tools when discovery fails
2. **Direct Tool Testing**: Test tools by name even when not discovered
3. **Delayed Discovery**: Retry discovery after a delay
4. **Manual Configuration**: Hardcode tool definitions for known servers

This comprehensive discovery process ensures that even when the MCP protocol discovery fails, the system can still function through intelligent fallbacks and manual testing! 🚀