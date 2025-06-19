"""
MCP Client for RAG Tools
Provides integration with Model Context Protocol servers using direct JSON-RPC communication
"""

import asyncio
import json
import logging
import subprocess
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    description: str = ""
    enabled: bool = True

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}


class MCPConnection:
    """Direct JSON-RPC connection to an MCP server"""
    
    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self.process: Optional[subprocess.Popen] = None
        self.connected = False
        self.request_id = 0
        self.capabilities: Dict[str, Any] = {}
        
    def _get_next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id
    
    async def connect(self) -> bool:
        """Connect to the MCP server"""
        try:
            logger.info(f"Starting MCP server: {self.config.name}")
            
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
                logger.error(f"MCP server {self.config.name} died immediately: {stderr_output}")
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
    
    async def _initialize(self) -> bool:
        """Send initialization request to server"""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "ragtools-mcp-client",
                    "version": "1.0.0"
                }
            }
        }
        
        response = await self._send_request(init_request)
        return response is not None and "result" in response
    
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
        else:
            self.capabilities["tools"] = []
        
        # Get resources (optional - not all servers support this)
        resources_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "resources/list"
        }
        
        resources_response = await self._send_request(resources_request, timeout=5.0)
        if resources_response and "result" in resources_response:
            self.capabilities["resources"] = resources_response["result"].get("resources", [])
        else:
            self.capabilities["resources"] = []
    
    async def _send_request(self, request: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC request and get response"""
        if not self.process or not self.process.stdin or not self.process.stdout:
            logger.error(f"Process not available for {self.config.name}")
            return None
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Read response with timeout
            response_line = await asyncio.wait_for(
                self._read_line_async(),
                timeout=timeout
            )
            
            if response_line:
                return json.loads(response_line)
            else:
                logger.warning(f"No response received from {self.config.name}")
                return None
                
        except asyncio.TimeoutError:
            logger.error(f"Request timed out for {self.config.name}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {self.config.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error sending request to {self.config.name}: {e}")
            return None
    
    async def _read_line_async(self) -> str:
        """Read a line from process stdout asynchronously"""
        if not self.process or not self.process.stdout:
            return ""
        
        loop = asyncio.get_event_loop()
        
        def read_line():
            return self.process.stdout.readline()
        
        return await loop.run_in_executor(None, read_line)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool on the server"""
        if not self.connected:
            logger.error(f"Not connected to {self.config.name}")
            return None
        
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self._send_request(request, timeout=30.0)
        if response and "result" in response:
            return response["result"]
        else:
            logger.error(f"Tool call failed for {tool_name} on {self.config.name}: {response}")
            return None
    
    async def read_resource(self, uri: str) -> Optional[str]:
        """Read a resource from the server"""
        if not self.connected:
            logger.error(f"Not connected to {self.config.name}")
            return None
        
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "resources/read",
            "params": {
                "uri": uri
            }
        }
        
        response = await self._send_request(request)
        if response and "result" in response:
            contents = response["result"].get("contents", [])
            if contents:
                content = contents[0]
                if content.get("type") == "text":
                    return content.get("text")
                elif "data" in content:
                    return str(content["data"])
        
        return None
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools"""
        return self.capabilities.get("tools", [])
    
    def get_resources(self) -> List[Dict[str, Any]]:
        """Get available resources"""
        return self.capabilities.get("resources", [])
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.process:
            logger.info(f"Disconnecting from MCP server: {self.config.name}")
            self.process.terminate()
            
            try:
                await asyncio.wait_for(
                    self._wait_for_process(),
                    timeout=5.0
                )
                logger.info(f"Server {self.config.name} disconnected gracefully")
            except asyncio.TimeoutError:
                logger.warning(f"Server {self.config.name} didn't terminate gracefully, killing...")
                self.process.kill()
                await self._wait_for_process()
                logger.info(f"Server {self.config.name} killed")
            
            self.process = None
            self.connected = False
    
    async def _wait_for_process(self):
        """Wait for process to terminate"""
        if self.process:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.process.wait)


class MCPClient:
    """
    MCP Client for RAG Tools
    Manages connections to MCP servers using direct JSON-RPC communication
    """

    def __init__(self, config_path: Optional[str] = None):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.connections: Dict[str, MCPConnection] = {}
        self.config_path = config_path or "mcp_config.json"
        self._load_config()

    def _load_config(self):
        """Load MCP server configurations from file"""
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    for server_name, server_config in config_data.get('servers', {}).items():
                        self.servers[server_name] = MCPServerConfig(
                            name=server_name,
                            **server_config
                        )
                logger.info(f"Loaded {len(self.servers)} MCP server configurations")
            except Exception as e:
                logger.error(f"Failed to load MCP config: {e}")
        else:
            # Create default config
            self._create_default_config()

    def _create_default_config(self):
        """Create a default configuration file"""
        default_config = {
            "servers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"],
                    "description": "Access filesystem resources",
                    "enabled": False
                },
                "brave-search": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {
                        "BRAVE_API_KEY": "your-brave-api-key"
                    },
                    "description": "Web search capabilities",
                    "enabled": False
                },
                "postgres": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-postgres"],
                    "env": {
                        "POSTGRES_CONNECTION_STRING": "postgresql://user:password@localhost:5432/dbname"
                    },
                    "description": "PostgreSQL database access",
                    "enabled": False
                }
            }
        }

        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default MCP config at {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")

    async def connect_server(self, server_name: str) -> bool:
        """Connect to an MCP server"""
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

    async def disconnect_server(self, server_name: str):
        """Disconnect from an MCP server"""
        if server_name in self.connections:
            connection = self.connections[server_name]
            await connection.disconnect()
            del self.connections[server_name]
            logger.info(f"Disconnected from MCP server: {server_name}")

    async def connect_all_servers(self):
        """Connect to all enabled MCP servers"""
        tasks = []
        for server_name, server_config in self.servers.items():
            if server_config.enabled:
                tasks.append(self.connect_server(server_name))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            connected = sum(1 for r in results if r is True)
            logger.info(f"Connected to {connected}/{len(tasks)} MCP servers")

    async def disconnect_all_servers(self):
        """Disconnect from all connected MCP servers"""
        tasks = []
        for server_name in list(self.connections.keys()):
            tasks.append(self.disconnect_server(server_name))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def cleanup(self):
        """Cleanup all resources and connections"""
        await self.disconnect_all_servers()

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

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict]:
        """Call a tool on a specific MCP server"""
        if server_name not in self.connections:
            logger.error(f"Server {server_name} not connected")
            return None

        connection = self.connections[server_name]
        result = await connection.call_tool(tool_name, arguments)
        
        # Extract content from result if it exists
        if result and "content" in result:
            return result["content"]
        return result

    async def read_resource(self, server_name: str, uri: str) -> Optional[str]:
        """Read a resource from a specific MCP server"""
        if server_name not in self.connections:
            logger.error(f"Server {server_name} not connected")
            return None

        connection = self.connections[server_name]
        return await connection.read_resource(uri)

    def add_server_config(self, config: MCPServerConfig):
        """Add a new server configuration"""
        self.servers[config.name] = config
        self._save_config()

    def remove_server_config(self, server_name: str):
        """Remove a server configuration"""
        if server_name in self.servers:
            del self.servers[server_name]
            self._save_config()

    def _save_config(self):
        """Save current server configurations to file"""
        config_data = {
            "servers": {
                name: {
                    "command": config.command,
                    "args": config.args,
                    "env": config.env,
                    "description": config.description,
                    "enabled": config.enabled
                }
                for name, config in self.servers.items()
            }
        }

        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get_server_status(self) -> Dict[str, str]:
        """Get connection status of all servers"""
        return {
            name: "connected" if name in self.connections and self.connections[name].connected else "disconnected"
            for name in self.servers.keys()
        }


class MCPRagIntegration:
    """
    Integration layer between MCP client and RAG functionality
    """

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client

    async def enhance_query_with_mcp(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance a RAG query using MCP tools and resources
        """
        enhanced_context = context or {}

        # Get available tools and resources
        available_tools = self.mcp_client.get_available_tools()
        available_resources = self.mcp_client.get_available_resources()

        # Example: Use web search if available
        if 'brave-search' in available_tools:
            search_tools = available_tools['brave-search']
            for tool in search_tools:
                if tool.get('name') == 'brave_web_search':
                    try:
                        search_result = await self.mcp_client.call_tool(
                            'brave-search',
                            'brave_web_search',
                            {'query': query, 'count': 3}
                        )
                        if search_result:
                            enhanced_context['web_search_results'] = search_result
                    except Exception as e:
                        logger.error(f"Web search failed: {e}")

        # Example: Search filesystem if available
        if 'filesystem' in available_tools:
            filesystem_tools = available_tools['filesystem']
            for tool in filesystem_tools:
                if tool.get('name') == 'search_files':
                    try:
                        # Search for files related to the query
                        search_result = await self.mcp_client.call_tool(
                            'filesystem',
                            'search_files',
                            {'path': '.', 'pattern': query}
                        )
                        if search_result:
                            enhanced_context['filesystem_search_results'] = search_result
                    except Exception as e:
                        logger.error(f"Filesystem search failed: {e}")

        return enhanced_context

    async def get_context_from_resources(self, resource_uris: List[str]) -> str:
        """
        Retrieve context from MCP resources
        """
        context_parts = []

        for uri in resource_uris:
            # Parse server name from URI (assuming format: server://path)
            if '://' in uri:
                server_name = uri.split('://')[0]
                if server_name in self.mcp_client.connections:
                    content = await self.mcp_client.read_resource(server_name, uri)
                    if content:
                        context_parts.append(f"From {uri}:\n{content}")

        return "\n\n".join(context_parts)


# Async context manager for the MCP client
class AsyncMCPClient:
    """Async context manager wrapper for MCPClient"""

    def __init__(self, config_path: Optional[str] = None):
        self.client = MCPClient(config_path)

    async def __aenter__(self):
        await self.client.connect_all_servers()
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.cleanup()