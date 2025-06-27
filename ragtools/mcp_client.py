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
                stdout_output = self.process.stdout.read() if self.process.stdout else "No stdout"
                logger.error(f"MCP server {self.config.name} died immediately. Command: {self.config.command} {' '.join(self.config.args)}")
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
        Abstract MCP enhancement that works with any MCP server and tools
        """
        enhanced_context = context or {}

        # Get available tools and resources from all connected servers
        available_tools = self.mcp_client.get_available_tools()
        available_resources = self.mcp_client.get_available_resources()

        logger.info(f"Available MCP servers and tools: {list(available_tools.keys())}")

        # Special handling for Terminal Server when tools aren't listed properly
        if 'Terminal Server' in available_tools and not available_tools['Terminal Server']:
            logger.info("Terminal Server found but no tools listed - adding fallback tools")
            available_tools['Terminal Server'] = [
                {'name': 'terminal_cmd', 'description': 'Execute terminal commands', 'inputSchema': {'properties': {'command': {'type': 'string'}}}}
            ]

        # Analyze query to determine intent and relevant tools
        query_analysis = await self._analyze_query_intent(query, available_tools)
        
        # Execute relevant tools based on analysis
        for server_name, tools_to_call in query_analysis.items():
            for tool_call in tools_to_call:
                try:
                    tool_name = tool_call['tool']
                    args = tool_call['args']
                    intent = tool_call['intent']
                    
                    logger.info(f"Calling {server_name}.{tool_name} with intent '{intent}'")
                    
                    result = await self.mcp_client.call_tool(server_name, tool_name, args)
                    
                    if result:
                        # Store result with generic key based on intent
                        result_key = f"mcp_{intent}_results"
                        enhanced_context[result_key] = {
                            'server': server_name,
                            'tool': tool_name,
                            'intent': intent,
                            'data': result,
                            'args': args
                        }
                        logger.info(f"Successfully executed {server_name}.{tool_name} for intent '{intent}'")
                    
                except Exception as e:
                    logger.error(f"Failed to call {server_name}.{tool_name}: {e}")

        return enhanced_context

    async def _analyze_query_intent(self, query: str, available_tools: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Analyze query to determine which MCP tools to call based on available capabilities
        """
        query_lower = query.lower()
        tools_to_call = {}

        for server_name, tools in available_tools.items():
            server_tools = []
            
            for tool in tools:
                tool_name = tool.get('name', '')
                tool_desc = tool.get('description', '').lower()
                
                # Determine intent and tool relevance based on keywords and descriptions
                intent, args = await self._match_tool_to_query(query_lower, tool_name, tool_desc, tool.get('inputSchema', {}), server_name)
                
                if intent:
                    server_tools.append({
                        'tool': tool_name,
                        'intent': intent,
                        'args': args
                    })
            
            if server_tools:
                tools_to_call[server_name] = server_tools

        return tools_to_call

    async def _match_tool_to_query(self, query: str, tool_name: str, tool_desc: str, schema: Dict, server_name: str = '') -> tuple[Optional[str], Dict]:
        """
        Match a query to a specific tool and generate appropriate arguments
        """
        # Terminal/shell command intent - more flexible matching
        if any(kw in query for kw in ['run', 'execute', 'command', 'terminal', 'shell', 'bash', 'zsh', 'cmd']):
            # Check tool name and description for command-related keywords
            if (any(kw in tool_name.lower() for kw in ['command', 'execute', 'run', 'shell', 'terminal', 'bash', 'cmd']) or
                any(kw in tool_desc for kw in ['command', 'execute', 'run', 'shell', 'terminal', 'bash', 'cmd'])):
                return 'terminal_command', await self._generate_command_args(query, schema)
            
            # If server name suggests it's a terminal/command server, use any tool from that server
            if any(kw in server_name.lower() for kw in ['terminal', 'command', 'shell', 'cmd']):
                return 'terminal_command', await self._generate_command_args(query, schema)
            
            # Special case: hardcode support for specific terminal tools
            if tool_name in ['terminal_cmd', 'execute_command', 'run_command', 'shell_exec']:
                return 'terminal_command', await self._generate_command_args(query, schema)

        # Common system commands - automatically detect them
        common_commands = ['ls', 'pwd', 'whoami', 'date', 'ps', 'top', 'df', 'du', 'free', 'uname', 'which', 'find', 'grep', 'cat', 'head', 'tail', 'mkdir', 'rmdir', 'cp', 'mv', 'rm']
        query_words = query.lower().split()
        if any(cmd in query_words for cmd in common_commands):
            # Check tool or server for command capability
            if (any(kw in tool_name.lower() for kw in ['command', 'execute', 'run', 'shell', 'terminal']) or
                any(kw in server_name.lower() for kw in ['terminal', 'command', 'shell', 'cmd'])):
                return 'terminal_command', await self._generate_command_args(query, schema)

        # Directory/file listing intent
        if any(kw in query for kw in ['show', 'list', 'content', 'folder', 'directory', 'files']):
            if any(kw in tool_name.lower() for kw in ['list', 'directory', 'files', 'tree']):
                return 'directory_listing', await self._generate_file_args(query, schema)
            if any(kw in tool_desc for kw in ['list', 'directory', 'files', 'folder']):
                return 'directory_listing', await self._generate_file_args(query, schema)
            # If no file tools available, try with shell command
            if (any(kw in tool_name.lower() for kw in ['command', 'execute', 'run', 'shell', 'terminal']) or
                any(kw in server_name.lower() for kw in ['terminal', 'command', 'shell', 'cmd'])):
                return 'terminal_command', await self._generate_command_args(f"ls {self._extract_path_from_query(query)}", schema)

        # File operations intent
        if any(kw in query for kw in ['read', 'open', 'view', 'show file']):
            if any(kw in tool_name.lower() for kw in ['read', 'get', 'file']):
                return 'file_operation', await self._generate_file_args(query, schema)
            # If no file tools available, try with shell command
            if (any(kw in tool_name.lower() for kw in ['command', 'execute', 'run', 'shell', 'terminal']) or
                any(kw in server_name.lower() for kw in ['terminal', 'command', 'shell', 'cmd'])):
                return 'terminal_command', await self._generate_command_args(f"cat {self._extract_file_from_query(query)}", schema)

        # Search intent
        if any(kw in query for kw in ['search', 'find', 'lookup']):
            if any(kw in tool_name.lower() for kw in ['search', 'find', 'query']):
                return 'search', await self._generate_search_args(query, schema)
            if any(kw in tool_desc for kw in ['search', 'find', 'query']):
                return 'search', await self._generate_search_args(query, schema)

        # Web search intent
        if any(kw in tool_name.lower() for kw in ['web', 'brave', 'search']) and 'local' not in query:
            return 'web_search', await self._generate_search_args(query, schema)

        # Database/data intent
        if any(kw in query for kw in ['database', 'query', 'sql', 'data']):
            if any(kw in tool_name.lower() for kw in ['db', 'database', 'sql', 'query']):
                return 'database_query', await self._generate_db_args(query, schema)

        return None, {}

    def _extract_path_from_query(self, query: str) -> str:
        """Extract path from query for directory operations"""
        if 'desktop' in query.lower():
            return '/Users/widojansen/Desktop'
        elif 'home' in query.lower():
            return '/Users/widojansen'
        else:
            return '.'

    def _extract_file_from_query(self, query: str) -> str:
        """Extract filename from query for file operations"""
        # Simple extraction - could be made more sophisticated
        words = query.split()
        for word in words:
            if '.' in word and not word.startswith('.'):  # Likely a filename
                return word
        return '.'

    async def _generate_file_args(self, query: str, schema: Dict) -> Dict:
        """Generate file operation arguments based on query and tool schema"""
        args = {}
        properties = schema.get('properties', {})
        
        # Handle path parameter
        if 'path' in properties:
            if 'desktop' in query.lower():
                args['path'] = '/Users/widojansen/Desktop'  # Could be made more generic
            elif 'home' in query.lower():
                args['path'] = '/Users/widojansen'
            else:
                args['path'] = '.'  # Default to current directory
        
        # Handle directory_path parameter (alternative naming)
        if 'directory_path' in properties:
            args['directory_path'] = args.get('path', '.')
            
        return args

    async def _generate_search_args(self, query: str, schema: Dict) -> Dict:
        """Generate search arguments based on query and tool schema"""
        args = {}
        properties = schema.get('properties', {})
        
        if 'query' in properties:
            args['query'] = query
        if 'q' in properties:
            args['q'] = query
        if 'pattern' in properties:
            args['pattern'] = '*'  # Default pattern
        if 'count' in properties:
            args['count'] = 3
        if 'path' in properties:
            args['path'] = '.'
            
        return args

    async def _generate_db_args(self, query: str, schema: Dict) -> Dict:
        """Generate database arguments based on query and tool schema"""
        args = {}
        properties = schema.get('properties', {})
        
        if 'query' in properties:
            args['query'] = query
        if 'sql' in properties:
            args['sql'] = query
            
        return args

    async def _generate_command_args(self, query: str, schema: Dict) -> Dict:
        """Generate terminal command arguments based on query and tool schema"""
        args = {}
        properties = schema.get('properties', {})
        
        # Extract command from query
        command = self._extract_command_from_query(query)
        
        # Handle different parameter names that command tools might use
        if 'command' in properties:
            args['command'] = command
        elif 'cmd' in properties:
            args['cmd'] = command
        elif 'shell_command' in properties:
            args['shell_command'] = command
        elif 'script' in properties:
            args['script'] = command
        elif 'code' in properties:
            args['code'] = command
        
        # Handle additional common parameters
        if 'shell' in properties:
            args['shell'] = '/bin/bash'  # Default shell
        if 'timeout' in properties:
            args['timeout'] = 30  # Default timeout
        if 'working_directory' in properties or 'cwd' in properties:
            path = self._extract_path_from_query(query)
            if 'working_directory' in properties:
                args['working_directory'] = path
            else:
                args['cwd'] = path
                
        return args

    def _extract_command_from_query(self, query: str) -> str:
        """Extract shell command from natural language query"""
        query_lower = query.lower()
        
        # If query already contains shell command keywords, extract the command
        if any(kw in query_lower for kw in ['run', 'execute', 'command']):
            # Look for quoted commands or commands after keywords
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() in ['run', 'execute', 'command'] and i + 1 < len(words):
                    # Return everything after the keyword
                    return ' '.join(words[i + 1:])
        
        # Common command patterns
        if 'current directory' in query_lower or 'working directory' in query_lower:
            return 'pwd'
        elif 'who am i' in query_lower or 'current user' in query_lower:
            return 'whoami'
        elif 'system info' in query_lower or 'system information' in query_lower:
            return 'uname -a'
        elif 'disk space' in query_lower or 'disk usage' in query_lower:
            return 'df -h'
        elif 'memory usage' in query_lower or 'ram usage' in query_lower:
            return 'free -h'
        elif 'running processes' in query_lower or 'process list' in query_lower:
            return 'ps aux'
        elif 'environment variables' in query_lower or 'env vars' in query_lower:
            return 'env'
        elif 'network interfaces' in query_lower:
            return 'ifconfig'
        
        # If the query contains common commands, return it as-is
        common_commands = ['ls', 'pwd', 'whoami', 'date', 'ps', 'top', 'df', 'du', 'free', 'uname', 'which', 'find', 'grep', 'cat', 'head', 'tail']
        query_words = query.split()
        for cmd in common_commands:
            if cmd in query_words:
                # Find the command and return it with its arguments
                cmd_index = query_words.index(cmd)
                return ' '.join(query_words[cmd_index:])
        
        # Default: return the query as-is (user might have typed a direct command)
        return query

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