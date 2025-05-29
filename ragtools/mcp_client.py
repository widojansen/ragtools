"""
MCP Client for RAG Tools
Provides integration with Model Context Protocol servers
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    description: str = ""
    enabled: bool = True

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}


class MCPClient:
    """
    MCP Client for RAG Tools
    Manages connections to MCP servers and provides access to their tools and resources
    """

    def __init__(self, config_path: Optional[str] = None):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.server_capabilities: Dict[str, Dict] = {}
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

        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env
            )

            # Connect to server
            stdio_transport = await stdio_client(server_params)
            session = ClientSession(stdio_transport[0], stdio_transport[1])

            # Initialize session
            await session.initialize()

            # Get server capabilities
            capabilities = await session.list_tools()

            self.sessions[server_name] = session
            self.server_capabilities[server_name] = {
                'tools': capabilities.tools if capabilities else [],
                'resources': []
            }

            # Try to get resources
            try:
                resources = await session.list_resources()
                self.server_capabilities[server_name]['resources'] = resources.resources if resources else []
            except Exception:
                # Some servers might not support resources
                pass

            logger.info(f"Connected to MCP server: {server_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to server {server_name}: {e}")
            return False

    async def disconnect_server(self, server_name: str):
        """Disconnect from an MCP server"""
        if server_name in self.sessions:
            try:
                await self.sessions[server_name].close()
                del self.sessions[server_name]
                if server_name in self.server_capabilities:
                    del self.server_capabilities[server_name]
                logger.info(f"Disconnected from MCP server: {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {server_name}: {e}")

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
        for server_name in list(self.sessions.keys()):
            tasks.append(self.disconnect_server(server_name))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_available_tools(self) -> Dict[str, List[Dict]]:
        """Get all available tools from connected servers"""
        return {
            server_name: caps.get('tools', [])
            for server_name, caps in self.server_capabilities.items()
        }

    def get_available_resources(self) -> Dict[str, List[Dict]]:
        """Get all available resources from connected servers"""
        return {
            server_name: caps.get('resources', [])
            for server_name, caps in self.server_capabilities.items()
        }

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict]:
        """Call a tool on a specific MCP server"""
        if server_name not in self.sessions:
            logger.error(f"Server {server_name} not connected")
            return None

        try:
            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, arguments)
            return result.content if result else None
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {server_name}: {e}")
            return None

    async def read_resource(self, server_name: str, uri: str) -> Optional[str]:
        """Read a resource from a specific MCP server"""
        if server_name not in self.sessions:
            logger.error(f"Server {server_name} not connected")
            return None

        try:
            session = self.sessions[server_name]
            result = await session.read_resource(uri)
            return result.contents[0].text if result and result.contents else None
        except Exception as e:
            logger.error(f"Error reading resource {uri} from {server_name}: {e}")
            return None

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
            name: "connected" if name in self.sessions else "disconnected"
            for name in self.servers.keys()
        }


class MCPRagIntegration:
    """
    Integration layer between MCP client and RAG functionality
    """

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client

    async def enhance_query_with_mcp(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
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

        # Example: Access filesystem resources if available
        if 'filesystem' in available_resources:
            # Could search for relevant files based on query
            pass

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
                if server_name in self.mcp_client.sessions:
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
        await self.client.disconnect_all_servers()