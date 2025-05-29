"""
RAG Tools - A toolkit for creating Retrieval-Augmented Generation applications
Enhanced with Model Context Protocol (MCP) support
"""

from ragtools.streamlit_ui import RagStreamlitUI, launch_streamlit_ui

# Try to import MCP functionality
try:
    from ragtools.mcp_client import MCPClient, MCPServerConfig, MCPRagIntegration, AsyncMCPClient
    MCP_AVAILABLE = True
except ImportError:
    # MCP functionality not available
    MCP_AVAILABLE = False
    MCPClient = None
    MCPServerConfig = None
    MCPRagIntegration = None
    AsyncMCPClient = None

__version__ = "0.2.4.0"
__all__ = [
    "RagStreamlitUI",
    "launch_streamlit_ui",
    "MCPClient",
    "MCPServerConfig",
    "MCPRagIntegration",
    "AsyncMCPClient",
    "MCP_AVAILABLE"
]