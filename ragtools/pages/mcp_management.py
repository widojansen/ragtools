"""
MCP Management Page for RAG Tools
Streamlit interface for managing MCP servers
"""

import streamlit as st
import asyncio
import json
from typing import Dict, Any
import logging

# Try to import MCP client
try:
    from ..mcp_client import MCPClient, MCPServerConfig, AsyncMCPClient

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    st.error("MCP client not available. Please install MCP dependencies.")

logger = logging.getLogger(__name__)


def run():
    """Main function for the MCP Management page"""

    if not MCP_AVAILABLE:
        st.error("❌ MCP functionality is not available")
        st.info("Please install MCP dependencies by running: `pip install mcp httpx anyio`")
        return

    st.title("🔌 MCP Server Management")
    st.markdown("Manage Model Context Protocol servers for enhanced RAG capabilities")

    # Initialize MCP client in session state
    if 'mcp_client' not in st.session_state:
        st.session_state.mcp_client = MCPClient()

    mcp_client = st.session_state.mcp_client

    # Tabs for different MCP management functions
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Server Status", "⚙️ Configuration", "🛠️ Tools & Resources", "➕ Add Server"])

    with tab1:
        _display_server_status(mcp_client)

    with tab2:
        _display_server_configuration(mcp_client)

    with tab3:
        _display_tools_and_resources(mcp_client)

    with tab4:
        _display_add_server(mcp_client)


def _display_server_status(mcp_client: MCPClient):
    """Display server connection status"""
    st.header("Server Connection Status")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

        if st.button("🔗 Connect All", use_container_width=True):
            with st.spinner("Connecting to servers..."):
                try:
                    asyncio.run(mcp_client.connect_all_servers())
                    st.success("Connection attempt completed")
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    with col1:
        server_status = mcp_client.get_server_status()

        if not server_status:
            st.info("No MCP servers configured")
            return

        for server_name, status in server_status.items():
            server_config = mcp_client.servers.get(server_name)
            if not server_config:
                continue

            # Create server status card
            with st.container():
                col_status, col_info, col_actions = st.columns([1, 2, 1])

                with col_status:
                    if status == "connected":
                        st.success(f"✅ {server_name}")
                    else:
                        st.error(f"❌ {server_name}")

                with col_info:
                    st.write(f"**Description:** {server_config.description or 'No description'}")
                    st.write(f"**Enabled:** {'Yes' if server_config.enabled else 'No'}")
                    st.write(f"**Command:** `{server_config.command} {' '.join(server_config.args)}`")

                with col_actions:
                    if status == "connected":
                        if st.button(f"🔌 Disconnect", key=f"disconnect_{server_name}"):
                            asyncio.run(mcp_client.disconnect_server(server_name))
                            st.rerun()
                    else:
                        if st.button(f"🔗 Connect", key=f"connect_{server_name}"):
                            with st.spinner(f"Connecting to {server_name}..."):
                                success = asyncio.run(mcp_client.connect_server(server_name))
                                if success:
                                    st.success(f"Connected to {server_name}")
                                else:
                                    st.error(f"Failed to connect to {server_name}")
                            st.rerun()

                st.divider()


def _display_server_configuration(mcp_client: MCPClient):
    """Display and manage server configurations"""
    st.header("Server Configuration")

    if not mcp_client.servers:
        st.info("No servers configured. Add a server in the 'Add Server' tab.")
        return

    # Select server to configure
    server_names = list(mcp_client.servers.keys())
    selected_server = st.selectbox("Select server to configure:", server_names)

    if selected_server:
        server_config = mcp_client.servers[selected_server]

        with st.form(f"config_form_{selected_server}"):
            st.subheader(f"Configure {selected_server}")

            # Editable fields
            description = st.text_input("Description", value=server_config.description)
            enabled = st.checkbox("Enabled", value=server_config.enabled)
            command = st.text_input("Command", value=server_config.command)
            args_text = st.text_area(
                "Arguments (one per line)",
                value="\n".join(server_config.args) if server_config.args else ""
            )

            # Environment variables
            st.subheader("Environment Variables")
            env_vars = server_config.env or {}

            # Display existing env vars
            for i, (key, value) in enumerate(env_vars.items()):
                col1, col2 = st.columns(2)
                with col1:
                    new_key = st.text_input(f"Env Key {i + 1}", value=key, key=f"env_key_{selected_server}_{i}")
                with col2:
                    new_value = st.text_input(f"Env Value {i + 1}", value=value, key=f"env_val_{selected_server}_{i}",
                                              type="password")

                # Update env_vars dict
                if new_key and new_key != key:
                    env_vars[new_key] = env_vars.pop(key, new_value)
                elif new_key:
                    env_vars[new_key] = new_value

            # Add new env var
            col1, col2 = st.columns(2)
            with col1:
                new_env_key = st.text_input("New Env Key", key=f"new_env_key_{selected_server}")
            with col2:
                new_env_value = st.text_input("New Env Value", key=f"new_env_value_{selected_server}", type="password")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.form_submit_button("💾 Save Configuration", use_container_width=True):
                    # Update server configuration
                    args_list = [arg.strip() for arg in args_text.split('\n') if arg.strip()]

                    if new_env_key and new_env_value:
                        env_vars[new_env_key] = new_env_value

                    updated_config = MCPServerConfig(
                        name=selected_server,
                        command=command,
                        args=args_list,
                        env=env_vars,
                        description=description,
                        enabled=enabled
                    )

                    mcp_client.servers[selected_server] = updated_config
                    mcp_client._save_config()
                    st.success("Configuration saved!")
                    st.rerun()

            with col2:
                if st.form_submit_button("🗑️ Delete Server", use_container_width=True):
                    # Disconnect if connected
                    if selected_server in mcp_client.sessions:
                        asyncio.run(mcp_client.disconnect_server(selected_server))

                    mcp_client.remove_server_config(selected_server)
                    st.success(f"Server {selected_server} deleted!")
                    st.rerun()

            with col3:
                if st.form_submit_button("🧪 Test Connection", use_container_width=True):
                    with st.spinner("Testing connection..."):
                        success = asyncio.run(mcp_client.connect_server(selected_server))
                        if success:
                            st.success("Connection test successful!")
                            # Disconnect after test
                            asyncio.run(mcp_client.disconnect_server(selected_server))
                        else:
                            st.error("Connection test failed!")


def _display_tools_and_resources(mcp_client: MCPClient):
    """Display available tools and resources from connected servers"""
    st.header("Available Tools & Resources")

    connected_servers = [name for name, session in mcp_client.sessions.items()]

    if not connected_servers:
        st.info("No servers connected. Connect to servers to see available tools and resources.")
        return

    # Tools section
    st.subheader("🛠️ Available Tools")
    tools = mcp_client.get_available_tools()

    if tools:
        for server_name, server_tools in tools.items():
            if server_tools:
                st.write(f"**{server_name}:**")
                for tool in server_tools:
                    with st.expander(f"🔧 {tool.get('name', 'Unknown Tool')}"):
                        st.write(f"**Description:** {tool.get('description', 'No description')}")

                        if 'inputSchema' in tool:
                            st.write("**Input Schema:**")
                            st.json(tool['inputSchema'])

                        # Tool testing interface
                        if st.button(f"Test {tool.get('name')}", key=f"test_tool_{server_name}_{tool.get('name')}"):
                            st.info("Tool testing interface would go here")
    else:
        st.info("No tools available from connected servers")

    # Resources section
    st.subheader("📚 Available Resources")
    resources = mcp_client.get_available_resources()

    if resources:
        for server_name, server_resources in resources.items():
            if server_resources:
                st.write(f"**{server_name}:**")
                for resource in server_resources:
                    with st.expander(f"📄 {resource.get('name', 'Unknown Resource')}"):
                        st.write(f"**URI:** {resource.get('uri', 'Unknown')}")
                        st.write(f"**Description:** {resource.get('description', 'No description')}")

                        if st.button(f"Read Resource", key=f"read_resource_{server_name}_{resource.get('uri')}"):
                            with st.spinner("Reading resource..."):
                                content = asyncio.run(mcp_client.read_resource(server_name, resource.get('uri')))
                                if content:
                                    st.text_area("Resource Content:", content, height=200)
                                else:
                                    st.error("Failed to read resource")
    else:
        st.info("No resources available from connected servers")


def _display_add_server(mcp_client: MCPClient):
    """Interface for adding new MCP servers"""
    st.header("Add New MCP Server")

    # Server templates
    templates = {
        "Custom": {
            "command": "",
            "args": [],
            "description": "Custom MCP server",
            "env": {}
        },
        "Filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"],
            "description": "Access filesystem resources",
            "env": {}
        },
        "Brave Search": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "description": "Web search capabilities",
            "env": {"BRAVE_API_KEY": "your-brave-api-key-here"}
        },
        "PostgreSQL": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres"],
            "description": "PostgreSQL database access",
            "env": {"POSTGRES_CONNECTION_STRING": "postgresql://user:password@localhost:5432/dbname"}
        },
        "GitHub": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "description": "GitHub repository access",
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-token-here"}
        }
    }

    # Template selection
    template_name = st.selectbox("Choose a template:", list(templates.keys()))
    template = templates[template_name]

    with st.form("add_server_form"):
        st.subheader("Server Configuration")

        server_name = st.text_input("Server Name", value=template_name.lower().replace(" ", "-"))
        description = st.text_input("Description", value=template["description"])
        command = st.text_input("Command", value=template["command"])
        args_text = st.text_area(
            "Arguments (one per line)",
            value="\n".join(template["args"])
        )
        enabled = st.checkbox("Enable server", value=True)

        # Environment variables
        st.subheader("Environment Variables")
        env_vars = {}

        # Show template env vars
        for i, (key, value) in enumerate(template["env"].items()):
            col1, col2 = st.columns(2)
            with col1:
                env_key = st.text_input(f"Env Key {i + 1}", value=key, key=f"template_env_key_{i}")
            with col2:
                env_value = st.text_input(f"Env Value {i + 1}", value=value, key=f"template_env_value_{i}",
                                          type="password")

            if env_key:
                env_vars[env_key] = env_value

        # Add additional env vars
        for i in range(3):  # Allow up to 3 additional env vars
            col1, col2 = st.columns(2)
            with col1:
                extra_key = st.text_input(f"Additional Env Key {i + 1}", key=f"extra_env_key_{i}")
            with col2:
                extra_value = st.text_input(f"Additional Env Value {i + 1}", key=f"extra_env_value_{i}",
                                            type="password")

            if extra_key and extra_value:
                env_vars[extra_key] = extra_value

        if st.form_submit_button("➕ Add Server", use_container_width=True):
            if not server_name:
                st.error("Server name is required")
            elif server_name in mcp_client.servers:
                st.error(f"Server '{server_name}' already exists")
            else:
                args_list = [arg.strip() for arg in args_text.split('\n') if arg.strip()]

                new_config = MCPServerConfig(
                    name=server_name,
                    command=command,
                    args=args_list,
                    env=env_vars,
                    description=description,
                    enabled=enabled
                )

                mcp_client.add_server_config(new_config)
                st.success(f"Server '{server_name}' added successfully!")

                # Test connection if enabled
                if enabled:
                    with st.spinner("Testing connection..."):
                        success = asyncio.run(mcp_client.connect_server(server_name))
                        if success:
                            st.success("Connection test successful!")
                        else:
                            st.warning("Connection test failed. Please check your configuration.")

                st.rerun()