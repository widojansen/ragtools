"""
Enhanced RAG Chatbot with MCP Integration
"""

import streamlit as st
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Try to import MCP functionality
try:
    from ..mcp_client import MCPClient, MCPRagIntegration

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)


def ask_llm(prompt: str, system_context: str = "", model_context: Dict[str, Any] = None) -> tuple[str, Dict[str, Any]]:
    """
    Enhanced LLM function with MCP integration
    This is a placeholder - replace with your actual LLM implementation
    """
    if model_context is None:
        model_context = {}

    # Use MCP enhancement if available
    if MCP_AVAILABLE and 'mcp_client' in st.session_state:
        try:
            mcp_client = st.session_state.mcp_client
            mcp_integration = MCPRagIntegration(mcp_client)

            # Enhance query with MCP
            enhanced_context = asyncio.run(
                mcp_integration.enhance_query_with_mcp(prompt, model_context)
            )
            model_context.update(enhanced_context)

            # Add web search results to system context if available
            if 'web_search_results' in enhanced_context:
                web_results = enhanced_context['web_search_results']
                system_context += f"\n\nWeb search results for additional context:\n{web_results}"

        except Exception as e:
            logger.error(f"MCP enhancement failed: {e}")

    # Placeholder LLM response
    response = f"This is a placeholder response to: {prompt}\n\nSystem context was: {system_context[:100]}..."

    # Add MCP context info to model context
    if 'web_search_results' in model_context:
        response += "\n\n(Enhanced with web search results via MCP)"

    return response, model_context


def run():
    """Main chatbot interface with MCP enhancements"""

    # Initialize session state with proper defaults
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "model_context" not in st.session_state or st.session_state.model_context is None:
        st.session_state.model_context = {}

    # Initialize MCP client if available
    if MCP_AVAILABLE and 'mcp_client' not in st.session_state:
        try:
            st.session_state.mcp_client = MCPClient()
            # Auto-connect to enabled servers
            asyncio.run(st.session_state.mcp_client.connect_all_servers())
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")

    # Header with MCP status
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🤖 RAG Chatbot")

        if MCP_AVAILABLE and 'mcp_client' in st.session_state:
            mcp_client = st.session_state.mcp_client
            connected_servers = len([s for s in mcp_client.connections.keys() if mcp_client.connections[s].connected])
            if connected_servers > 0:
                st.success(f"🔌 {connected_servers} MCP server(s) connected")
            else:
                st.info("💡 Connect MCP servers for enhanced capabilities")

    with col2:
        if MCP_AVAILABLE:
            if st.button("⚙️ MCP Settings", use_container_width=True):
                st.session_state.current_page = "mcp_management"
                st.rerun()

    # Sidebar with enhanced settings
    with st.sidebar:
        st.header("🎛️ Chat Settings")

        # Temperature setting
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get("temperature", 0.7),
            step=0.1,
            help="Controls randomness in responses"
        )
        st.session_state.temperature = temperature

        # System context
        system_context = st.text_area(
            "🎯 System Context",
            value=st.session_state.get("system_context", ""),
            height=100,
            help="Additional context to guide the assistant's responses"
        )
        st.session_state.system_context = system_context

        # MCP Enhancement Options (if available)
        if MCP_AVAILABLE and 'mcp_client' in st.session_state:
            st.header("🔌 MCP Enhancements")

            mcp_client = st.session_state.mcp_client
            available_tools = mcp_client.get_available_tools()

            # Web search toggle
            if 'brave-search' in available_tools:
                use_web_search = st.checkbox(
                    "🌐 Enable Web Search",
                    value=st.session_state.get("use_web_search", True),
                    help="Automatically search the web for additional context"
                )
                st.session_state.use_web_search = use_web_search

            # Show connected servers
            if mcp_client.connections:
                st.write("**Connected Servers:**")
                for server_name, connection in mcp_client.connections.items():
                    if connection.connected:
                        st.write(f"✅ {server_name}")
            else:
                st.write("No MCP servers connected")

        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.model_context = {}
            st.rerun()

        # Download conversation
        if st.session_state.messages:
            conversation_text = _format_conversation_for_download()
            st.download_button(
                "💾 Download Chat",
                data=conversation_text,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show MCP enhancements if present
            if message.get("mcp_enhanced") and message["role"] == "assistant":
                with st.expander("🔌 MCP Enhancements Used"):
                    if "web_search_results" in message.get("metadata", {}):
                        st.write("🌐 **Web Search Results:**")
                        st.json(message["metadata"]["web_search_results"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Ensure model_context is always a dictionary
                    current_context = st.session_state.model_context
                    if current_context is None:
                        current_context = {}
                    
                    response, updated_context = ask_llm(
                        prompt,
                        st.session_state.get("system_context", ""),
                        current_context.copy()
                    )

                    # Update model context
                    st.session_state.model_context = updated_context

                    # Display response
                    st.markdown(response)

                    # Show MCP enhancements if used
                    mcp_enhanced = any(key.startswith('web_search') for key in updated_context.keys())
                    if mcp_enhanced:
                        with st.expander("🔌 MCP Enhancements Used"):
                            if "web_search_results" in updated_context:
                                st.write("🌐 **Web Search Results:**")
                                st.json(updated_context["web_search_results"])

                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "mcp_enhanced": mcp_enhanced,
                        "metadata": updated_context if mcp_enhanced else {}
                    }
                    st.session_state.messages.append(assistant_message)

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def _format_conversation_for_download() -> str:
    """Format conversation for download"""
    lines = [f"RAG Chatbot Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 50, ""]

    for message in st.session_state.messages:
        role = message["role"].title()
        content = message["content"]
        lines.append(f"{role}: {content}")

        # Add MCP enhancement info
        if message.get("mcp_enhanced"):
            lines.append("  [Enhanced with MCP]")

        lines.append("")

    return "\n".join(lines)


# Utility functions for MCP integration
def get_mcp_status() -> Dict[str, Any]:
    """Get current MCP status for display"""
    if not MCP_AVAILABLE or 'mcp_client' not in st.session_state:
        return {"available": False, "connected_servers": 0, "available_tools": 0}

    mcp_client = st.session_state.mcp_client
    connected_servers = len([c for c in mcp_client.connections.values() if c.connected])
    available_tools = sum(len(tools) for tools in mcp_client.get_available_tools().values())

    return {
        "available": True,
        "connected_servers": connected_servers,
        "available_tools": available_tools,
        "server_names": [name for name, conn in mcp_client.connections.items() if conn.connected]
    }


def refresh_mcp_connections():
    """Refresh MCP server connections"""
    if MCP_AVAILABLE and 'mcp_client' in st.session_state:
        try:
            mcp_client = st.session_state.mcp_client
            asyncio.run(mcp_client.connect_all_servers())
            return True
        except Exception as e:
            logger.error(f"Failed to refresh MCP connections: {e}")
            return False
    return False