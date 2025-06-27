"""
Enhanced RAG Chatbot with MCP Integration
"""

import streamlit as st
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Try to import MCP functionality
try:
    from ..mcp_client import MCPClient, MCPRagIntegration

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)


def ask_llm(prompt: str, system_context: str = "", model_context: Dict[str, Any] = None) -> tuple[str, Dict[str, Any]]:
    """
    Enhanced LLM function with MCP integration and RAG
    This function integrates both RAG retrieval and MCP enhancements
    """
    if model_context is None:
        model_context = {}

    # Initialize variables for building the final context
    enhanced_system_context = system_context
    mcp_enhancements = {}
    rag_context = ""

    # First, get RAG context from vector store
    try:
        from ragtools.import_rag_data import get_retriever
        from langchain.prompts import PromptTemplate
        
        # Get the retriever from the vector store
        knowledge_dir = st.session_state.knowledge_dir
        db_dir = st.session_state.db_dir
        retriever = get_retriever(knowledge_dir, db_dir)
        
        if retriever:
            # Get relevant documents
            docs = retriever.get_relevant_documents(prompt)
            rag_context = "\n\n".join(doc.page_content for doc in docs)
            logger.info(f"Retrieved {len(docs)} documents from RAG")
        else:
            logger.warning("Couldn't initialize RAG retriever")
            
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")

    # Use MCP enhancement if available
    if MCP_AVAILABLE and 'mcp_client' in st.session_state:
        try:
            logger.info(f"Starting MCP enhancement for query: '{prompt}'")
            mcp_client = st.session_state.mcp_client
            mcp_integration = MCPRagIntegration(mcp_client)

            # Enhance a query with MCP
            enhanced_context = asyncio.run(
                mcp_integration.enhance_query_with_mcp(prompt, model_context)
            )
            logger.info(f"MCP enhancement result: {enhanced_context}")
            model_context.update(enhanced_context)
            mcp_enhancements = enhanced_context

            # Process any MCP results generically
            for key, value in enhanced_context.items():
                if key.startswith('mcp_') and key.endswith('_results'):
                    intent = key.replace('mcp_', '').replace('_results', '')
                    server = value.get('server', 'unknown')
                    tool = value.get('tool', 'unknown')
                    data = value.get('data', '')
                    
                    # Format the MCP result for the system context
                    context_addition = f"\n\n{intent.replace('_', ' ').title()} from {server} ({tool}):\n{self._format_mcp_data(data)}"
                    enhanced_system_context += context_addition
                    logger.info(f"Added {intent} results from {server}.{tool} to context")

        except Exception as e:
            logger.error(f"MCP enhancement failed: {e}")
            import traceback
            logger.error(f"Full MCP enhancement error: {traceback.format_exc()}")

    # Combine system context, RAG context, and MCP enhancements
    combined_context = enhanced_system_context
    if rag_context:
        combined_context += f"\n\nKnowledge Base Context:\n{rag_context}"

    # Prepare the messages for the LLM
    messages = []
    
    # Add system message with combined context
    if combined_context.strip():
        messages.append({
            "role": "system", 
            "content": combined_context
        })
    
    # Add user prompt
    messages.append({
        "role": "user",
        "content": prompt
    })

    try:
        # Get LLM configuration from session state or use defaults
        llm_model = st.session_state.get('llm', 'qwen3:8b')
        temperature = st.session_state.get('temperature', 0.7)
        
        # Make the actual LLM call using Ollama
        import requests
        import json
        
        ollama_url = "http://127.0.0.1:11434/api/chat"
        
        payload = {
            "model": llm_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        
        response_data = response.json()
        llm_response = response_data.get('message', {}).get('content', 'Sorry, I could not generate a response.')
        
        # Add information about enhancements used
        enhancement_info = []
        if rag_context:
            enhancement_info.append("knowledge base")
        
        # Generic MCP enhancement detection
        if mcp_enhancements:
            mcp_tools_used = []
            for key, value in mcp_enhancements.items():
                if key.startswith('mcp_') and key.endswith('_results'):
                    intent = key.replace('mcp_', '').replace('_results', '')
                    server = value.get('server', 'MCP')
                    tool = value.get('tool', 'tool')
                    mcp_tools_used.append(f"{intent.replace('_', ' ')} ({server})")
            
            if mcp_tools_used:
                enhancement_info.extend(mcp_tools_used)
        
        if enhancement_info:
            llm_response += f"\n\n*This response was enhanced with: {', '.join(enhancement_info)}*"
        
        return llm_response, model_context
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call Ollama LLM: {e}")
        fallback_response = f"I apologize, but I'm having trouble connecting to the language model. Please ensure Ollama is running and the model '{llm_model}' is available."
        
        # Add context info even in fallback
        context_info = []
        if rag_context:
            context_info.append("knowledge base context")
        if 'web_search_results' in model_context:
            context_info.append("web search context")
        
        if context_info:
            fallback_response += f"\n\n(Note: I was able to gather {', '.join(context_info)}, but couldn't process it due to LLM connection issues.)"
        
        return fallback_response, model_context
        
    except Exception as e:
        logger.error(f"Unexpected error in ask_llm: {e}")
        return f"An unexpected error occurred: {str(e)}", model_context


def _format_mcp_data(data):
    """Format MCP data for display in context"""
    if isinstance(data, list):
        formatted_items = []
        for item in data:
            if isinstance(item, dict) and 'text' in item:
                formatted_items.append(item['text'])
            else:
                formatted_items.append(str(item))
        return '\n'.join(formatted_items)
    elif isinstance(data, dict):
        if 'text' in data:
            return data['text']
        else:
            return str(data)
    else:
        return str(data)


def _get_intent_emoji(intent: str) -> str:
    """Get appropriate emoji for MCP intent"""
    intent_emojis = {
        'directory_listing': '📁',
        'file_operation': '📄',
        'search': '🔍',
        'web_search': '🌐',
        'database_query': '💽',
        'api_call': '🔗',
        'computation': '🧮',
        'data_analysis': '📊',
        'terminal_command': '💻'
    }
    return intent_emojis.get(intent, '🔧')


def run():
    """Main chatbot interface with MCP enhancements"""

    # Initialize the session state with proper defaults
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
            st.error(f"MCP initialization failed: {e}")

    # Header with MCP status
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🤖 RAG Chatbot")

        if MCP_AVAILABLE and 'mcp_client' in st.session_state:
            mcp_client = st.session_state.mcp_client
            connected_servers = len([s for s in mcp_client.connections.keys() if mcp_client.connections[s].connected])
            
            # Debug information
            st.write("**Debug Info:**")
            st.write(f"- Total configured servers: {len(mcp_client.servers)}")
            st.write(f"- Server configs: {list(mcp_client.servers.keys())}")
            st.write(f"- Active connections: {list(mcp_client.connections.keys())}")
            st.write(f"- Connected servers: {[name for name, conn in mcp_client.connections.items() if conn.connected]}")
            
            if connected_servers > 0:
                st.success(f"🔌 {connected_servers} MCP server(s) connected")
                
                # Show available tools
                available_tools = mcp_client.get_available_tools()
                st.write(f"**Available Tools:** {available_tools}")
            else:
                st.warning("💡 No MCP servers connected - check configuration")
                
                # Show server status
                server_status = mcp_client.get_server_status()
                st.write(f"**Server Status:** {server_status}")

    with col2:
        if MCP_AVAILABLE:
            if st.button("⚙️ MCP Settings", use_container_width=True):
                st.session_state.current_page = "mcp_management"
                st.rerun()
            
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("🔄 Refresh MCP", use_container_width=True):
                    if 'mcp_client' in st.session_state:
                        with st.spinner("Reconnecting MCP servers..."):
                            try:
                                # Disconnect all and reconnect
                                asyncio.run(st.session_state.mcp_client.disconnect_all_servers())
                                
                                # Try to connect specifically to filesystem server
                                mcp_client = st.session_state.mcp_client
                                if 'filesystem' in mcp_client.servers:
                                    st.write("**Attempting to connect to filesystem server...**")
                                    result = asyncio.run(mcp_client.connect_server('filesystem'))
                                    st.write(f"Filesystem connection result: {result}")
                                
                                asyncio.run(st.session_state.mcp_client.connect_all_servers())
                                st.success("MCP servers refreshed!")
                            except Exception as e:
                                st.error(f"Failed to refresh MCP: {e}")
                                import traceback
                                st.error(f"Full error: {traceback.format_exc()}")
                    st.rerun()
            
            with col2b:
                if st.button("🔄 Reload Config", use_container_width=True):
                    with st.spinner("Reloading MCP configuration..."):
                        try:
                            # Force reload the entire MCP client
                            from ragtools.mcp_client import MCPClient
                            st.session_state.mcp_client = MCPClient()
                            asyncio.run(st.session_state.mcp_client.connect_all_servers())
                            st.success("MCP configuration reloaded!")
                        except Exception as e:
                            st.error(f"Failed to reload config: {e}")
                    st.rerun()
        
        # Add manual MCP test buttons
        if MCP_AVAILABLE and 'mcp_client' in st.session_state:
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                if st.button("🧪 Test Desktop Listing", use_container_width=True):
                    with st.spinner("Testing desktop directory listing..."):
                        try:
                            mcp_client = st.session_state.mcp_client
                            
                            # Test the directory listing directly
                            result = asyncio.run(mcp_client.call_tool(
                                'filesystem',
                                'list_directory', 
                                {'path': '/Users/widojansen/Desktop'}
                            ))
                            
                            if result:
                                st.success("✅ Desktop listing successful!")
                                st.json(result)
                            else:
                                st.error("❌ No result from desktop listing")
                                
                        except Exception as e:
                            st.error(f"❌ Desktop listing failed: {e}")
                            import traceback
                            st.error(f"Full error: {traceback.format_exc()}")
            
            with col_test2:
                if st.button("💻 Test Terminal Command", use_container_width=True):
                    with st.spinner("Testing terminal command..."):
                        try:
                            mcp_client = st.session_state.mcp_client
                            available_tools = mcp_client.get_available_tools()
                            
                            # Show raw debug info
                            st.write("**Debug Info:**")
                            st.write(f"Connected servers: {list(mcp_client.connections.keys())}")
                            st.write(f"Server status: {[name for name, conn in mcp_client.connections.items() if conn.connected]}")
                            
                            # Show all available tools for debugging
                            st.write("**All Available Tools:**")
                            st.write(f"Raw tools data: {available_tools}")
                            
                            for server_name, tools in available_tools.items():
                                st.write(f"**{server_name}:** ({len(tools)} tools)")
                                if not tools:
                                    st.write("  - No tools found!")
                                    
                                    # Try to get tools directly from the connection
                                    if server_name in mcp_client.connections:
                                        connection = mcp_client.connections[server_name]
                                        st.write(f"  - Connection connected: {connection.connected}")
                                        st.write(f"  - Connection capabilities: {getattr(connection, 'capabilities', 'None')}")
                                        
                                        # Try to call tools/list directly
                                        try:
                                            direct_tools = connection.get_tools()
                                            st.write(f"  - Direct tools call result: {direct_tools}")
                                            
                                            # Try to send a raw JSON-RPC tools/list request
                                            import asyncio
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
                                            import traceback
                                            st.write(f"  - Full error: {traceback.format_exc()}")
                                else:
                                    for tool in tools:
                                        st.write(f"  - {tool.get('name')} ({tool.get('description', 'No description')})")
                            
                            # Find a command tool
                            command_server = None
                            command_tool = None
                            
                            for server_name, tools in available_tools.items():
                                st.write(f"Checking server: {server_name}")
                                for tool in tools:
                                    tool_name = tool.get('name', '').lower()
                                    tool_desc = tool.get('description', '').lower()
                                    st.write(f"  Tool: {tool_name} | Description: {tool_desc}")
                                    
                                    # More flexible matching - check tool name and description
                                    if (any(kw in tool_name for kw in ['command', 'execute', 'run', 'shell', 'terminal', 'bash', 'cmd']) or
                                        any(kw in tool_desc for kw in ['command', 'execute', 'run', 'shell', 'terminal', 'bash', 'cmd'])):
                                        command_server = server_name
                                        command_tool = tool.get('name')
                                        st.write(f"✅ Found matching tool: {command_tool}")
                                        break
                                if command_tool:
                                    break
                            
                            # If no tools found but we have a terminal server, try known tool names
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
                            
                            if command_server and command_tool:
                                # Test with a simple command
                                schema = available_tools[command_server][0].get('inputSchema', {}) if available_tools[command_server] else {}
                                args = {}
                                properties = schema.get('properties', {})
                                
                                # Generate args based on schema
                                if 'command' in properties:
                                    args['command'] = 'pwd'
                                elif 'cmd' in properties:
                                    args['cmd'] = 'pwd'
                                elif 'shell_command' in properties:
                                    args['shell_command'] = 'pwd'
                                
                                st.write(f"Testing {command_server}.{command_tool} with args: {args}")
                                
                                result = asyncio.run(mcp_client.call_tool(command_server, command_tool, args))
                                
                                if result:
                                    st.success("✅ Terminal command successful!")
                                    st.json(result)
                                else:
                                    st.error("❌ No result from terminal command")
                            else:
                                st.error("❌ No terminal command tool found")
                                
                        except Exception as e:
                            st.error(f"❌ Terminal command failed: {e}")
                            import traceback
                            st.error(f"Full error: {traceback.format_exc()}")

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

        # Load MCP-specific system prompt if available
        if "mcp_system_context" not in st.session_state:
            default_mcp_prompt = """You are an advanced RAG assistant with enhanced capabilities through Model Context Protocol (MCP) integration. You have access to:

1. **Knowledge Base**: Comprehensive information from the indexed documents and data sources
2. **Web Search**: Real-time information from the internet via Brave Search
3. **External Tools**: Various MCP servers that can provide additional context and capabilities

**Instructions:**
- Always prioritize information from the knowledge base when available and relevant
- Use web search to supplement knowledge base information or get current information
- Clearly indicate when you're using enhanced capabilities (knowledge base, web search, etc.)
- Be transparent about your information sources
- If knowledge base and web search provide conflicting information, explain the discrepancy
- Provide comprehensive, accurate, and well-sourced responses
- When uncertain, acknowledge limitations and suggest additional resources

**Response Style:**
- Be helpful, accurate, and thorough
- Use clear, professional language
- Structure responses logically with proper formatting
- Include relevant context from all available sources
- Acknowledge when information comes from enhanced capabilities"""
            
            # Try to load from system_prompt_mcp.txt
            try:
                import os
                system_prompt_file = os.path.join(st.session_state.get('project_dir', ''), "system_prompt_mcp.txt")
                if os.path.exists(system_prompt_file):
                    with open(system_prompt_file, 'r', encoding='utf-8') as f:
                        st.session_state.mcp_system_context = f.read().strip()
                else:
                    st.session_state.mcp_system_context = default_mcp_prompt
            except Exception as e:
                logger.error(f"Failed to load MCP system prompt: {e}")
                st.session_state.mcp_system_context = default_mcp_prompt

        # System context
        system_context = st.text_area(
            "🎯 System Context (MCP Enhanced)",
            value=st.session_state.get("mcp_system_context", ""),
            height=150,
            help="System instructions for the MCP-enhanced assistant. This prompt is optimized for RAG + MCP capabilities."
        )
        st.session_state.mcp_system_context = system_context
        st.session_state.system_context = system_context  # Keep compatibility

        # Button to save MCP system prompt
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save MCP Prompt", help="Save current system context as system_prompt_mcp.txt"):
                try:
                    import os
                    system_prompt_file = os.path.join(st.session_state.get('project_dir', ''), "system_prompt_mcp.txt")
                    with open(system_prompt_file, 'w', encoding='utf-8') as f:
                        f.write(system_context)
                    st.success("MCP system prompt saved!")
                except Exception as e:
                    st.error(f"Failed to save: {e}")
        
        with col2:
            if st.button("🔄 Reset to Default", help="Reset to default MCP system prompt"):
                st.session_state.mcp_system_context = default_mcp_prompt
                st.rerun()

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

            # Show MCP enhancements if present (generic approach)
            if message.get("mcp_enhanced") and message["role"] == "assistant":
                with st.expander("🔌 MCP Enhancements Used"):
                    metadata = message.get("metadata", {})
                    mcp_results = {k: v for k, v in metadata.items() if k.startswith('mcp_') and k.endswith('_results')}
                    
                    if mcp_results:
                        for key, value in mcp_results.items():
                            intent = key.replace('mcp_', '').replace('_results', '')
                            server = value.get('server', 'Unknown')
                            tool = value.get('tool', 'Unknown')
                            data = value.get('data', {})
                            
                            # Choose appropriate emoji based on intent
                            emoji = _get_intent_emoji(intent)
                            
                            st.write(f"{emoji} **{intent.replace('_', ' ').title()}** ({server}.{tool}):")
                            
                            # Display the data appropriately
                            if isinstance(data, (list, dict)):
                                st.json(data)
                            else:
                                st.text(str(data))
                    else:
                        # Fallback for old format
                        for key, value in metadata.items():
                            if key in ["web_search_results", "directory_listing", "filesystem_search_results"]:
                                st.write(f"**{key.replace('_', ' ').title()}:**")
                                st.json(value)

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add a user message
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

                    # Show MCP enhancements if used (generic approach)
                    mcp_results = {k: v for k, v in updated_context.items() if k.startswith('mcp_') and k.endswith('_results')}
                    if mcp_results:
                        with st.expander("🔌 MCP Enhancements Used"):
                            for key, value in mcp_results.items():
                                intent = key.replace('mcp_', '').replace('_results', '')
                                server = value.get('server', 'Unknown')
                                tool = value.get('tool', 'Unknown')
                                data = value.get('data', {})
                                
                                # Choose appropriate emoji based on intent
                                emoji = _get_intent_emoji(intent)
                                
                                st.write(f"{emoji} **{intent.replace('_', ' ').title()}** ({server}.{tool}):")
                                
                                # Display the data appropriately
                                if isinstance(data, (list, dict)):
                                    st.json(data)
                                else:
                                    st.text(str(data))

                    # Check if any MCP tools were used (generic detection)
                    mcp_enhanced = any(key.startswith('mcp_') and key.endswith('_results') for key in updated_context.keys())
                    
                    # Add an assistant message to history
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

        # Add MCP enhancement info (generic approach)
        if message.get("mcp_enhanced"):
            metadata = message.get("metadata", {})
            enhancements = []
            
            # Extract MCP enhancements generically
            for key, value in metadata.items():
                if key.startswith('mcp_') and key.endswith('_results'):
                    intent = key.replace('mcp_', '').replace('_results', '')
                    server = value.get('server', 'MCP')
                    enhancements.append(f"{intent.replace('_', ' ')} ({server})")
            
            # Fallback to old format if no new format found
            if not enhancements:
                for key in ["web_search_results", "directory_listing", "filesystem_search_results"]:
                    if key in metadata:
                        enhancements.append(key.replace('_', ' '))
            
            if enhancements:
                lines.append(f"  [Enhanced with MCP: {', '.join(enhancements)}]")
            else:
                lines.append("  [Enhanced with MCP]")

        lines.append("")

    return "\n".join(lines)


# Utility functions for MCP integration
def get_mcp_status() -> Dict[str, Any]:
    """Get the current MCP status for display"""
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