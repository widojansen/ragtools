# MCP Chatbot Process Documentation

## Overview

The MCP (Model Context Protocol) enhanced chatbot is a sophisticated system that combines three powerful technologies:
- **RAG (Retrieval Augmented Generation)**: Access to your knowledge base
- **MCP Integration**: Real-time tool execution (terminal commands, file operations, web search)
- **LLM Processing**: AI-powered response generation via Ollama

This document provides a detailed walkthrough of how these systems work together to create an enhanced conversational AI experience.

## 🔄 Complete Chat Process Flow

### 1. Initialization Phase

```python
def run():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize MCP client if available
    if MCP_AVAILABLE and 'mcp_client' not in st.session_state:
        st.session_state.mcp_client = MCPClient()
        # Auto-connect to enabled servers
        asyncio.run(st.session_state.mcp_client.connect_all_servers())
```

**What happens:**
- Creates empty message history for the chat session
- Initializes the MCP client to manage server connections
- Connects to all enabled MCP servers (like Terminal Server)
- Loads server configurations from `mcp_config.json`

### 2. User Input Processing

```python
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
```

**What happens:**
- User enters a query (e.g., "run pwd", "show me my desktop folder")
- Message is immediately added to the chat history
- User message is displayed in the chat interface

### 3. Core Processing - The `ask_llm()` Function

This is the heart of the system where three enhancement layers work together:

#### 3A. RAG (Retrieval Augmented Generation) Enhancement

```python
# First, get RAG context from vector store
try:
    from ragtools.import_rag_data import get_retriever
    
    knowledge_dir = st.session_state.knowledge_dir
    db_dir = st.session_state.db_dir
    retriever = get_retriever(knowledge_dir, db_dir)
    
    if retriever:
        docs = retriever.get_relevant_documents(prompt)
        rag_context = "\n\n".join(doc.page_content for doc in docs)
        logger.info(f"Retrieved {len(docs)} documents from RAG")
```

**Process:**
1. **Semantic Search**: Uses embeddings to find documents similar to the user's query
2. **Document Retrieval**: Extracts relevant content from your knowledge base
3. **Context Building**: Combines retrieved documents into a context string
4. **Relevance Scoring**: Orders results by similarity to the query

**Example:**
- Query: "dental procedures billing"
- RAG finds: Documents about dental billing codes, procedures, insurance policies
- Context: Relevant sections from 3-4 most similar documents

#### 3B. MCP (Model Context Protocol) Enhancement

```python
# Use MCP enhancement if available
if MCP_AVAILABLE and 'mcp_client' in st.session_state:
    mcp_client = st.session_state.mcp_client
    mcp_integration = MCPRagIntegration(mcp_client)
    
    # Enhance query with MCP
    enhanced_context = asyncio.run(
        mcp_integration.enhance_query_with_mcp(prompt, model_context)
    )
```

This triggers a sophisticated multi-step MCP process:

##### Step 1: Intent Analysis

```python
async def _analyze_query_intent(self, query: str, available_tools: Dict[str, List[Dict]]):
    query_lower = query.lower()
    
    for server_name, tools in available_tools.items():
        for tool in tools:
            # Match query to tool based on keywords and descriptions
            intent, args = await self._match_tool_to_query(
                query_lower, tool_name, tool_desc, schema, server_name
            )
```

**Intent Detection Examples:**
- `"run pwd"` → `terminal_command` intent
- `"show desktop folder"` → `directory_listing` intent  
- `"search for Python files"` → `search` intent
- `"what's the weather"` → `web_search` intent

##### Step 2: Tool Matching

```python
async def _match_tool_to_query(self, query: str, tool_name: str, tool_desc: str, schema: Dict, server_name: str):
    # Terminal command detection
    if any(kw in query for kw in ['run', 'execute', 'command', 'terminal', 'shell']):
        if any(kw in server_name.lower() for kw in ['terminal', 'command']):
            return 'terminal_command', await self._generate_command_args(query, schema)
    
    # Common system commands
    common_commands = ['ls', 'pwd', 'whoami', 'date', 'ps', 'top', 'df']
    if any(cmd in query.split() for cmd in common_commands):
        return 'terminal_command', await self._generate_command_args(query, schema)
    
    # Directory/file operations
    if any(kw in query for kw in ['show', 'list', 'content', 'folder', 'directory']):
        # Try dedicated file tools first, fallback to terminal commands
        return 'directory_listing', await self._generate_file_args(query, schema)
```

**Tool Matching Process:**
1. **Keyword Analysis**: Scans query for command-related keywords
2. **Server Capability Check**: Matches intent to available MCP servers
3. **Tool Selection**: Chooses appropriate tool (e.g., `terminal_cmd` for shell commands)
4. **Schema Adaptation**: Generates arguments based on tool's input schema

##### Step 3: Command Extraction & Translation

```python
def _extract_command_from_query(self, query: str) -> str:
    query_lower = query.lower()
    
    # Direct command extraction
    if any(kw in query_lower for kw in ['run', 'execute', 'command']):
        # Extract command after keywords: "run pwd" → "pwd"
        
    # Natural language to command translation
    if 'current directory' in query_lower:
        return 'pwd'
    elif 'who am i' in query_lower:
        return 'whoami'
    elif 'system info' in query_lower:
        return 'uname -a'
    elif 'disk space' in query_lower:
        return 'df -h'
    elif 'memory usage' in query_lower:
        return 'free -h'
```

**Translation Examples:**
- `"run pwd"` → `"pwd"`
- `"what's my current directory"` → `"pwd"`
- `"who am i"` → `"whoami"`
- `"show disk space"` → `"df -h"`
- `"list desktop files"` → `"ls /Users/widojansen/Desktop"`

##### Step 4: Tool Execution

```python
# Execute the MCP tool
result = await self.mcp_client.call_tool(server_name, tool_name, args)

if result:
    # Store result with metadata
    result_key = f"mcp_{intent}_results"
    enhanced_context[result_key] = {
        'server': server_name,        # "Terminal Server"
        'tool': tool_name,           # "terminal_cmd"
        'intent': intent,            # "terminal_command"
        'data': result,              # {"stdout": "...", "stderr": "...", "return_code": 0}
        'args': args                 # {"command": "pwd"}
    }
```

**What happens:**
1. **JSON-RPC Call**: Sends request to your Terminal Server
2. **Command Execution**: Server runs the shell command
3. **Result Capture**: Gets stdout, stderr, and return code
4. **Metadata Storage**: Stores result with full context about what was executed

#### 3C. Context Combination

```python
# Combine all enhancement sources
combined_context = enhanced_system_context

# Add RAG knowledge base context
if rag_context:
    combined_context += f"\n\nKnowledge Base Context:\n{rag_context}"

# Add MCP tool results
for key, value in enhanced_context.items():
    if key.startswith('mcp_') and key.endswith('_results'):
        intent = key.replace('mcp_', '').replace('_results', '')
        server = value.get('server', 'unknown')
        tool = value.get('tool', 'unknown')
        data = value.get('data', '')
        
        # Format for LLM consumption
        context_addition = f"\n\n{intent.replace('_', ' ').title()} from {server} ({tool}):\n{_format_mcp_data(data)}"
        combined_context += context_addition
```

**Context Structure:**
```
System Prompt: "You are an advanced RAG assistant with MCP capabilities..."

Knowledge Base Context:
[Retrieved documents about dental procedures, billing codes, etc.]

Terminal Command from Terminal Server (terminal_cmd):
{"stdout": "/Users/widojansen/Projects/Agents/RAG/BraceFaceRag", "stderr": "", "return_code": 0}

User Query: "run pwd"
```

### 4. LLM Processing

```python
# Prepare messages for the language model
messages = []
if combined_context.strip():
    messages.append({"role": "system", "content": combined_context})
messages.append({"role": "user", "content": prompt})

# Call Ollama LLM
ollama_url = "http://127.0.0.1:11434/api/chat"
payload = {
    "model": llm_model,           # e.g., "qwen3:8b"
    "messages": messages,
    "stream": False,
    "options": {"temperature": temperature}
}

response = requests.post(ollama_url, json=payload, timeout=120)
llm_response = response.json().get('message', {}).get('content', '')
```

**What happens:**
1. **Message Formatting**: Structures conversation for the LLM
2. **API Call**: Sends to local Ollama server
3. **AI Processing**: LLM generates response based on enriched context
4. **Response Extraction**: Gets the generated text

### 5. Response Enhancement & Transparency

```python
# Track enhancement sources
enhancement_info = []
if rag_context:
    enhancement_info.append("knowledge base")

# Detect MCP enhancements
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

# Add transparency footer
if enhancement_info:
    llm_response += f"\n\n*This response was enhanced with: {', '.join(enhancement_info)}*"
```

**Enhancement Transparency Examples:**
- `*This response was enhanced with: knowledge base*`
- `*This response was enhanced with: terminal command (Terminal Server)*`
- `*This response was enhanced with: knowledge base, directory listing (filesystem)*`

### 6. UI Display & Storage

```python
# Display the AI response
st.markdown(response)

# Show MCP enhancements in expandable section
mcp_results = {k: v for k, v in updated_context.items() if k.startswith('mcp_') and k.endswith('_results')}
if mcp_results:
    with st.expander("🔌 MCP Enhancements Used"):
        for key, value in mcp_results.items():
            intent = key.replace('mcp_', '').replace('_results', '')
            server = value.get('server', 'Unknown')
            tool = value.get('tool', 'Unknown')
            data = value.get('data', {})
            
            # Choose appropriate emoji
            emoji = _get_intent_emoji(intent)  # 💻 for terminal_command, 📁 for directory_listing
            
            st.write(f"{emoji} **{intent.replace('_', ' ').title()}** ({server}.{tool}):")
            
            # Display the raw tool results
            if isinstance(data, (list, dict)):
                st.json(data)
            else:
                st.text(str(data))

# Store complete conversation with metadata
assistant_message = {
    "role": "assistant",
    "content": response,
    "mcp_enhanced": mcp_enhanced,
    "metadata": updated_context if mcp_enhanced else {}
}
st.session_state.messages.append(assistant_message)
```

**UI Features:**
- **Immediate Response**: AI answer displayed instantly
- **Expandable Details**: Click to see raw MCP tool results
- **Visual Indicators**: Emojis for different tool types (💻🔍📁🌐)
- **Full History**: Complete conversation stored with metadata
- **Download Option**: Export chat with enhancement details

## 🔄 Complete Example: "run pwd" Flow

Let's trace through a complete interaction:

### Input
User types: `"run pwd"`

### Step-by-Step Process

1. **Input Processing**
   ```
   User message: "run pwd"
   Added to chat history
   Displayed in UI
   ```

2. **RAG Enhancement**
   ```
   Query: "run pwd"
   Knowledge base search: Finds 2 documents about terminal commands
   RAG context: "Terminal commands in Linux/macOS..."
   ```

3. **MCP Intent Analysis**
   ```
   Query analysis: "run pwd"
   Keywords detected: ["run", "pwd"]
   Intent determined: terminal_command
   ```

4. **Tool Matching**
   ```
   Available servers: ["Terminal Server"]
   Available tools: ["terminal_cmd"]
   Match: Terminal Server.terminal_cmd
   ```

5. **Command Extraction**
   ```
   Raw query: "run pwd"
   Extracted command: "pwd"
   Generated args: {"command": "pwd"}
   ```

6. **Tool Execution**
   ```
   JSON-RPC call: Terminal Server.terminal_cmd({"command": "pwd"})
   Server execution: subprocess.run("pwd")
   Result: {"stdout": "/Users/widojansen/Projects/Agents/RAG/BraceFaceRag", "stderr": "", "return_code": 0}
   ```

7. **Context Building**
   ```
   System prompt: "You are an advanced RAG assistant..."
   + RAG context: "Terminal commands documentation..."
   + MCP context: "Terminal Command from Terminal Server: /Users/widojansen/Projects/Agents/RAG/BraceFaceRag"
   + User query: "run pwd"
   ```

8. **LLM Processing**
   ```
   API call to Ollama (qwen3:8b)
   Input: Combined context + user query
   Output: "Your current working directory is /Users/widojansen/Projects/Agents/RAG/BraceFaceRag. This is the root folder of your BraceFaceRag project."
   ```

9. **Response Enhancement**
   ```
   Base response + "*This response was enhanced with: knowledge base, terminal command (Terminal Server)*"
   ```

10. **UI Display**
    ```
    Main response displayed
    Expandable section: "🔌 MCP Enhancements Used"
    └── 💻 Terminal Command (Terminal Server.terminal_cmd):
        {"stdout": "/Users/widojansen/Projects/Agents/RAG/BraceFaceRag", "stderr": "", "return_code": 0}
    ```

## 🛠️ Supported MCP Intents

The system supports multiple intent types:

| Intent | Emoji | Description | Example Queries |
|--------|-------|-------------|-----------------|
| `terminal_command` | 💻 | Execute shell commands | "run pwd", "whoami", "ls -la" |
| `directory_listing` | 📁 | List folder contents | "show desktop folder", "list files" |
| `file_operation` | 📄 | Read/write files | "read config.json", "show file content" |
| `search` | 🔍 | Search operations | "find Python files", "search for logs" |
| `web_search` | 🌐 | Web searches | "search for Python tutorials" |
| `database_query` | 💽 | Database operations | "query user table", "select from logs" |

## 🔧 Configuration

### MCP Server Configuration (`mcp_config.json`)
```json
{
  "servers": {
    "Terminal Server": {
      "command": "/Users/widojansen/.asdf/shims/uv",
      "args": ["run", "/Users/widojansen/Projects/Agents/MCP/shellserver/server.py"],
      "env": {},
      "description": "Terminal command execution on local computer",
      "enabled": true
    }
  }
}
```

### System Prompt (`system_prompt_mcp.txt`)
```
You are an advanced RAG assistant with enhanced capabilities through Model Context Protocol (MCP) integration. You have access to:

1. **Knowledge Base**: Comprehensive information from indexed documents
2. **Terminal Commands**: Execute real shell commands via Terminal Server
3. **External Tools**: Various MCP servers for enhanced capabilities

Always be transparent about which tools you're using and acknowledge the sources of your enhanced capabilities.
```

## 🚀 Key Benefits

1. **Multi-Source Intelligence**: Combines knowledge base + real-time tool execution
2. **Transparency**: Always shows which tools were used
3. **Flexibility**: Works with any MCP server (filesystem, database, API, etc.)
4. **Real-time Capability**: Execute actual commands and get live results
5. **Extensibility**: Easy to add new MCP servers and tools

## 🔍 Debugging & Monitoring

The system includes comprehensive debugging features:

- **Debug Info Panel**: Shows connected servers and available tools
- **Test Buttons**: Manual testing of MCP functionality
- **Detailed Logging**: Step-by-step process logging
- **Raw Data Display**: View exact tool results and API responses
- **Enhancement Tracking**: Complete audit trail of what tools were used

This creates a powerful, transparent, and extensible conversational AI system that can access your knowledge base while also interacting with your computer and external systems in real-time!