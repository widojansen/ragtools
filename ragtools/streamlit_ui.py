"""
Streamlit UI module for RAG applications
"""
import streamlit as st
from typing import Optional, Dict, Any

try:
    from .mcp_client import MCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class RagStreamlitUI:
    """
    A standard Streamlit UI for RAG applications
    """
    def __init__(
        self,
        project_name: str = "RAG Tools",
        page_title: str = "RAG Assistant",
        page_icon: str = "🧠",
        input_field_label: str = "Enter your query",
        input_field_default: str = "",
        input_field_placeholder: str = "Ask me anything...",
        input_field_help: str = "Type your question here and press Enter",
        project_dir: str ="",
        db_dir: str = "",
        knowledge_dir: str = "",
        llm: str = "",
        embeddings: str = "",
        collection_name: str = "",
        mcp_available: bool = MCP_AVAILABLE,
    ):
        """
        Initialize the Streamlit UI for RAG applications
        """
        self.project_name = project_name
        self.page_title = page_title
        self.page_icon = page_icon
        self.input_field_label = input_field_label
        self.input_field_default = input_field_default
        self.input_field_placeholder = input_field_placeholder
        self.input_field_help = input_field_help
        self.project_dir = project_dir
        self.db_dir = db_dir
        self.knowledge_dir = knowledge_dir
        self.llm = llm
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.mcp_available = mcp_available

        
        # Remove the st.set_page_config call from here
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.sidebar_state = "expanded"
            st.session_state.current_page = "chatbot"
        
            # Store UI configuration in the session state
            st.session_state.input_field_label = self.input_field_label
            st.session_state.input_field_default = self.input_field_default
            st.session_state.input_field_placeholder = self.input_field_placeholder
            st.session_state.input_field_help = self.input_field_help
            st.session_state.project_dir = self.project_dir
            st.session_state.db_dir = self.db_dir
            st.session_state.knowledge_dir = self.knowledge_dir
            st.session_state.llm = self.llm
            st.session_state.embeddings = self.embeddings
            st.session_state.collection_name = self.collection_name

        # Initialize MCP client in session state if available
        if MCP_AVAILABLE and 'mcp_client' not in st.session_state:
            st.session_state.mcp_client = MCPClient()

    def run(self):
        """Run the Streamlit UI application"""
        # Display sidebar
        with st.sidebar:
            st.title(f"{self.project_name}")
            st.divider()
            
            # Navigation
            st.subheader("Navigation")
            if st.button("💬 Chatbot", use_container_width=True):
                st.session_state.current_page = "chatbot"
                st.rerun()

            if st.button("💬 Chatbot with MCP", use_container_width=True):
                st.session_state.current_page = "chatbot_mcp"
                st.rerun()
            
            if st.button("📥 Import Data", use_container_width=True):
                st.session_state.current_page = "import_data"
                st.rerun()

            # Add the MCP Management button
            if MCP_AVAILABLE:
                if st.button("🔌 MCP Servers", use_container_width=True):
                    st.session_state.current_page = "mcp_management"
                    st.rerun()
            
            # Add more navigation buttons here as needed
            # if st.button("📊 Dashboard", use_container_width=True):
            #     st.session_state.current_page = "dashboard"
            #     st.rerun()
            
            st.divider()
            st.caption("© 2024 RAG Tools")
        
        # Display the selected page
        if st.session_state.current_page == "chatbot":
            self._display_chatbot_page()
        elif st.session_state.current_page == "chatbot_mcp":
            self._display_chatbot_mcp_page()
        elif st.session_state.current_page == "import_data":
            self._display_import_data_page()
        elif st.session_state.current_page == "mcp_management":
            self._display_mcp_management_page()

        # Add more page conditions as needed
        # elif st.session_state.current_page == "dashboard":
        #     self._display_dashboard_page()
    
    @staticmethod
    def _display_chatbot_page():
        """Display the Chatbot page"""
        try:
            # Import the chatbot module from the pages directory
            from .pages import chatbot
            
            # Call the run function
            chatbot.run()
        except ImportError as e:
            st.error(f"Error loading chatbot: {str(e)}")
            st.info("Make sure the chatbot.py file exists in the pages directory.")
        except Exception as e:
            st.error(f"Error displaying chatbot: {str(e)}")

    @staticmethod
    def _display_chatbot_mcp_page():
        """Display the Chatbot with MCP enhancement page"""
        try:
            # Import the chatbot module from the pages directory
            from .pages import chatbot_mcp

            # Call the run function
            chatbot_mcp.run()
        except ImportError as e:
            st.error(f"Error loading chatbot with MCP: {str(e)}")
            st.info("Make sure the chatbot_mcp.py file exists in the pages directory.")
        except Exception as e:
            st.error(f"Error displaying chatbot_mcp: {str(e)}")

    @staticmethod
    def _display_import_data_page():
        """Display the Import Data page"""
        try:
            # Import the import_data module from the pages directory
            from .pages import import_data
            
            # Call the run function
            import_data.run()
        except ImportError as e:
            st.error(f"Error loading import data page: {str(e)}")
            st.info("Make sure the import_data.py file exists in the pages directory.")
        except Exception as e:
            st.error(f"Error displaying import data page: {str(e)}")

    @staticmethod
    def _display_mcp_management_page():
        """Display the MCP Management page"""
        try:
            from .pages import mcp_management
            mcp_management.run()
        except ImportError as e:
            st.error(f"MCP management not available: {str(e)}")
            st.info("Install MCP dependencies: pip install mcp httpx anyio")
        except Exception as e:
            st.error(f"Error displaying MCP management: {str(e)}")

def launch_streamlit_ui(config: Optional[Dict[str, Any]] = None):
    """
    Launch the Streamlit UI
    
    Args:
        config: Optional configuration dictionary with any of these keys:
            - project_name: Name of the project (default: "RAG Tools")
            - page_title: Title displayed in browser tab (default: "RAG Assistant")
            - page_icon: Icon displayed in browser tab (default: "🧠")
            - input_field_label: Label for the input field (default: "Enter your query")
            - input_field_default: Default text for the input field (default: "")
            - input_field_placeholder: Placeholder text (default: "Ask me anything...")
            - input_field_help: Help text for the input field (default: "Type your question here and press Enter")
    """
    # Create configuration with defaults for all parameters
    ui_config = {
        "project_name": "RAG Tools",
        "page_title": "RAG Assistant",
        "page_icon": "🧠",
        "input_field_label": "Enter your query",
        "input_field_default": "",
        "input_field_placeholder": "Ask me anything...",
        "input_field_help": "Type your question here and press Enter"
    }

    # Add MCP status to config if available
    if MCP_AVAILABLE:
        config["mcp_available"] = True
    
    # Update with the provided config if any
    if config:
        ui_config.update(config)
        print(f"Launching Streamlit UI with config: {ui_config}")

    # Create the UI
    ui = RagStreamlitUI(**ui_config)

    # Modify the session state to persist vectorstore reference
    if "vectorstore" not in st.session_state:
        print("vectorstore not found in session state")

        try:
            # Import and initialize vector store
            import ragtools.import_rag_data as rag_import

            # Only create/load the vector store if it doesn't exist or needs updating
            vectorstore = rag_import.create_or_load_vectorstore(
                embeddings=st.session_state.embeddings,
                content_directory=st.session_state.knowledge_dir,
                db_directory=st.session_state.db_dir,
                force_reload=False
            )
            print("Vectorstore loaded")
            # Store in the session state
            st.session_state.vectorstore = vectorstore
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            st.session_state.vectorstore = None
    
    # Run the UI
    ui.run()


if __name__ == "__main__":
    launch_streamlit_ui()