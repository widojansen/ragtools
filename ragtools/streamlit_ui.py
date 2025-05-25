"""
Streamlit UI module for RAG applications
"""
import os
import streamlit as st
from typing import Optional, Dict, Any, List, Callable


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
        input_field_help: str = "Type your question here and press Enter"
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
        
        # Remove the st.set_page_config call from here
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.sidebar_state = "expanded"
            st.session_state.current_page = "chatbot"

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
            
            if st.button("📥 Import Data", use_container_width=True):
                st.session_state.current_page = "import_data"
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
        elif st.session_state.current_page == "import_data":
            self._display_import_data_page()
        # Add more page conditions as needed
        # elif st.session_state.current_page == "dashboard":
        #     self._display_dashboard_page()
    
    def _display_chatbot_page(self):
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

    def _display_import_data_page(self):
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
    
    # Update with provided config if any
    if config:
        ui_config.update(config)
    
    # Create and run the UI
    ui = RagStreamlitUI(**ui_config)
    ui.run()


if __name__ == "__main__":
    launch_streamlit_ui()