# App2.py
from config import configure_environment
configure_environment()
import streamlit as st
import os
import sys
import traceback
from ui import render_interface


# Create the data directory if it doesn't exist
os.makedirs("data", exist_ok=True)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


# Ensure the search_results.txt file exists
if not os.path.exists("data/search_results.txt"):
    with open("data/search_results.txt", "w", encoding="utf-8") as f:
        pass

# Explicitly set the working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add module directory to path
sys.path.append(script_dir)

# Import after setting up paths
try:
    from logic import initialize_session_state
    from ui import render_interface
except Exception as e:
    st.error(f"Failed to import modules: {str(e)}")
    st.error(traceback.format_exc())
    st.stop()

# Initialize session state with error handling
try:
    initialize_session_state()
except Exception as e:
    st.error(f"Failed to initialize session state: {str(e)}")
    st.error(traceback.format_exc())
    st.stop()

# Then render the interface with error handling
if __name__ == "__main__":
    try:
        render_interface()
    except Exception as e:
        st.error(f"Failed to render interface: {str(e)}")
        st.error(traceback.format_exc())
        st.stop()