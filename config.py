"""
Configuration file for RAG Application
"""
# Google API Configuration
GOOGLE_API_KEY = "AIzaSyDgv1-cg0MVONQcEXuS1JlUCJYOgjhiUUU"

# Gemini Model Configuration
DEFAULT_MODEL = "gemini-2.0-flash-001"
AVAILABLE_MODELS = [
    "gemini-2.0-flash-001",
    "gemini-2.5-flash", 
    "gemini-2.5-pro"
]

# Document Processing Configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval Configuration
DEFAULT_NUM_DOCS = 4
DEFAULT_ALPHA = 0.7
DEFAULT_TEMPERATURE = 0.1
DEFAULT_SEARCH_METHOD = "semantic"  # "semantic" or "hybrid"

# UI Configuration
SHOW_CONFIGURATION_SIDEBAR = False
AUTO_INITIALIZE_GEMINI = True

# Application Configuration
APP_TITLE = "RAG Chat with Gemini API"
APP_ICON = "🤖"
