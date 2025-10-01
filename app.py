"""
Main Streamlit application for RAG with Gemini API
Uses Streamlit secrets for API key management
"""
import os
import streamlit as st
import faiss
from document_loader import DocumentProcessor
from vector_store import VectorStore
from gemini_rag import GeminiRAG
from langchain.schema import Document
from config import (
    GOOGLE_API_KEY, DEFAULT_MODEL, AVAILABLE_MODELS, APP_TITLE, APP_ICON,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_NUM_DOCS, 
    DEFAULT_ALPHA, DEFAULT_TEMPERATURE, DEFAULT_SEARCH_METHOD,
    SHOW_CONFIGURATION_SIDEBAR, AUTO_INITIALIZE_GEMINI
)

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "gemini_rag" not in st.session_state:
    st.session_state.gemini_rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_api_key():
    """
    Get API key from multiple sources in order of preference:
    1. Streamlit secrets (best for cloud deployment)
    2. Environment variable
    3. Config file
    """
    # Option 1: Streamlit secrets (best for cloud)
    try:
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            return st.secrets['GOOGLE_API_KEY']
    except:
        pass
    
    # Option 2: Environment variable
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key and env_key != "your_google_api_key_here":
        return env_key
    
    # Option 3: Config file
    if GOOGLE_API_KEY and GOOGLE_API_KEY != "your_google_api_key_here":
        return GOOGLE_API_KEY
    
    return None

def initialize_gemini_with_model(model_name: str = DEFAULT_MODEL):
    """Initialize Gemini RAG system with specified model"""
    try:
        api_key = get_api_key()
        
        if not api_key:
            return None, "No API key found. Please configure your Google API key."
            
        gemini_rag = GeminiRAG(api_key, model_name)
        if gemini_rag.test_connection():
            return gemini_rag, "Success"
        else:
            return None, "Failed to connect to Gemini API. Please check your API key."
    except Exception as e:
        return None, f"Failed to initialize Gemini: {str(e)}"

def initialize_gemini():
    """Initialize Gemini RAG system (backward compatibility)"""
    return initialize_gemini_with_model(DEFAULT_MODEL)

def add_conversation_to_vector_store(question: str, answer: str, vector_store):
    """Add Q&A pair to vector store for future reference"""
    conversation_text = f"Question: {question}\nAnswer: {answer}"
    conversation_doc = Document(
        page_content=conversation_text,
        metadata={
            "source": "conversation_memory",
            "type": "qa_pair",
            "question": question,
            "answer": answer,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    )
    vector_store.add_documents([conversation_doc])
    return conversation_doc

def main():
    """Main application function"""
    
    # Auto-initialize Gemini if enabled (moved inside main function)
    if AUTO_INITIALIZE_GEMINI and st.session_state.gemini_rag is None:
        try:
            gemini_rag, message = initialize_gemini_with_model(DEFAULT_MODEL)
            if gemini_rag:
                st.session_state.gemini_rag = gemini_rag
            else:
                # Store error message for display
                st.session_state.gemini_error = message
        except Exception as e:
            st.session_state.gemini_error = f"Failed to initialize Gemini: {str(e)}"
    
    # Header
    st.title("🤖 RAG Chat with Gemini API")
    st.markdown("Upload PDF or text documents and chat with them using AI!")
    
    # Show API key configuration status
    api_key = get_api_key()
    if api_key:
        st.success("✅ API key configured and ready!")
    else:
        st.warning("⚠️ API key not found. Please configure it using one of the methods below.")
    
    # Configuration section (only show if no API key)
    if not api_key:
        with st.expander("🔧 API Key Configuration", expanded=True):
            st.markdown("""
            **Choose your preferred method:**
            
            1. **Streamlit Secrets** (Recommended for cloud):
               Create `.streamlit/secrets.toml`:
               ```toml
               GOOGLE_API_KEY = "your_key_here"
               ```
            
            2. **Environment Variable** (For local development):
               ```bash
               export GOOGLE_API_KEY="your_key_here"
               ```
            
            3. **Config File** (For local development):
               Edit `config.py` and set your API key
            """)
    
    # Show error if initialization failed
    if hasattr(st.session_state, 'gemini_error'):
        st.error(f"❌ {st.session_state.gemini_error}")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.gemini_rag:
            st.success("✅ Gemini Ready")
        else:
            st.error("❌ Gemini Not Ready")
    
    with col2:
        if st.session_state.vector_store and st.session_state.vector_store.documents:
            doc_count = len([d for d in st.session_state.vector_store.documents if d.metadata.get('type') != 'qa_pair'])
            st.info(f"📚 {doc_count} Documents")
        else:
            st.warning("📚 No Documents")
    
    with col3:
        if st.session_state.chat_history:
            st.info(f"💬 {len(st.session_state.chat_history)} Messages")
        else:
            st.info("💬 No Messages")
    
    st.divider()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Upload Documents")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload PDF or text files to create your knowledge base"
        )
        
        # Process documents
        if uploaded_files and st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                # Initialize document processor with hardcoded settings
                doc_processor = DocumentProcessor(
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP
                )
                
                # Process files
                documents = doc_processor.process_multiple_files(uploaded_files)
                
                if documents:
                    # Initialize vector store
                    st.session_state.vector_store = VectorStore()
                    st.session_state.vector_store.add_documents(documents)
                    
                    st.success(f"Successfully processed {len(documents)} document chunks!")
                else:
                    st.error("No documents were processed successfully")
        
        # Vector store stats
        if st.session_state.vector_store:
            st.subheader("📊 Statistics")
            stats = st.session_state.vector_store.get_stats()
            
            # Count conversation entries
            conversation_count = 0
            document_count = 0
            for doc in st.session_state.vector_store.documents:
                if doc.metadata.get('type') == 'qa_pair':
                    conversation_count += 1
                else:
                    document_count += 1
            
            st.metric("Document Sources", document_count)
            st.metric("Conversation Entries", conversation_count)
            st.metric("Total Chunks", stats.get('total_documents', 0))
            
            # Clear conversation memory
            if conversation_count > 0:
                if st.button("🗑️ Clear Conversation Memory", type="secondary"):
                    # Remove conversation entries from vector store
                    st.session_state.vector_store.documents = [
                        doc for doc in st.session_state.vector_store.documents 
                        if doc.metadata.get('type') != 'qa_pair'
                    ]
                    # Rebuild index without conversation entries
                    if st.session_state.vector_store.documents:
                        texts = [doc.page_content for doc in st.session_state.vector_store.documents]
                        embeddings = st.session_state.vector_store.embedding_model.encode(texts)
                        faiss.normalize_L2(embeddings)
                        st.session_state.vector_store.index = faiss.IndexFlatIP(st.session_state.vector_store.dimension)
                        st.session_state.vector_store.index.add(embeddings.astype('float32'))
                    st.success("Conversation memory cleared!")
                    st.rerun()
    
    with col2:
        st.header("💬 Chat with Your Documents")
        
        # Check if everything is ready
        if not st.session_state.gemini_rag:
            st.error("❌ Gemini is not initialized. Please configure your API key first.")
        elif not st.session_state.vector_store:
            st.warning("⚠️ Please upload and process documents first")
        else:
            # Chat interface
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("📚 Sources"):
                            for i, (source, score) in enumerate(zip(message["sources"], message["similarity_scores"])):
                                st.write(f"**Source {i+1}:** {source} (Similarity: {score:.3f})")
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your documents..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Update Gemini temperature if available
                        if hasattr(st.session_state.gemini_rag.llm, 'temperature'):
                            st.session_state.gemini_rag.llm.temperature = DEFAULT_TEMPERATURE
                        
                        response = st.session_state.gemini_rag.chat_with_context(
                            prompt, 
                            st.session_state.vector_store, 
                            k=DEFAULT_NUM_DOCS,
                            use_hybrid=(DEFAULT_SEARCH_METHOD == "hybrid"),
                            alpha=DEFAULT_ALPHA
                        )
                        
                        st.write(response["answer"])
                        
                        # Add conversation to memory
                        try:
                            add_conversation_to_vector_store(prompt, response["answer"], st.session_state.vector_store)
                        except Exception as e:
                            st.warning(f"Could not add to conversation memory: {str(e)}")
                        
                        # Show sources
                        with st.expander("📚 Sources & Scores"):
                            st.write(f"**Search Method:** {response['search_method'].title()}")
                            st.write(f"**Number of Sources:** {response['num_sources']}")
                            
                            for i, (source, score) in enumerate(zip(response["sources"], response["similarity_scores"])):
                                source_info = f"**Source {i+1}:** {source} (Similarity: {score:.3f})"
                                
                                # Add hybrid scores if available
                                if "combined_scores" in response:
                                    combined_score = response["combined_scores"][i]
                                    keyword_score = response["keyword_scores"][i]
                                    source_info += f" | Combined: {combined_score:.3f} | Keyword: {keyword_score:.3f}"
                                
                                st.write(source_info)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response["sources"],
                    "similarity_scores": response["similarity_scores"]
                })
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()
