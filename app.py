"""
Main Streamlit application for RAG with Gemini API
"""
import os
import streamlit as st
import pandas as pd
import faiss
from dotenv import load_dotenv
from document_loader import DocumentProcessor
from vector_store import VectorStore
from gemini_rag import GeminiRAG

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chat with Gemini",
    page_icon="🤖",
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

def initialize_gemini_with_model(model_name: str = "gemini-2.0-flash-001"):
    """Initialize Gemini RAG system with specified model"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set your GOOGLE_API_KEY in the .env file")
        return None
    
    try:
        gemini_rag = GeminiRAG(api_key, model_name)
        if gemini_rag.test_connection():
            return gemini_rag
        else:
            return None
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None

def initialize_gemini():
    """Initialize Gemini RAG system (backward compatibility)"""
    return initialize_gemini_with_model("gemini-2.0-flash-001")

def add_conversation_to_vector_store(question: str, answer: str, vector_store):
    """Add Q&A pair to vector store for future reference"""
    from langchain.schema import Document
    
    # Create conversation document
    conversation_text = f"Question: {question}\nAnswer: {answer}"
    
    conversation_doc = Document(
        page_content=conversation_text,
        metadata={
            "source": "conversation_history",
            "type": "qa_pair",
            "question": question,
            "answer": answer,
            "timestamp": str(pd.Timestamp.now())
        }
    )
    
    # Add to vector store
    vector_store.add_documents([conversation_doc])
    
    return conversation_doc

def main():
    """Main application function"""
    
    # Header
    st.title("🤖 RAG Chat with Gemini API")
    st.markdown("Upload PDF or text documents and chat with them using AI!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            help="Enter your Google API key for Gemini"
        )
        
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Model selection
        model_choice = st.selectbox(
            "Select Gemini Model",
            ["gemini-2.0-flash-001", "gemini-2.5-flash", "gemini-2.5-pro"],
            help="gemini-2.0-flash-001 is fastest, gemini-2.5-pro is most capable"
        )
        
        # Initialize Gemini
        if st.button("Initialize Gemini", type="primary"):
            with st.spinner("Initializing Gemini..."):
                st.session_state.gemini_rag = initialize_gemini_with_model(model_choice)
                if st.session_state.gemini_rag:
                    st.success(f"Gemini {model_choice} initialized successfully!")
                else:
                    st.error("Failed to initialize Gemini")
        
        # Document processing settings
        st.subheader("📄 Document Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, help="Larger chunks preserve more context but may be less precise")
        chunk_overlap = st.slider("Chunk Overlap", 100, 500, 200, help="Higher overlap ensures continuity between chunks")
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        search_method = st.selectbox(
            "Search Method",
            ["Semantic", "Hybrid"],
            help="Semantic: Pure vector similarity. Hybrid: Combines semantic + keyword matching"
        )
        num_docs = st.slider("Number of documents to retrieve", 2, 10, 4, help="More documents = more context but potentially more noise")
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            if search_method == "Hybrid":
                alpha = st.slider(
                    "Semantic vs Keyword Weight", 
                    0.0, 1.0, 0.7, 0.1,
                    help="0.0 = pure keyword, 1.0 = pure semantic, 0.7 = balanced"
                )
            else:
                alpha = 0.7
            
            score_threshold = st.slider(
                "Similarity Score Threshold",
                0.0, 1.0, 0.0, 0.05,
                help="Minimum similarity score for including results (0.0 = include all)"
            )
            
            temperature = st.slider(
                "AI Response Creativity",
                0.0, 1.0, 0.1, 0.1,
                help="Lower = more focused, Higher = more creative"
            )
            
            # Conversation memory settings
            st.subheader("🧠 Conversation Memory")
            use_conversation_memory = st.checkbox(
                "Enable Conversation Memory",
                value=True,
                help="Store Q&A pairs in vector store for future reference"
            )
            
            conversation_weight = st.slider(
                "Conversation vs Document Weight",
                0.0, 1.0, 0.3, 0.1,
                help="0.0 = only documents, 1.0 = only conversation, 0.3 = balanced"
            ) if use_conversation_memory else 0.0
    
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
                # Initialize document processor with optimized settings
                doc_processor = DocumentProcessor(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
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
            st.subheader("📊 Vector Store Stats")
            stats = st.session_state.vector_store.get_stats()
            
            # Count conversation entries
            conversation_count = 0
            document_count = 0
            for doc in st.session_state.vector_store.documents:
                if doc.metadata.get('type') == 'qa_pair':
                    conversation_count += 1
                else:
                    document_count += 1
            
            stats['document_sources'] = document_count
            stats['conversation_entries'] = conversation_count
            st.json(stats)
            
            # Conversation memory management
            if conversation_count > 0:
                st.subheader("🧠 Conversation Memory")
                st.info(f"Stored {conversation_count} Q&A pairs in memory")
                
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
            
            # Save/Load options
            st.subheader("💾 Save/Load Vector Store")
            save_path = st.text_input("Save path", "vector_store/index")
            if st.button("Save Vector Store"):
                st.session_state.vector_store.save_index(save_path)
            
            load_path = st.text_input("Load path", "vector_store/index")
            if st.button("Load Vector Store"):
                st.session_state.vector_store = VectorStore()
                st.session_state.vector_store.load_index(load_path)
    
    with col2:
        st.header("💬 Chat with Your Documents")
        
        # Check if everything is ready
        if not st.session_state.gemini_rag:
            st.warning("Please initialize Gemini in the sidebar first")
        elif not st.session_state.vector_store:
            st.warning("Please upload and process documents first")
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
                            st.session_state.gemini_rag.llm.temperature = temperature
                        
                        response = st.session_state.gemini_rag.chat_with_context(
                            prompt, 
                            st.session_state.vector_store, 
                            k=num_docs,
                            use_hybrid=(search_method == "Hybrid"),
                            alpha=alpha
                        )
                        
                        st.write(response["answer"])
                        
                        # Add conversation to vector store if enabled
                        if use_conversation_memory and conversation_weight > 0:
                            try:
                                add_conversation_to_vector_store(prompt, response["answer"], st.session_state.vector_store)
                                st.success("💾 Added to conversation memory")
                            except Exception as e:
                                st.warning(f"Could not add to conversation memory: {str(e)}")
                        
                        # Show enhanced sources
                        with st.expander("📚 Sources & Scores"):
                            st.write(f"**Search Method:** {response['search_method'].title()}")
                            st.write(f"**Number of Sources:** {response['num_sources']}")
                            if use_conversation_memory:
                                st.write(f"**Conversation Memory:** Enabled (Weight: {conversation_weight:.1f})")
                            
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
