"""
Example usage of the RAG system components
"""
from document_loader import DocumentProcessor
from vector_store import VectorStore
from gemini_rag import GeminiRAG
from config import GOOGLE_API_KEY

def main():
    """Example of how to use the RAG system programmatically"""
    
    # Initialize components
    print("Initializing RAG system...")
    
    # Document processor
    doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Vector store
    vector_store = VectorStore()
    
    # Gemini RAG (uses hardcoded API key)
    gemini_rag = GeminiRAG(GOOGLE_API_KEY)
    
    # Test connection
    if not gemini_rag.test_connection():
        print("Failed to connect to Gemini API")
        return
    
    print("✅ RAG system initialized successfully!")
    
    # Example: Process a text document
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that can perform tasks that typically require human intelligence. 
    These tasks include learning, reasoning, problem-solving, perception, and language understanding.
    
    Machine Learning is a subset of AI that focuses on the development of algorithms and 
    statistical models that enable computer systems to improve their performance on a specific 
    task through experience, without being explicitly programmed.
    
    Deep Learning is a subset of machine learning that uses artificial neural networks with 
    multiple layers to model and understand complex patterns in data.
    """
    
    # Create a sample document
    from langchain.schema import Document
    sample_doc = Document(
        page_content=sample_text,
        metadata={"source": "sample_ai_text.txt", "chunk_id": 0}
    )
    
    # Add to vector store
    print("Adding sample document to vector store...")
    vector_store.add_documents([sample_doc])
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        "What is the difference between machine learning and deep learning?",
        "How do neural networks work?"
    ]
    
    print("\n" + "="*50)
    print("EXAMPLE QUERIES AND ANSWERS")
    print("="*50)
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        
        # Get response
        response = gemini_rag.chat_with_context(query, vector_store, k=2)
        
        print(f"🤖 Answer: {response['answer']}")
        print(f"📚 Sources: {response['sources']}")
        print(f"📊 Similarity Scores: {[f'{score:.3f}' for score in response['similarity_scores']]}")
        print("-" * 50)

if __name__ == "__main__":
    main()
