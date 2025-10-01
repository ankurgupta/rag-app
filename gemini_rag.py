"""
Gemini RAG integration module
"""
import os
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st


class GeminiRAG:
    """RAG system using Gemini API for question answering"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-001"):
        """
        Initialize the Gemini RAG system
        
        Args:
            api_key: Google API key
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Create enhanced prompt template for RAG
        self.prompt_template = PromptTemplate(
            template="""You are an expert AI assistant that provides accurate, detailed answers based on the provided context. Your goal is to help users understand and find information from their documents.

CONTEXT INFORMATION:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Analyze the provided context carefully to find relevant information
2. Answer the question comprehensively using information from the context
3. If the context contains relevant information, provide a detailed and accurate answer
4. If the answer requires information not present in the context, clearly state what information is missing
5. When possible, cite specific sections or sources from the context
6. If the context doesn't contain enough information to answer the question, respond with: "I don't have enough information in the provided context to answer this question completely. The context covers [brief summary of what is covered], but doesn't include [what's missing]."
7. Be thorough but concise in your response
8. If the question is ambiguous, ask for clarification while providing what information you can from the context

ANSWER:""",
            input_variables=["context", "question"]
        )
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string with better organization
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        context_parts.append("=== RELEVANT DOCUMENT SECTIONS ===\n")
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            similarity_score = doc.metadata.get('similarity_score', 0)
            combined_score = doc.metadata.get('combined_score', similarity_score)
            keyword_score = doc.metadata.get('keyword_score', 0)
            
            # Create a more informative header
            header = f"--- Section {i} from {source} ---"
            if 'combined_score' in doc.metadata:
                header += f" (Relevance: {combined_score:.3f}, Keyword Match: {keyword_score:.3f})"
            else:
                header += f" (Similarity: {similarity_score:.3f})"
            
            context_parts.append(header)
            context_parts.append(f"{doc.page_content}\n")
        
        context_parts.append("=== END OF CONTEXT ===\n")
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, context_documents: List[Document]) -> str:
        """
        Generate answer using Gemini with retrieved context
        
        Args:
            question: User's question
            context_documents: Retrieved relevant documents
            
        Returns:
            Generated answer
        """
        try:
            # Format context
            context = self.format_context(context_documents)
            
            # Create prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            return response.content
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "Sorry, I encountered an error while generating the answer. Please try again."
    
    def chat_with_context(self, question: str, vector_store, k: int = 4, use_hybrid: bool = False, alpha: float = 0.7) -> dict:
        """
        Complete RAG pipeline: retrieve relevant documents and generate answer
        
        Args:
            question: User's question
            vector_store: VectorStore instance for retrieval
            k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid search (semantic + keyword)
            alpha: Weight for semantic search in hybrid mode
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant documents
        if use_hybrid:
            relevant_docs = vector_store.hybrid_search(question, k=k, alpha=alpha)
        else:
            relevant_docs = vector_store.similarity_search(question, k=k)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_docs)
        
        # Prepare response with enhanced metadata
        response = {
            "answer": answer,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in relevant_docs],
            "similarity_scores": [doc.metadata.get('similarity_score', 0) for doc in relevant_docs],
            "num_sources": len(relevant_docs),
            "search_method": "hybrid" if use_hybrid else "semantic"
        }
        
        # Add hybrid search scores if available
        if use_hybrid:
            response["combined_scores"] = [doc.metadata.get('combined_score', 0) for doc in relevant_docs]
            response["keyword_scores"] = [doc.metadata.get('keyword_score', 0) for doc in relevant_docs]
        
        return response
    
    def test_connection(self) -> bool:
        """
        Test the connection to Gemini API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_response = self.llm.invoke("Hello, this is a test message.")
            return True
        except Exception as e:
            st.error(f"Failed to connect to Gemini API: {str(e)}")
            return False
