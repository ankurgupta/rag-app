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
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
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
        
        # Create prompt template for RAG
        self.prompt_template = PromptTemplate(
            template="""You are a helpful AI assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based on the context provided above
- If the answer is not in the context, say "I don't have enough information to answer this question based on the provided context"
- Be concise and accurate
- If relevant, cite specific parts of the context

Answer:""",
            input_variables=["context", "question"]
        )
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            similarity_score = doc.metadata.get('similarity_score', 0)
            context_parts.append(f"Source {i} ({source}, similarity: {similarity_score:.3f}):\n{doc.page_content}\n")
        
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
    
    def chat_with_context(self, question: str, vector_store, k: int = 4) -> dict:
        """
        Complete RAG pipeline: retrieve relevant documents and generate answer
        
        Args:
            question: User's question
            vector_store: VectorStore instance for retrieval
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant documents
        relevant_docs = vector_store.similarity_search(question, k=k)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_docs)
        
        # Prepare response
        response = {
            "answer": answer,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in relevant_docs],
            "similarity_scores": [doc.metadata.get('similarity_score', 0) for doc in relevant_docs],
            "num_sources": len(relevant_docs)
        }
        
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
