"""
Vector store module for document embeddings and retrieval
"""
import os
import pickle
from typing import List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import streamlit as st


class VectorStore:
    """Handles document embeddings and similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            return
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Initialize FAISS index if it doesn't exist
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents.extend(documents)
        
        st.success(f"Added {len(documents)} documents to vector store. Total documents: {len(self.documents)}")
    
    def similarity_search(self, query: str, k: int = 4, score_threshold: float = 0.0) -> List[Document]:
        """
        Perform similarity search with improved ranking
        
        Args:
            query: Query string
            k: Number of similar documents to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of most similar Document objects
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for more results than needed for better filtering
        search_k = min(k * 3, len(self.documents))
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Filter and rank results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and score >= score_threshold:
                doc = self.documents[idx]
                # Add similarity score to metadata
                doc.metadata['similarity_score'] = float(score)
                results.append(doc)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x.metadata['similarity_score'], reverse=True)
        
        # Return top k results
        return results[:k]
    
    def hybrid_search(self, query: str, k: int = 4, alpha: float = 0.7) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword matching
        
        Args:
            query: Query string
            k: Number of documents to return
            alpha: Weight for semantic search (1-alpha for keyword search)
            
        Returns:
            List of most relevant Document objects
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Semantic search
        semantic_results = self.similarity_search(query, k * 2)
        
        # Keyword search (simple implementation)
        query_words = set(query.lower().split())
        keyword_scores = {}
        
        for i, doc in enumerate(self.documents):
            doc_words = set(doc.page_content.lower().split())
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            keyword_score = intersection / union if union > 0 else 0
            keyword_scores[i] = keyword_score
        
        # Combine scores
        combined_results = []
        for doc in semantic_results:
            doc_idx = self.documents.index(doc)
            semantic_score = doc.metadata['similarity_score']
            keyword_score = keyword_scores.get(doc_idx, 0)
            
            # Weighted combination
            combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
            
            doc.metadata['combined_score'] = combined_score
            doc.metadata['keyword_score'] = keyword_score
            combined_results.append(doc)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.metadata['combined_score'], reverse=True)
        
        return combined_results[:k]
    
    def save_index(self, filepath: str) -> None:
        """
        Save the vector store to disk
        
        Args:
            filepath: Path to save the index
        """
        if self.index is None:
            st.warning("No index to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save documents and metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        st.success(f"Vector store saved to {filepath}")
    
    def load_index(self, filepath: str) -> bool:
        """
        Load the vector store from disk
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load documents
            with open(f"{filepath}.pkl", 'rb') as f:
                self.documents = pickle.load(f)
            
            st.success(f"Vector store loaded from {filepath}")
            return True
            
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": self.model_name,
            "dimension": self.dimension
        }
