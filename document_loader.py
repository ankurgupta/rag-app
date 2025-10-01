"""
Document loader module for handling PDF and text file uploads
"""
import os
import tempfile
from typing import List, Union
import PyPDF2
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


class DocumentProcessor:
    """Handles loading and processing of PDF and text documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: list = None):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators optimized for better chunking
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation marks
                "? ",    # Question marks
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Spaces
                ""       # Character level
            ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            is_separator_regex=False,
        )
    
    def load_pdf(self, file_path: str) -> str:
        """
        Load text content from a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return ""
    
    def load_text_file(self, file_path: str) -> str:
        """
        Load text content from a text file
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Error loading text file: {str(e)}")
            return ""
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """
        Process an uploaded file and return document chunks
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract text based on file type
            if uploaded_file.name.lower().endswith('.pdf'):
                text = self.load_pdf(tmp_file_path)
            elif uploaded_file.name.lower().endswith(('.txt', '.md')):
                text = self.load_text_file(tmp_file_path)
            else:
                st.error("Unsupported file type. Please upload PDF or text files.")
                return []
            
            if not text.strip():
                st.warning("No text content found in the uploaded file.")
                return []
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": uploaded_file.name,
                        "chunk_id": i,
                        "chunk_size": len(chunk)
                    }
                )
                documents.append(doc)
            
            return documents
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def process_multiple_files(self, uploaded_files: List) -> List[Document]:
        """
        Process multiple uploaded files
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            List of Document objects from all files
        """
        all_documents = []
        
        for uploaded_file in uploaded_files:
            st.info(f"Processing {uploaded_file.name}...")
            documents = self.process_uploaded_file(uploaded_file)
            all_documents.extend(documents)
        
        return all_documents
