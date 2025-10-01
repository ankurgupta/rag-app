# RAG Chat Application with Gemini API

A Retrieval-Augmented Generation (RAG) application that allows you to upload PDF and text documents, create a searchable knowledge base, and chat with your documents using Google's Gemini AI.

## Features

- 📄 **Document Upload**: Support for PDF and text files (.txt, .md)
- 🔍 **Semantic Search**: Vector-based similarity search using sentence transformers
- 🤖 **AI Chat**: Interactive chat with your documents using Gemini API
- 💾 **Persistent Storage**: Save and load vector stores for reuse
- 🎨 **Modern UI**: Clean Streamlit interface with real-time chat
- 📊 **Analytics**: View document statistics and similarity scores

## Prerequisites

- Python 3.8 or higher
- Google API key for Gemini (get it from [Google AI Studio](https://makersuite.google.com/app/apikey))

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `env_template.txt` to `.env`
   - Add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Set up Gemini**:
   - Enter your Google API key in the sidebar
   - Click "Initialize Gemini"

4. **Upload documents**:
   - Upload PDF or text files using the file uploader
   - Adjust chunk size and overlap settings if needed
   - Click "Process Documents"

5. **Start chatting**:
   - Ask questions about your uploaded documents
   - View source citations and similarity scores
   - Clear chat history when needed

## Configuration

### Document Processing
- **Chunk Size**: Size of text chunks (500-2000 characters)
- **Chunk Overlap**: Overlap between chunks (100-500 characters)

### Retrieval Settings
- **Number of Documents**: How many relevant documents to retrieve (2-10)

### Vector Store
- **Save/Load**: Persist your vector store for future use
- **Statistics**: View document count and index information

## File Structure

```
rag-2/
├── app.py                 # Main Streamlit application
├── document_loader.py     # PDF/text document processing
├── vector_store.py        # Vector embeddings and similarity search
├── gemini_rag.py         # Gemini API integration
├── requirements.txt      # Python dependencies
├── env_template.txt      # Environment variables template
└── README.md            # This file
```

## How It Works

1. **Document Processing**: Uploaded files are processed and split into chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in a FAISS index for fast similarity search
4. **Query Processing**: User questions are converted to embeddings
5. **Retrieval**: Most similar document chunks are retrieved
6. **Generation**: Gemini generates answers based on retrieved context

## Dependencies

- `streamlit`: Web interface
- `langchain`: LLM framework
- `langchain-google-genai`: Gemini integration
- `pypdf2`: PDF processing
- `sentence-transformers`: Text embeddings
- `faiss-cpu`: Vector similarity search
- `python-dotenv`: Environment variable management

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Google API key is correctly set in the `.env` file
2. **PDF Processing Error**: Ensure PDF files are not corrupted or password-protected
3. **Memory Issues**: Reduce chunk size or number of documents for large files
4. **Connection Error**: Check your internet connection and API key validity

### Performance Tips

- Use smaller chunk sizes for more precise retrieval
- Adjust the number of retrieved documents based on your needs
- Save vector stores to avoid reprocessing documents
- Clear chat history regularly to manage memory usage

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this application.

## Support

If you encounter any issues or have questions, please check the troubleshooting section or create an issue in the repository.
