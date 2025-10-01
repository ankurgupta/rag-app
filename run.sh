#!/bin/bash

# RAG Chat Application Runner Script

echo "🤖 Starting RAG Chat Application with Gemini API"
echo "================================================"

# API key is hardcoded - no .env file needed
echo "🔑 Using hardcoded Google API key"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Start the application
echo "🚀 Starting Streamlit application..."
echo "   Open your browser to http://localhost:8501"
echo "   Press Ctrl+C to stop the application"
echo ""

streamlit run app.py
