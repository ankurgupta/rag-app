#!/bin/bash

# Cloud Deployment Script for RAG Application

echo "☁️  Preparing for Cloud Deployment"
echo "=================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: RAG application"
fi

# Check if remote is set
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "🔗 Please set up your GitHub remote:"
    echo "   git remote add origin https://github.com/your-username/rag-2.git"
    echo "   git push -u origin main"
    exit 1
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git add .
git commit -m "Update: Cloud deployment ready" || echo "No changes to commit"
git push origin main

echo ""
echo "✅ Code pushed to GitHub!"
echo ""
echo "🚀 Next steps for Streamlit Cloud:"
echo "1. Go to https://share.streamlit.io"
echo "2. Click 'New app'"
echo "3. Connect your GitHub repository"
echo "4. Set main file path: app.py"
echo "5. Set requirements file: requirements-cloud.txt"
echo "6. Deploy!"
echo ""
echo "📚 See CLOUD_DEPLOYMENT.md for detailed instructions"
