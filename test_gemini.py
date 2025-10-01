"""
Test script to verify Gemini API connection with different models
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def test_gemini_models():
    """Test different Gemini models"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No API key found")
        return
    
    models_to_test = [
        "gemini-2.0-flash-001",
        "gemini-2.5-flash", 
        "gemini-2.5-pro"
    ]
    
    for model_name in models_to_test:
        print(f"\n=== Testing {model_name} ===")
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1
            )
            
            # Test with a simple message
            response = llm.invoke("Hello, this is a test. Please respond with 'Test successful'.")
            print(f"✅ {model_name}: {response.content}")
            
        except Exception as e:
            print(f"❌ {model_name}: {str(e)}")

if __name__ == "__main__":
    test_gemini_models()
