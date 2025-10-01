"""
Script to check available Gemini models
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def check_available_models():
    """Check what models are available with the current API key"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No API key found in .env file")
        return
    
    print(f"Checking models with API key: {api_key[:10]}...")
    
    # Try different API versions
    versions = ["v1beta", "v1"]
    
    for version in versions:
        print(f"\n=== Checking API version: {version} ===")
        url = f'https://generativelanguage.googleapis.com/{version}/models?key={api_key}'
        
        try:
            response = requests.get(url)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                print(f"Found {len(models)} models:")
                
                for model in models:
                    name = model.get('name', 'Unknown')
                    display_name = model.get('displayName', 'No display name')
                    description = model.get('description', 'No description')
                    supported_methods = model.get('supportedMethods', [])
                    
                    print(f"\nModel: {name}")
                    print(f"Display Name: {display_name}")
                    print(f"Description: {description}")
                    print(f"Supported Methods: {supported_methods}")
                    print("-" * 50)
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Exception: {str(e)}")

if __name__ == "__main__":
    check_available_models()
