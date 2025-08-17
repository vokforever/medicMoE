#!/usr/bin/env python3
"""
Test script for Cerebras API connection and model availability
"""

import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_cerebras_connection():
    """Test Cerebras API connection and model availability"""
    
    # Check if API key is set
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("❌ CEREBRAS_API_KEY not found in environment variables")
        return False
    
    print(f"✅ CEREBRAS_API_KEY found: {api_key[:10]}...")
    
    # Initialize client
    try:
        client = OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key
        )
        print("✅ Cerebras client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Cerebras client: {e}")
        return False
    
    # Test simple API call
    try:
        print("🔍 Testing API connection...")
        
        # Try to list models first
        try:
            models_response = client.models.list()
            print(f"✅ Models API call successful")
            print(f"📋 Available models: {[model.id for model in models_response.data]}")
        except Exception as e:
            print(f"⚠️ Models API call failed: {e}")
        
        # Test with the specific model
        model_name = "qwen-3-235b-a22b-thinking-2507"
        print(f"🧪 Testing model: {model_name}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, please respond with just 'OK'."}
        ]
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=100,
            temperature=0.1
        )
        
        response = completion.choices[0].message.content
        print(f"✅ Model response: {response}")
        
        # Check for thinking process
        if hasattr(completion.choices[0], 'thinking'):
            thinking = completion.choices[0].thinking
            print(f"🧠 Thinking process: {thinking}")
        elif hasattr(completion.choices[0].message, 'thinking'):
            thinking = completion.choices[0].message.thinking
            print(f"🧠 Thinking process: {thinking}")
        else:
            print("ℹ️ No thinking process found in response")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Testing Cerebras API connection...")
    print("=" * 50)
    
    success = await test_cerebras_connection()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed! Cerebras API is working correctly.")
    else:
        print("❌ Some tests failed. Please check your configuration.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
