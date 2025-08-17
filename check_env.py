#!/usr/bin/env python3
"""
Check environment variables and configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check all required environment variables"""
    print("🔍 Checking environment variables...")
    print("=" * 50)
    
    # Required variables
    required_vars = [
        "TELEGRAM_BOT_TOKEN",
        "OPENROUTER_API_KEY", 
        "TAVILY_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY"
    ]
    
    # Optional variables
    optional_vars = [
        "CEREBRAS_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "OPENROUTER_DAILY_LIMIT",
        "CEREBRAS_DAILY_LIMIT",
        "GROQ_DAILY_LIMIT"
    ]
    
    print("📋 Required variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value[:10]}..." if len(value) > 10 else f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: NOT SET")
    
    print("\n📋 Optional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value[:10]}..." if len(value) > 10 else f"  ✅ {var}: {value}")
        else:
            print(f"  ⚠️ {var}: NOT SET (optional)")
    
    print("\n🔧 Configuration:")
    
    # Check Cerebras configuration
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if cerebras_key:
        print(f"  🧠 Cerebras API Key: {'✅ Set' if cerebras_key else '❌ Not set'}")
        print(f"  🧠 Cerebras Base URL: https://api.cerebras.ai/v1")
    else:
        print("  🧠 Cerebras: ⚠️ API key not set")
    
    # Check OpenRouter configuration
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        print(f"  🌐 OpenRouter API Key: {'✅ Set' if openrouter_key else '❌ Not set'}")
        print(f"  🌐 OpenRouter Base URL: https://openrouter.ai/api/v1")
    else:
        print("  🌐 OpenRouter: ⚠️ API key not set")
    
    print("=" * 50)
    
    # Summary
    missing_required = [var for var in required_vars if not os.getenv(var)]
    if missing_required:
        print(f"❌ Missing required variables: {', '.join(missing_required)}")
        return False
    else:
        print("✅ All required variables are set!")
        return True

if __name__ == "__main__":
    check_environment()
