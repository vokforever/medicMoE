#!/usr/bin/env python3
"""
Debug Supabase connection and API key issues
"""

import os
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def generate_user_uuid(telegram_user_id: int) -> str:
    """
    Генерирует детерминированный UUID на основе Telegram user ID.
    Один и тот же Telegram user ID всегда будет генерировать один и тот же UUID.
    """
    # Создаем namespace UUID для Telegram (используем фиксированный UUID)
    telegram_namespace = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')
    
    # Создаем UUID на основе namespace и user_id
    return str(uuid.uuid5(telegram_namespace, str(telegram_user_id)))

def debug_supabase():
    """Debug Supabase connection"""
    print("🔍 Debugging Supabase connection...")
    print("=" * 50)
    
    # Check environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    print(f"SUPABASE_URL: {supabase_url}")
    print(f"SUPABASE_KEY: {supabase_key[:20] if supabase_key else 'None'}...")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials!")
        return
    
    try:
        # Create client
        print("\n🔧 Creating Supabase client...")
        supabase: Client = create_client(
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )
        
        # Test basic connection
        print("\n🧪 Testing basic connection...")
        response = supabase.table("doc_patient_profiles").select("count").limit(1).execute()
        print(f"✅ Basic connection successful! Response: {response}")
        
        # Test the specific function that's failing
        print("\n🧪 Testing get_patient_profile function...")
        test_user_id = generate_user_uuid(12345)  # Test with a dummy Telegram ID
        print(f"Generated user_id: {test_user_id}")
        
        try:
            response = supabase.table("doc_patient_profiles").select("*").eq("user_id", test_user_id).execute()
            print(f"✅ get_patient_profile successful! Response: {response}")
        except Exception as e:
            print(f"❌ get_patient_profile failed: {e}")
            print(f"Error type: {type(e)}")
            
            # Try to get more details about the error
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response headers: {e.response.headers}")
                print(f"Response body: {e.response.text}")
        
        # Test medical records function
        print("\n🧪 Testing get_medical_records function...")
        try:
            response = supabase.table("doc_medical_records").select("*").eq("user_id", test_user_id).order("created_at", desc=True).execute()
            print(f"✅ get_medical_records successful! Response: {response}")
        except Exception as e:
            print(f"❌ get_medical_records failed: {e}")
            print(f"Error type: {type(e)}")
            
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response headers: {e.response.headers}")
                print(f"Response body: {e.response.text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")
        
        # Try to get more details about the error
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response headers: {e.response.headers}")
            print(f"Response body: {e.response.text}")

if __name__ == "__main__":
    debug_supabase()
