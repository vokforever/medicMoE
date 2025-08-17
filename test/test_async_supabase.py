#!/usr/bin/env python3
"""
Test Supabase in async context similar to main application
"""

import os
import asyncio
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def generate_user_uuid(telegram_user_id: int) -> str:
    """Генерирует детерминированный UUID на основе Telegram user ID."""
    telegram_namespace = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')
    return str(uuid.uuid5(telegram_namespace, str(telegram_user_id)))

async def test_async_supabase():
    """Test Supabase in async context"""
    print("🔍 Testing Supabase in async context...")
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
        
        # Test multiple async operations
        print("\n🧪 Testing multiple async operations...")
        
        # Simulate the exact operations from your logs
        test_user_id = generate_user_uuid(12345)
        print(f"Generated user_id: {test_user_id}")
        
        # Operation 1: Get patient profiles (this was successful in logs)
        print("\n📋 Operation 1: Getting patient profiles...")
        try:
            response1 = supabase.table("doc_patient_profiles").select("*").eq("user_id", test_user_id).execute()
            print(f"✅ Patient profiles: {response1}")
        except Exception as e:
            print(f"❌ Patient profiles failed: {e}")
        
        # Operation 2: Get medical records (this was successful in logs)
        print("\n📋 Operation 2: Getting medical records...")
        try:
            response2 = supabase.table("doc_medical_records").select("*").eq("user_id", test_user_id).order("created_at", desc=True).execute()
            print(f"✅ Medical records: {response2}")
        except Exception as e:
            print(f"❌ Medical records failed: {e}")
        
        # Operation 3: Try to insert a test record (this might trigger the error)
        print("\n📋 Operation 3: Trying to insert test record...")
        try:
            response3 = supabase.table("doc_patient_profiles").insert({
                "user_id": test_user_id,
                "name": "Test User",
                "telegram_id": 12345,
                "created_at": "2024-01-01T00:00:00Z"
            }).execute()
            print(f"✅ Insert successful: {response3}")
        except Exception as e:
            print(f"❌ Insert failed: {e}")
            print(f"Error type: {type(e)}")
            
            # Try to get more details about the error
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response headers: {e.response.headers}")
                print(f"Response body: {e.response.text}")
        
        # Operation 4: Try another operation that might fail
        print("\n📋 Operation 4: Testing another operation...")
        try:
            response4 = supabase.table("doc_medical_records").select("count").execute()
            print(f"✅ Count query successful: {response4}")
        except Exception as e:
            print(f"❌ Count query failed: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")

async def main():
    """Main async function"""
    await test_async_supabase()

if __name__ == "__main__":
    asyncio.run(main())
