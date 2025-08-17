#!/usr/bin/env python3
"""
Тестовый скрипт для проверки доступности моделей Groq
Проверяет правильность названий моделей и их доступность через API
"""

import os
import requests
import json
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def test_groq_models():
    """Тестирует доступность моделей Groq"""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY не найден в переменных окружения")
        return False
    
    print("🔑 GROQ_API_KEY найден")
    print(f"📡 Проверяю доступность моделей Groq...")
    
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model["id"] for model in models_data.get("data", [])]
            
            print(f"✅ Успешно получен список моделей ({len(available_models)} моделей)")
            
            # Проверяем наши vision модели
            vision_models = [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "meta-llama/llama-4-maverick-17b-128e-instruct"
            ]
            
            print("\n🔍 Проверяю vision модели:")
            for model in vision_models:
                if model in available_models:
                    print(f"  ✅ {model} - ДОСТУПНА")
                else:
                    print(f"  ❌ {model} - НЕ ДОСТУПНА")
            
            # Проверяем text модели
            text_models = [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "meta-llama/llama-guard-4-12b"
            ]
            
            print("\n🔍 Проверяю text модели:")
            for model in text_models:
                if model in available_models:
                    print(f"  ✅ {model} - ДОСТУПНА")
                else:
                    print(f"  ❌ {model} - НЕ ДОСТУПНА")
            
            # Показываем все доступные модели
            print(f"\n📋 Все доступные модели Groq ({len(available_models)}):")
            for i, model in enumerate(available_models[:20], 1):  # Показываем первые 20
                print(f"  {i:2d}. {model}")
            
            if len(available_models) > 20:
                print(f"  ... и еще {len(available_models) - 20} моделей")
            
            return True
            
        else:
            print(f"❌ Ошибка API: {response.status_code}")
            print(f"Ответ: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при запросе: {e}")
        return False

def test_groq_chat():
    """Тестирует простой чат с Groq"""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY не найден")
        return False
    
    print("\n🧪 Тестирую простой чат с Groq...")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": "Привет! Как дела?"}
        ],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ Чат успешен!")
            print(f"📝 Ответ: {content}")
            return True
        else:
            print(f"❌ Ошибка чата: {response.status_code}")
            print(f"Ответ: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при чате: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Тестирование Groq API")
    print("=" * 50)
    
    # Тест 1: Проверка доступности моделей
    models_ok = test_groq_models()
    
    if models_ok:
        # Тест 2: Простой чат
        chat_ok = test_groq_chat()
        
        if chat_ok:
            print("\n🎉 Все тесты прошли успешно!")
        else:
            print("\n⚠️ Тест моделей прошел, но чат не работает")
    else:
        print("\n❌ Тест моделей не прошел")
    
    print("\n" + "=" * 50)
    print("Тестирование завершено")
