#!/usr/bin/env python3
"""
Тестовый скрипт для проверки ИИ-функции очистки дубликатов медицинских записей
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация Supabase
supabase: Client = create_client(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)

async def test_ai_duplicate_detection():
    """
    Тестирует ИИ-функцию определения дубликатов
    """
    logging.info("=== Тестирование ИИ-функции определения дубликатов ===")
    
    # Получаем первого пользователя для тестирования
    try:
        response = supabase.table("doc_patient_profiles").select("user_id").limit(1).execute()
        if not response.data:
            logging.error("Пользователи не найдены")
            return
        
        test_user_id = response.data[0]["user_id"]
        logging.info(f"Тестируем на пользователе: {test_user_id}")
        
        # Получаем записи пользователя
        records_response = supabase.table("doc_medical_records").select("*").eq("user_id", test_user_id).order("created_at", desc=True).execute()
        
        if not records_response.data or len(records_response.data) < 2:
            logging.info("Недостаточно записей для тестирования")
            return
        
        records = records_response.data
        logging.info(f"Найдено {len(records)} записей для анализа")
        
        # Показываем содержимое первых двух записей
        for i, record in enumerate(records[:2]):
            content = record.get("content", "")
            content_preview = content[:200] + "..." if len(content) > 200 else content
            logging.info(f"\n--- Запись {i+1} ---")
            logging.info(f"ID: {record.get('id')}")
            logging.info(f"Создано: {record.get('created_at')}")
            logging.info(f"Содержимое: {content_preview}")
        
        # Тестируем ИИ-функцию определения дубликатов
        logging.info("\n=== Тестирование ИИ-определения дубликатов ===")
        
        # Импортируем функции из main.py
        sys.path.append('.')
        try:
            from main import is_duplicate_by_ai
            logging.info("Функция is_duplicate_by_ai успешно импортирована")
            
            # Тестируем на двух записях
            if len(records) >= 2:
                record1 = records[0]
                record2 = records[1]
                
                logging.info(f"\nСравниваем записи {record1.get('id')} и {record2.get('id')}")
                
                is_dup = await is_duplicate_by_ai(
                    record1.get("content", ""),
                    record2.get("content", "")
                )
                
                logging.info(f"ИИ определил дубликат: {is_dup}")
                
                if is_dup:
                    logging.info("✅ ИИ правильно определил дубликат!")
                else:
                    logging.info("ℹ️ ИИ определил, что это разные записи")
            
        except ImportError as e:
            logging.error(f"Ошибка импорта: {e}")
            logging.info("Попробуйте запустить скрипт из корневой директории проекта")
        
    except Exception as e:
        logging.error(f"Ошибка в тестировании: {e}")

async def test_cleanup_function():
    """
    Тестирует функцию очистки дубликатов
    """
    logging.info("\n=== Тестирование функции очистки дубликатов ===")
    
    try:
        # Импортируем функции из main.py
        sys.path.append('.')
        from main import cleanup_duplicate_medical_records
        
        # Получаем первого пользователя
        response = supabase.table("doc_patient_profiles").select("user_id").limit(1).execute()
        if not response.data:
            logging.error("Пользователи не найдены")
            return
        
        test_user_id = response.data[0]["user_id"]
        
        # Показываем записи до очистки
        records_before = supabase.table("doc_medical_records").select("*").eq("user_id", test_user_id).execute()
        logging.info(f"Записей до очистки: {len(records_before.data) if records_before.data else 0}")
        
        # Очищаем дубликаты
        deleted_count = cleanup_duplicate_medical_records(test_user_id)
        logging.info(f"Удалено дубликатов: {deleted_count}")
        
        # Показываем записи после очистки
        records_after = supabase.table("doc_medical_records").select("*").eq("user_id", test_user_id).execute()
        logging.info(f"Записей после очистки: {len(records_after.data) if records_after.data else 0}")
        
    except ImportError as e:
        logging.error(f"Ошибка импорта: {e}")
    except Exception as e:
        logging.error(f"Ошибка в тестировании очистки: {e}")

async def main():
    """
    Основная функция
    """
    logging.info("Запуск тестирования ИИ-функций очистки дубликатов")
    
    # Тестируем ИИ-определение дубликатов
    await test_ai_duplicate_detection()
    
    # Тестируем функцию очистки
    await test_cleanup_function()
    
    logging.info("Тестирование завершено")

if __name__ == "__main__":
    asyncio.run(main())
