#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности очистки результатов анализов
"""

import sys
import os
import logging
from datetime import datetime

# Добавляем корневую папку в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_cleanup_test_results.log'),
        logging.StreamHandler()
    ]
)

def test_cleanup_function():
    """Тестирует функцию очистки результатов анализов"""
    logging.info("\n=== Тестирование функции очистки результатов анализов ===")
    
    try:
        # Импортируем необходимые модули
        from config import supabase
        from structured_tests_agent import TestExtractionAgent
        
        # Создаем агент
        agent = TestExtractionAgent(supabase)
        
        # Получаем первого пользователя для тестирования
        response = supabase.table("doc_patient_profiles").select("user_id").limit(1).execute()
        if not response.data:
            logging.error("Пользователи не найдены")
            return
        
        test_user_id = response.data[0]["user_id"]
        logging.info(f"Тестируем на пользователе: {test_user_id}")
        
        # Показываем анализы до очистки
        tests_before = supabase.table("doc_structured_test_results").select("*").eq("user_id", test_user_id).execute()
        logging.info(f"Анализов до очистки: {len(tests_before.data) if tests_before.data else 0}")
        
        if tests_before.data:
            logging.info("\n--- Анализы ДО очистки ---")
            for test in tests_before.data:
                logging.info(f"ID: {test.get('id')}")
                logging.info(f"Название: {test.get('test_name')}")
                logging.info(f"Результат: '{test.get('result')}'")
                logging.info(f"Тест-система: '{test.get('test_system')}'")
                logging.info(f"Оборудование: '{test.get('equipment')}'")
                logging.info("---")
        
        # Выполняем очистку
        logging.info("\n--- Выполняем очистку ---")
        cleanup_result = await agent.cleanup_existing_test_results(test_user_id)
        
        if cleanup_result.get("success"):
            cleaned_count = cleanup_result.get("cleaned_count", 0)
            logging.info(f"Очистка завершена успешно. Очищено: {cleaned_count}")
            
            if cleaned_count > 0:
                updated_tests = cleanup_result.get("updated_tests", [])
                logging.info("\n--- Детали очистки ---")
                for test in updated_tests:
                    logging.info(f"Анализ: {test.get('test_name')}")
                    logging.info(f"  Результат: '{test.get('old_result')}' -> '{test.get('new_result')}'")
                    logging.info(f"  Тест-система: '{test.get('old_test_system')}' -> '{test.get('new_test_system')}'")
                    logging.info(f"  Оборудование: '{test.get('old_equipment')}' -> '{test.get('new_equipment')}'")
                    logging.info("  ---")
        else:
            logging.error(f"Ошибка при очистке: {cleanup_result.get('message')}")
            return
        
        # Показываем анализы после очистки
        tests_after = supabase.table("doc_structured_test_results").select("*").eq("user_id", test_user_id).execute()
        logging.info(f"\nАнализов после очистки: {len(tests_after.data) if tests_after.data else 0}")
        
        if tests_after.data:
            logging.info("\n--- Анализы ПОСЛЕ очистки ---")
            for test in tests_after.data:
                logging.info(f"ID: {test.get('id')}")
                logging.info(f"Название: {test.get('test_name')}")
                logging.info(f"Результат: '{test.get('result')}'")
                logging.info(f"Тест-система: '{test.get('test_system')}'")
                logging.info(f"Оборудование: '{test.get('equipment')}'")
                logging.info("---")
        
        logging.info(f"\nРезультат: очищено {cleaned_count} результатов анализов")
        
    except ImportError as e:
        logging.error(f"Ошибка импорта: {e}")
    except Exception as e:
        logging.error(f"Ошибка в тестировании: {e}")

def test_reprocess_function():
    """Тестирует функцию переобработки медицинских записей"""
    logging.info("\n=== Тестирование функции переобработки медицинских записей ===")
    
    try:
        # Импортируем необходимые модули
        from config import supabase
        from structured_tests_agent import TestExtractionAgent
        
        # Создаем агент
        agent = TestExtractionAgent(supabase)
        
        # Получаем первого пользователя для тестирования
        response = supabase.table("doc_patient_profiles").select("user_id").limit(1).execute()
        if not response.data:
            logging.error("Пользователи не найдены")
            return
        
        test_user_id = response.data[0]["user_id"]
        logging.info(f"Тестируем на пользователе: {test_user_id}")
        
        # Показываем медицинские записи до переобработки
        records_before = supabase.table("doc_medical_records").select("*").eq("user_id", test_user_id).execute()
        logging.info(f"Медицинских записей до переобработки: {len(records_before.data) if records_before.data else 0}")
        
        # Показываем анализы до переобработки
        tests_before = supabase.table("doc_structured_test_results").select("*").eq("user_id", test_user_id).execute()
        logging.info(f"Анализов до переобработки: {len(tests_before.data) if tests_before.data else 0}")
        
        # Выполняем переобработку
        logging.info("\n--- Выполняем переобработку ---")
        reprocess_result = await agent.reprocess_medical_records(test_user_id)
        
        if reprocess_result.get("success"):
            tests_count = reprocess_result.get("tests_count", 0)
            logging.info(f"Переобработка завершена успешно. Извлечено: {tests_count}")
        else:
            logging.error(f"Ошибка при переобработке: {reprocess_result.get('message')}")
            return
        
        # Показываем анализы после переобработки
        tests_after = supabase.table("doc_structured_test_results").select("*").eq("user_id", test_user_id).execute()
        logging.info(f"\nАнализов после переобработки: {len(tests_after.data) if tests_after.data else 0}")
        
        if tests_after.data:
            logging.info("\n--- Анализы ПОСЛЕ переобработки ---")
            for test in tests_after.data:
                logging.info(f"ID: {test.get('id')}")
                logging.info(f"Название: {test.get('test_name')}")
                logging.info(f"Результат: '{test.get('result')}'")
                logging.info(f"Тест-система: '{test.get('test_system')}'")
                logging.info(f"Оборудование: '{test.get('equipment')}'")
                logging.info("---")
        
        logging.info(f"\nРезультат: переобработано {tests_count} анализов")
        
    except ImportError as e:
        logging.error(f"Ошибка импорта: {e}")
    except Exception as e:
        logging.error(f"Ошибка в тестировании: {e}")

async def main():
    """Основная функция"""
    logging.info("Запуск тестирования функций очистки результатов анализов")
    
    try:
        # Тестируем очистку
        await test_cleanup_function()
        
        # Тестируем переобработку
        await test_reprocess_function()
        
        logging.info("\n=== Тестирование завершено ===")
        
    except Exception as e:
        logging.error(f"Ошибка в основной функции: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
