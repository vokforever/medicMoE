#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функций очистки дубликатов медицинских записей
"""

import os
import sys
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

def check_duplicate_medical_record(user_id: str, content: str, record_type: str = "image_analysis") -> bool:
    """
    Проверяет, есть ли уже запись с таким же содержимым у пользователя.
    Возвращает True, если дубликат найден.
    """
    try:
        logging.info(f"Проверка дублирования для пользователя: {user_id}")
        
        # Получаем последние записи пользователя
        query = supabase.table("doc_medical_records").select("*").eq("user_id", user_id).eq("record_type", record_type)
        response = query.order("created_at", desc=True).limit(10).execute()
        
        if not response.data:
            logging.info("Записей для сравнения не найдено")
            return False
        
        # Нормализуем содержимое для сравнения
        normalized_content = normalize_content_for_comparison(content)
        
        # Проверяем на дублирование по нормализованному содержимому
        for record in response.data:
            record_content = record.get("content", "")
            normalized_record_content = normalize_content_for_comparison(record_content)
            
            if normalized_content == normalized_record_content:
                logging.info(f"Найден дубликат записи с ID: {record.get('id')}")
                return True
        
        logging.info("Дубликаты не найдены")
        return False
        
    except Exception as e:
        logging.error(f"Ошибка при проверке дублирования: {e}")
        return False

def normalize_content_for_comparison(content: str) -> str:
    """
    Нормализует содержимое для более точного сравнения.
    Убирает лишние пробелы, приводит к нижнему регистру, убирает незначимые различия.
    """
    if not content:
        return ""
    
    # Приводим к нижнему регистру
    normalized = content.lower()
    
    # Убираем лишние пробелы и переносы строк
    normalized = ' '.join(normalized.split())
    
    # Убираем пунктуацию и специальные символы
    import re
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Убираем множественные пробелы
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def cleanup_duplicate_medical_records(user_id: str, record_type: str = "image_analysis") -> int:
    """
    Удаляет дублирующиеся записи для пользователя, оставляя только самую новую.
    Возвращает количество удаленных записей.
    """
    try:
        logging.info(f"Очистка дубликатов для пользователя: {user_id}")
        
        # Получаем все записи пользователя
        query = supabase.table("doc_medical_records").select("*").eq("user_id", user_id).eq("record_type", record_type)
        response = query.order("created_at", desc=True).execute()
        
        if not response.data or len(response.data) <= 1:
            logging.info("Дубликатов для очистки не найдено")
            return 0
        
        records = response.data
        seen_normalized_contents = set()
        duplicates_to_delete = []
        
        # Находим дубликаты по нормализованному содержимому
        for record in records:
            content = record.get("content", "")
            normalized_content = normalize_content_for_comparison(content)
            
            if normalized_content in seen_normalized_contents:
                duplicates_to_delete.append(record["id"])
                logging.info(f"Найден дубликат: ID {record['id']} с содержимым: {content[:100]}...")
            else:
                seen_normalized_contents.add(normalized_content)
        
        if not duplicates_to_delete:
            logging.info("Дубликатов для удаления не найдено")
            return 0
        
        logging.info(f"Найдено {len(duplicates_to_delete)} дубликатов для удаления")
        
        # Удаляем дубликаты
        deleted_count = 0
        for record_id in duplicates_to_delete:
            try:
                delete_response = supabase.table("doc_medical_records").delete().eq("id", record_id).execute()
                if delete_response.data:
                    deleted_count += 1
                    logging.info(f"Удален дубликат с ID: {record_id}")
            except Exception as e:
                logging.error(f"Ошибка при удалении записи {record_id}: {e}")
        
        logging.info(f"Удалено {deleted_count} дублирующихся записей")
        return deleted_count
        
    except Exception as e:
        logging.error(f"Ошибка при очистке дубликатов: {e}")
        return 0

def cleanup_all_duplicates():
    """
    Очищает дубликаты у всех пользователей в системе.
    """
    try:
        logging.info("Начинаю автоматическую очистку дубликатов для всех пользователей")
        
        # Получаем всех пользователей
        response = supabase.table("doc_patient_profiles").select("user_id").execute()
        
        if not response.data:
            logging.info("Пользователей для очистки не найдено")
            return
        
        total_deleted = 0
        for profile in response.data:
            user_id = profile.get("user_id")
            if user_id:
                deleted_count = cleanup_duplicate_medical_records(user_id)
                total_deleted += deleted_count
        
        logging.info(f"Автоматическая очистка завершена. Всего удалено {total_deleted} дублирующихся записей")
        
    except Exception as e:
        logging.error(f"Ошибка при автоматической очистке дубликатов: {e}")

def show_user_records(user_id: str):
    """
    Показывает все записи пользователя для анализа
    """
    try:
        logging.info(f"Показываю записи пользователя: {user_id}")
        
        response = supabase.table("doc_medical_records").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not response.data:
            logging.info("Записей не найдено")
            return
        
        logging.info(f"Найдено {len(response.data)} записей:")
        for i, record in enumerate(response.data):
            content_preview = record.get("content", "")[:100] + "..." if len(record.get("content", "")) > 100 else record.get("content", "")
            logging.info(f"  {i+1}. ID: {record.get('id')}, Тип: {record.get('record_type')}, Создано: {record.get('created_at')}")
            logging.info(f"     Содержимое: {content_preview}")
        
    except Exception as e:
        logging.error(f"Ошибка при получении записей пользователя: {e}")

def main():
    """
    Основная функция для тестирования
    """
    logging.info("=== Тестирование функций очистки дубликатов ===")
    
    # Получаем первого пользователя для тестирования
    try:
        response = supabase.table("doc_patient_profiles").select("user_id").limit(1).execute()
        if not response.data:
            logging.error("Пользователи не найдены")
            return
        
        test_user_id = response.data[0]["user_id"]
        logging.info(f"Тестируем на пользователе: {test_user_id}")
        
        # Показываем записи до очистки
        logging.info("\n--- Записи ДО очистки ---")
        show_user_records(test_user_id)
        
        # Очищаем дубликаты
        logging.info("\n--- Очистка дубликатов ---")
        deleted_count = cleanup_duplicate_medical_records(test_user_id)
        
        # Показываем записи после очистки
        logging.info("\n--- Записи ПОСЛЕ очистки ---")
        show_user_records(test_user_id)
        
        logging.info(f"\nРезультат: удалено {deleted_count} дублирующихся записей")
        
    except Exception as e:
        logging.error(f"Ошибка в основной функции: {e}")

if __name__ == "__main__":
    main()
