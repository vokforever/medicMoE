"""
Test script for enhanced medical test extraction system
Проверяет работу улучшенной системы извлечения анализов
"""

import asyncio
import logging
from datetime import datetime
from enhanced_test_extractor import EnhancedTestExtractor
from enhanced_database_cleanup import EnhancedDatabaseCleanup
from config import supabase

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_enhanced_extractor():
    """Тестирует улучшенный экстрактор"""
    print("🧪 Тестирование улучшенного экстрактора анализов...")
    
    # Пример текста с анализами (симуляция распознанного текста с изображения)
    sample_text = """
    Пациент: Иванов Иван Петрович
    Дата анализа: 17.08.2025
    
    1. **Anti-HB core total (анти-HBc):** **
       Результат: отрицательно
       Тест-система: ** Anti-HBc, Abbott
       Оборудование: ** Abbott, Alinity i
    
    2. **Anti-HCV total (анти-HCV):** **
       Результат: отрицательно
       Тест-система: ** Anti-HCV, Abbott
       Оборудование: ** Abbott, Alinity i
    
    3. **IgE (total):** **
       Результат: 45.6 МЕ/мл
       Референсные значения: 0.0-100.0
       Тест-система: IgE, Abbott
       Оборудование: Abbott, Alinity i
    
    4. **Anti-Opisthorchis IgG:** **
       Результат: отрицательно
       Тест-система: Roche, Cobas e602
    """
    
    try:
        extractor = EnhancedTestExtractor()
        
        # Тестируем извлечение из текста
        print("📝 Тестирование извлечения из текста...")
        
        # Создаем временный URL для теста
        temp_url = f"data:text/plain,{sample_text}"
        
        result = await extractor.extract_tests_from_image(temp_url, "Тестовый анализ")
        
        if result.get("success"):
            tests = result.get("structured_tests", [])
            print(f"✅ Успешно извлечено {len(tests)} анализов:")
            
            for i, test in enumerate(tests, 1):
                print(f"\n{i}. {test.get('test_name')}")
                print(f"   Результат: {test.get('result')}")
                print(f"   Норма: {test.get('reference_values')}")
                print(f"   Единицы: {test.get('units')}")
                print(f"   Тест-система: {test.get('test_system')}")
                print(f"   Оборудование: {test.get('equipment')}")
                print(f"   Дата: {test.get('test_date')}")
        else:
            print(f"❌ Ошибка извлечения: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Исключение при тестировании: {e}")

async def test_database_cleanup():
    """Тестирует очистку базы данных"""
    print("\n🧹 Тестирование очистки базы данных...")
    
    try:
        # Создаем тестовые данные с проблемами
        test_user_id = "test-user-123"
        
        # Добавляем тестовые записи с проблемами
        test_records = [
            {
                "user_id": test_user_id,
                "test_name": "Anti-HCV total",
                "result": "**",  # Проблема: символы вместо результата
                "reference_values": "0.0-1.0",
                "units": "МЕ/мл",
                "test_system": "** Anti-HCV, Abbott",
                "equipment": "** Abbott, Alinity i",
                "test_date": None,
                "notes": None,
                "source_record_id": 1
            },
            {
                "user_id": test_user_id,
                "test_name": "IgE total",
                "result": "45.6",
                "reference_values": None,
                "units": "МЕ/мл",
                "test_system": "IgE, Abbott",
                "equipment": "Abbott, Alinity i",
                "test_date": "invalid-date",  # Проблема: невалидная дата
                "notes": None,
                "source_record_id": 2
            },
            {
                "user_id": test_user_id,
                "test_name": "Anti-HCV total",  # Дубликат
                "result": "отрицательно",
                "reference_values": "0.0-1.0",
                "units": "МЕ/мл",
                "test_system": "Anti-HCV, Abbott",
                "equipment": "Abbott, Alinity i",
                "test_date": "2025-08-17",
                "notes": None,
                "source_record_id": 3
            }
        ]
        
        print("📝 Добавление тестовых записей с проблемами...")
        
        # Добавляем тестовые записи
        for record in test_records:
            try:
                supabase.table("doc_structured_test_results").insert(record).execute()
                print(f"✅ Добавлена запись: {record['test_name']}")
            except Exception as e:
                print(f"⚠️ Ошибка добавления записи: {e}")
        
        # Тестируем очистку
        print("\n🧹 Запуск комплексной очистки...")
        cleanup = EnhancedDatabaseCleanup(supabase)
        cleanup_result = await cleanup.cleanup_all_test_results(test_user_id)
        
        if cleanup_result.get("success"):
            print("✅ Очистка завершена успешно!")
            print(f"📊 Результаты: {cleanup_result.get('message')}")
            
            details = cleanup_result.get("details", [])
            for detail in details:
                print(f"   • {detail}")
        else:
            print(f"❌ Ошибка очистки: {cleanup_result.get('message')}")
        
        # Очищаем тестовые данные
        print("\n🗑️ Очистка тестовых данных...")
        try:
            supabase.table("doc_structured_test_results").delete().eq(
                "user_id", test_user_id
            ).execute()
            print("✅ Тестовые данные очищены")
        except Exception as e:
            print(f"⚠️ Ошибка очистки тестовых данных: {e}")
            
    except Exception as e:
        print(f"❌ Исключение при тестировании очистки: {e}")

async def test_sql_data_fix():
    """Тестирует исправление данных из SQL примера"""
    print("\n🔧 Тестирование исправления данных из SQL примера...")
    
    # Пример проблемных данных из SQL
    problematic_data = [
        {
            "id": "10",
            "test_name": "Anti-HB core total (анти-HBc)",
            "result": "**",  # Проблема
            "reference_values": None,
            "units": None,
            "test_system": "** Anti-HBc, Abbott",  # Проблема
            "equipment": None
        },
        {
            "id": "11", 
            "test_name": "- Тест-система",
            "result": "** Anti-HBc, Abbott",  # Проблема: результат в названии
            "reference_values": None,
            "units": None,
            "test_system": "** Anti-HBc, Abbott",  # Проблема
            "equipment": "** Abbott, Alinity i"  # Проблема
        }
    ]
    
    print("📝 Анализ проблемных данных:")
    
    for data in problematic_data:
        test_name = data.get("test_name", "")
        result = data.get("result", "")
        test_system = data.get("test_system", "")
        
        print(f"\n🔍 Анализ: {test_name}")
        print(f"   Текущий результат: '{result}'")
        print(f"   Текущая тест-система: '{test_system}'")
        
        # Определяем проблемы
        issues = []
        if result in ["**", "*"]:
            issues.append("Результат содержит только символы форматирования")
        if test_system in ["**", "*"]:
            issues.append("Тест-система содержит только символы форматирования")
        
        if issues:
            print(f"   ❌ Обнаружены проблемы:")
            for issue in issues:
                print(f"      • {issue}")
            
            # Предлагаем исправления
            print(f"   💡 Предлагаемые исправления:")
            
            # Имитация улучшенного извлечения
            if "Anti-HB core" in test_name:
                print(f"      • Результат: отрицательно")
                print(f"      • Тест-система: Anti-HBc, Abbott")
                print(f"      • Оборудование: Abbott, Alinity i")
        else:
            print(f"   ✅ Проблем не обнаружено")

async def main():
    """Главная функция тестирования"""
    print("🚀 Запуск тестирования улучшенной системы обработки анализов")
    print("=" * 60)
    
    try:
        # Тест 1: Улучшенный экстрактор
        await test_enhanced_extractor()
        
        print("\n" + "=" * 60)
        
        # Тест 2: Очистка базы данных
        await test_database_cleanup()
        
        print("\n" + "=" * 60)
        
        # Тест 3: Исправление SQL данных
        await test_sql_data_fix()
        
        print("\n" + "=" * 60)
        print("✅ Все тесты завершены!")
        
    except Exception as e:
        print(f"❌ Критическая ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())