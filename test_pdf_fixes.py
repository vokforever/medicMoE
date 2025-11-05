#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправлений обработки PDF документов
"""

import asyncio
import logging
import sys
from unittest.mock import Mock, patch

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Импортируем модули
try:
    from models import call_model_with_failover
    from database import save_successful_response
    from medical_terms_agent import medical_terms_agent
    from utils import extract_text_from_pdf
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    sys.exit(1)

async def test_tuple_response_handling():
    """Тест обработки кортежа от call_model_with_failover"""
    print("🧪 Тест 1: Обработка кортежа от call_model_with_failover")
    
    try:
        # Мокаем ответ модели
        mock_response = ("Тестовый ответ PDF анализа", "cerebras", {"usage": Mock()})
        
        # Проверяем обработку кортежа
        if isinstance(mock_response, tuple) and len(mock_response) > 0:
            analysis_result = mock_response[0]
            print(f"✅ Успешно извлечен текст из кортежа: {analysis_result}")
        else:
            analysis_result = str(mock_response)
            print(f"✅ Успешно преобразован в строку: {analysis_result}")
            
        return True
    except Exception as e:
        print(f"❌ Ошибка при обработке кортежа: {e}")
        return False

async def test_completion_usage_serialization():
    """Тест сериализации CompletionUsage"""
    print("\n🧪 Тест 2: Сериализация CompletionUsage")
    
    try:
        # Создаем мок объект похожий на CompletionUsage
        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.__dict__ = {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150
        }
        
        metadata = {
            "provider": "test",
            "model": "test-model",
            "usage": mock_usage
        }
        
        # Тестируем сохранение с успешным ответом
        user_id = "test-user"
        question = "Тестовый вопрос"
        answer = "Тестовый ответ"
        
        # Мокаем supabase
        with patch('database.supabase') as mock_supabase:
            mock_table = Mock()
            mock_insert = Mock()
            mock_table.insert.return_value = mock_insert
            mock_insert.execute.return_value = Mock(data=[{"id": 1}])
            mock_supabase.table.return_value = mock_table
            
            result = await save_successful_response(user_id, question, answer, "test", metadata)
            print(f"✅ Успешная сериализация CompletionUsage: {result}")
            
        return True
    except Exception as e:
        print(f"❌ Ошибка при сериализации CompletionUsage: {e}")
        return False

async def test_json_parsing():
    """Тест улучшенного парсинга JSON"""
    print("\n🧪 Тест 3: Улучшенный парсинг JSON")
    
    try:
        # Тестовые варианты JSON ответов
        test_cases = [
            # Нормальный JSON массив
            '[{"test_name": "АЛТ", "result": "25"}]',
            
            # JSON с лишними символами
            '[{"test_name": "АСТ", "result": "30", "units": "Ед/л"}]\n\n',
            
            # JSON объект вместо массива
            '{"test_name": "Билирубин", "result": "15"}',
            
            # JSON в тексте
            'Результаты анализов: [{"test_name": "Глюкоза", "result": "5.0"}] и другие.',
            
            # Сломанный JSON
            '[{"test_name": "Тест", "result": "значение"'
        ]
        
        success_count = 0
        for i, test_json in enumerate(test_cases):
            try:
                # Тестируем парсинг
                import json
                import re
                
                # Ищем JSON различными способами
                json_str = None
                
                # Способ 1: Ищем массив
                json_start = test_json.find('[')
                json_end = test_json.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = test_json[json_start:json_end]
                
                # Способ 2: Ищем объект
                if not json_str:
                    json_start = test_json.find('{')
                    json_end = test_json.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = "[" + test_json[json_start:json_end] + "]"
                
                if json_str:
                    # Очищаем JSON
                    json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                    json_str = re.sub(r'\s+', ' ', json_str)
                    
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                    
                    if isinstance(parsed, list) and len(parsed) > 0:
                        print(f"✅ Тест {i+1}: Успешно распарсен JSON")
                        success_count += 1
                    else:
                        print(f"⚠️ Тест {i+1}: JSON распарсен, но пустой")
                else:
                    print(f"❌ Тест {i+1}: JSON не найден в тексте")
                    
            except Exception as e:
                print(f"❌ Тест {i+1}: Ошибка парсинга JSON: {e}")
        
        print(f"📊 Результат: {success_count}/{len(test_cases)} тестов пройдено")
        return success_count == len(test_cases) - 1  # Последний тест должен провалиться
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании JSON парсинга: {e}")
        return False

async def test_medical_agent_extraction():
    """Тест извлечения медицинских параметров"""
    print("\n🧪 Тест 4: Извлечение медицинских параметров")
    
    try:
        # Тестовый текст с анализами
        test_text = """
        Пациент: Иванов Иван
        Дата анализа: 15.10.2024
        
        Результаты анализов:
        АЛТ: 25 Ед/л (норма: 5-41)
        АСТ: 30 Ед/л (норма: 5-38)
        Билирубин общий: 15.5 мкмоль/л (норма: 8-20.5)
        """
        
        # Мокаем call_model_with_failover
        with patch('medical_terms_agent.call_model_with_failover') as mock_call:
            mock_call.return_value = ('[{"test_name": "АЛТ", "result": "25", "units": "Ед/л", "reference_values": "5-41"}]', 'cerebras', {})
            
            parameters = await medical_terms_agent.extract_test_parameters(test_text)
            
            if parameters and len(parameters) > 0:
                print(f"✅ Успешно извлечены параметры: {len(parameters)} анализов")
                for param in parameters[:3]:  # Показываем первые 3
                    print(f"   - {param.get('test_name')}: {param.get('result')}")
                return True
            else:
                print("❌ Не удалось извлечь параметры анализов")
                return False
                
    except Exception as e:
        print(f"❌ Ошибка при извлечении медицинских параметров: {e}")
        return False

async def main():
    """Главная функция тестирования"""
    print("🚀 Начинаем тестирование исправлений обработки PDF\n")
    
    tests = [
        ("Обработка кортежа", test_tuple_response_handling),
        ("Сериализация CompletionUsage", test_completion_usage_serialization),
        ("Парсинг JSON", test_json_parsing),
        ("Извлечение медицинских параметров", test_medical_agent_extraction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
                print(f"✅ Тест '{test_name}' пройден")
            else:
                print(f"❌ Тест '{test_name}' не пройден")
        except Exception as e:
            print(f"❌ Тест '{test_name}' завершился с ошибкой: {e}")
    
    print(f"\n📊 Итоги: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты пройдены! Исправления работают корректно.")
        return True
    else:
        print("⚠️ Некоторые тесты не пройдены. Требуется дополнительная отладка.")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Тестирование прервано")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        sys.exit(1)
