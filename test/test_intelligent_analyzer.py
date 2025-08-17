#!/usr/bin/env python3
"""
Тест интеллектуального анализатора запросов для медицинского бота.
"""

import asyncio
import sys
import os

# Добавляем путь к основному модулю
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import IntelligentQueryAnalyzer, get_medical_records, generate_user_uuid

async def test_intelligent_analyzer():
    """Тестирует интеллектуальный анализатор запросов."""
    
    print("🧪 Тестирование интеллектуального анализатора запросов...")
    
    # Создаем экземпляр анализатора
    analyzer = IntelligentQueryAnalyzer()
    
    # Тестовые вопросы
    test_questions = [
        "Какие у меня анализы по anti-HEV IgG?",
        "Что показывает мой гемоглобин?",
        "Мой сахар в норме?",
        "Покажи результаты моих анализов",
        "Что такое гепатит?",
        "Какие бывают виды анализов крови?",
        "Как питаться при диабете?",
        "Объясни, что такое аллергия",
        "Какие у меня показатели по IgE?",
        "Покажи мои результаты на паразитов"
    ]
    
    # Тестируем каждый вопрос
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Тест {i}: {question}")
        print("-" * 50)
        
        try:
            # Анализируем вопрос (используем тестовый user_id)
            test_user_id = "test_user_123"
            analysis = await analyzer.analyze_query_type(question, test_user_id)
            
            print(f"✅ Анализ завершен успешно")
            print(f"📊 Является ли вопросом об анализах: {analysis['is_analysis_question']}")
            print(f"🎯 Спрашивает о конкретных показателях: {analysis['is_specific_indicator_question']}")
            print(f"👨‍⚕️ Нужен ли режим врача: {analysis['needs_doctor_mode']}")
            print(f"📋 Есть ли медицинские данные: {analysis['has_medical_data']}")
            
            # Тестируем извлечение контекста
            if analysis['has_medical_data']:
                context = await analyzer.get_relevant_medical_context(question, analysis['medical_records'])
                print(f"📖 Релевантный контекст: {'Да' if context else 'Нет'}")
                if context:
                    print(f"📏 Длина контекста: {len(context)} символов")
            
        except Exception as e:
            print(f"❌ Ошибка при анализе: {e}")
        
        print()
    
    print("🎉 Тестирование завершено!")

async def test_with_real_data():
    """Тестирует анализатор с реальными данными из базы."""
    
    print("\n🔍 Тестирование с реальными данными из базы...")
    
    try:
        # Получаем реальные медицинские записи (если есть)
        real_user_id = "test_real_user"
        medical_records = get_medical_records(real_user_id)
        
        if medical_records:
            print(f"📊 Найдено {len(medical_records)} медицинских записей")
            
            analyzer = IntelligentQueryAnalyzer()
            
            # Тестируем с реальными данными
            question = "Какие у меня анализы по anti-HEV IgG?"
            analysis = await analyzer.analyze_query_type(question, real_user_id)
            
            print(f"✅ Анализ с реальными данными:")
            print(f"📊 Результат анализа: {analysis}")
            
            if analysis['has_medical_data']:
                context = await analyzer.get_relevant_medical_context(question, analysis['medical_records'])
                print(f"📖 Контекст извлечен: {'Да' if context else 'Нет'}")
        
        else:
            print("ℹ️ Реальных медицинских записей не найдено")
            
    except Exception as e:
        print(f"❌ Ошибка при работе с реальными данными: {e}")

def main():
    """Основная функция тестирования."""
    
    print("🚀 Запуск тестов интеллектуального анализатора...")
    
    # Запускаем тесты
    asyncio.run(test_intelligent_analyzer())
    asyncio.run(test_with_real_data())
    
    print("\n✨ Все тесты выполнены!")

if __name__ == "__main__":
    main()
