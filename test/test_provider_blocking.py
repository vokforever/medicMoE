#!/usr/bin/env python3
"""
Тест системы блокировки провайдеров при получении 429 ошибок
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    block_provider_for_day, 
    is_provider_blocked, 
    reset_provider_blocks,
    check_model_availability
)
import asyncio

def test_provider_blocking():
    """Тестирует основные функции блокировки провайдеров"""
    print("🧪 Тестирование системы блокировки провайдеров")
    
    # Тест 1: Блокировка провайдера
    print("\n1. Тест блокировки провайдера")
    provider = "openrouter"
    block_provider_for_day(provider, "Rate limit exceeded (429)")
    
    # Проверяем, что провайдер заблокирован
    if is_provider_blocked(provider):
        print(f"✅ Провайдер {provider} успешно заблокирован")
    else:
        print(f"❌ Провайдер {provider} не заблокирован")
    
    # Тест 2: Проверка доступности заблокированного провайдера
    print("\n2. Тест проверки доступности заблокированного провайдера")
    async def test_availability():
        is_available = await check_model_availability(provider, "test-model")
        if not is_available:
            print(f"✅ Заблокированный провайдер {provider} правильно помечен как недоступный")
        else:
            print(f"❌ Заблокированный провайдер {provider} помечен как доступный")
    
    asyncio.run(test_availability())
    
    # Тест 3: Сброс блокировок
    print("\n3. Тест сброса блокировок")
    reset_provider_blocks()
    
    if not is_provider_blocked(provider):
        print(f"✅ Блокировка провайдера {provider} успешно сброшена")
    else:
        print(f"❌ Блокировка провайдера {provider} не сброшена")
    
    # Тест 4: Проверка доступности после сброса
    print("\n4. Тест проверки доступности после сброса блокировки")
    async def test_availability_after_reset():
        is_available = await check_model_availability(provider, "test-model")
        print(f"Провайдер {provider} доступен после сброса: {is_available}")
    
    asyncio.run(test_availability_after_reset())
    
    print("\n🎉 Все тесты завершены!")

if __name__ == "__main__":
    test_provider_blocking()
