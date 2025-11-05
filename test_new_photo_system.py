#!/usr/bin/env python3
"""
Тестирование новой упрощенной системы обработки фото
"""

import asyncio
import logging
from datetime import datetime

# Импорты наших модулей
from photo_processor import SimplePhotoProcessor
from utils import safe_send_message, escape_markdown_improved

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    handlers=[
        logging.FileHandler('test_new_photo_system.log'),
        logging.StreamHandler()
    ]
)

class MockBot:
    """Мок бот для тестирования"""
    
    def __init__(self, token: str):
        self.token = token
    
    async def get_file(self, file_id: str):
        """Мок получение файла"""
        class MockFileInfo:
            file_path = f"test_photos/test_{file_id}.jpg"
        
        return MockFileInfo()
    
    async def send_message(self, chat_id: int, text: str, **kwargs):
        """Мок отправка сообщения"""
        print(f"📨 MESSAGE to {chat_id}: {text[:100]}...")
        return {"message_id": f"msg_{chat_id}_{datetime.now().timestamp()}"}

class MockState:
    """Мок состояние"""
    async def clear(self):
        """Мок очистка состояния"""
        pass

class MockUser:
    """Мок пользователь"""
    def __init__(self, id: int):
        self.id = id

class TestPhotoSystem:
    """Тестирование новой системы обработки фото"""
    
    def __init__(self):
        self.bot = MockBot("test_token")
        self.processor = SimplePhotoProcessor()
    
    async def test_photo_processing(self, image_url: str, test_name: str):
        """Тестирование обработки фото"""
        print(f"\n🧪 ТЕСТ: {test_name}")
        print(f"🔗 URL: {image_url}")
        
        try:
            # Тестируем обработку фото
            result = await self.processor.process_photo(image_url)
            
            print(f"✅ Результат: {result['success']}")
            
            if result['success']:
                print(f"📊 Длина ответа: {len(result['response'])} символов")
                print(f"🔬 Структурированных данных: {len(result.get('structured_data', []))}")
                
                # Тестируем сохранение
                await self._test_save_to_database(result)
                
                # Тестируем отправку
                await self._test_message_sending(result)
                
                return True
            else:
                print(f"❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")
                return False
                
        except Exception as e:
            print(f"💥 Исключение в тесте: {e}")
            return False
    
    async def _test_save_to_database(self, result):
        """Тестирование сохранения в базу"""
        print(f"💾 ТЕСТ: Сохранение в базу данных...")
        # В реальной системе здесь будет сохранение в Supabase
        pass
    
    async def _test_message_sending(self, result):
        """Тестирование отправки сообщения"""
        print(f"📤 ТЕСТ: Отправка результата пользователю...")
        
        try:
            # Тестируем безопасную отправку
            mock_state = MockState()
            mock_message = types.Message()
            mock_message.from_user = MockUser(12345)
            
            await safe_send_message(mock_message, result["response"])
            print("✅ Безопасная отправка прошла успешно")
            
        except Exception as e:
            print(f"💥 Ошибка безопасной отправки: {e}")
    
    async def test_various_scenarios(self):
        """Тестирование различных сценариев"""
        print("\n" + "="*50)
        print("🧪 НАЧАЛО ТЕСТИРОВАНИЯ НОВОЙ СИСТЕМЫ ОБРАБОТКИ ФОТО")
        print("="*50)
        
        # Тест 1: Успешное фото
        await self.test_photo_processing(
            "https://example.com/photo1.jpg",
            "Успешная обработка фото"
        )
        
        # Тест 2: Пустое фото
        await self.test_photo_processing(
            "https://example.com/photo_empty.jpg",
            "Обработка пустого фото"
        )
        
        # Тест 3: Фото с ошибкой
        await self.test_photo_processing(
            "https://example.com/photo_error.jpg",
            "Обработка фото с ошибкой"
        )

async def main():
    """Основная функция тестирования"""
    tester = TestPhotoSystem()
    await tester.test_various_scenarios()
    
    print("\n" + "="*50)
    print("🧪 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
