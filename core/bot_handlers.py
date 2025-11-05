"""
Обработчики для Telegram бота с использованием новой Low-Code архитектуры
"""
import logging
import asyncio
from typing import Dict, Any, Optional
from aiogram import types, F
from aiogram.fsm.context import FSMContext

from core.universal_processor import universal_processor, ProcessingResult
from database import save_medical_record, generate_user_uuid
from utils import safe_send_message
from keyboards import get_feedback_keyboard

logger = logging.getLogger(__name__)

class BotHandlers:
    """Класс обработчиков для бота"""
    
    def __init__(self, bot):
        self.bot = bot
        self.processor = universal_processor
    
    async def handle_photo_message(self, message: types.Message, state: FSMContext):
        """Универсальный обработчик фото с новой архитектурой"""
        try:
            user_id = generate_user_uuid(message.from_user.id)
            logger.info(f"Получено фото от пользователя {message.from_user.id}")
            
            # Отправляем сообщение о начале обработки
            processing_msg = await message.answer("🔍 Анализирую изображение с новой системой...")
            
            try:
                # Получаем URL файла
                photo = message.photo[-1]
                file_info = await self.bot.get_file(photo.file_id)
                file_url = f"https://api.telegram.org/file/bot{self.bot.token}/{file_info.file_path}"
                
                # Обрабатываем документ с новым универсальным процессором
                result = await self.processor.process_document(file_url, "image")
                
                # Обрабатываем результат
                await self._handle_processing_result(
                    result, user_id, processing_msg, message, "image_analysis"
                )
                
            except Exception as e:
                logger.error(f"Ошибка при обработке фото: {e}")
                await processing_msg.edit_text(
                    "😔 Не удалось обработать изображение. Попробуйте еще раз."
                )
                
        except Exception as e:
            logger.error(f"Ошибка при обработке фото: {e}")
            await message.answer("😔 Произошла ошибка. Попробуйте еще раз.")
    
    async def handle_document_message(self, message: types.Message, state: FSMContext):
        """Универсальный обработчик документов с новой архитектурой"""
        try:
            user_id = generate_user_uuid(message.from_user.id)
            document = message.document
            
            logger.info(f"Получен документ от пользователя {message.from_user.id}: {document.file_name}")
            
            # Проверяем, что это PDF
            if not document.file_name.lower().endswith('.pdf'):
                await message.answer("❌ Поддерживаются только PDF файлы. Пожалуйста, загрузите PDF документ.")
                return
            
            # Отправляем сообщение о начале обработки
            processing_msg = await message.answer("📄 Обрабатываю PDF документ с новой системой...")
            
            try:
                # Получаем URL файла
                file_info = await self.bot.get_file(document.file_id)
                file_url = f"https://api.telegram.org/file/bot{self.bot.token}/{file_info.file_path}"
                
                # Обрабатываем документ с новым универсальным процессором
                result = await self.processor.process_document(file_url, "pdf")
                
                # Обрабатываем результат
                await self._handle_processing_result(
                    result, user_id, processing_msg, message, "pdf_analysis"
                )
                
            except Exception as e:
                logger.error(f"Ошибка при обработке PDF: {e}")
                await processing_msg.edit_text(
                    "😔 Не удалось обработать PDF документ. Попробуйте еще раз."
                )
                
        except Exception as e:
            logger.error(f"Ошибка при обработке документа: {e}")
            await message.answer("😔 Произошла ошибка. Попробуйте еще раз.")
    
    async def _handle_processing_result(self, 
                                   result: ProcessingResult, 
                                   user_id: str, 
                                   processing_msg: types.Message,
                                   original_message: types.Message,
                                   record_type: str):
        """Обработать результат обработки документа"""
        try:
            if result.success:
                # Сохраняем в базу данных
                formatted_response = result.data.get("formatted_response", "")
                await save_medical_record(
                    user_id, 
                    record_type, 
                    formatted_response, 
                    result.extraction_method
                )
                
                # Сохраняем структурированные данные
                medical_tests = result.data.get("medical_tests", [])
                if medical_tests:
                    await self._save_structured_tests(user_id, medical_tests)
                
                # Отправляем ответ с использованием безопасной функции
                await safe_send_message(
                    original_message,
                    formatted_response,
                    reply_markup=get_feedback_keyboard()
                )
                
                # Удаляем сообщение о обработке
                await processing_msg.delete()
                
                # Логируем успешную обработку
                logger.info(
                    f"Документ успешно обработан: {result.extraction_method}, "
                    f"тестов: {len(medical_tests)}, уверенность: {result.confidence:.2f}"
                )
                
            else:
                # Обработка ошибки
                error_message = self._format_error_message(result)
                await processing_msg.edit_text(error_message)
                
                # Сохраняем информацию об ошибке
                await save_medical_record(
                    user_id, 
                    f"{record_type}_error", 
                    f"Ошибка обработки: {result.error}", 
                    result.extraction_method
                )
                
        except Exception as e:
            logger.error(f"Ошибка при обработке результата: {e}")
            await processing_msg.edit_text(
                "😔 Произошла ошибка при формировании ответа."
            )
    
    async def _save_structured_tests(self, user_id: str, tests: list):
        """Сохранение структурированных тестов"""
        try:
            from database import supabase
            from datetime import datetime
            
            for test in tests:
                test_data = {
                    "user_id": user_id,
                    "test_name": test.get("test_name", ""),
                    "result": test.get("result", ""),
                    "reference_values": test.get("reference_values", ""),
                    "units": test.get("units", ""),
                    "category": test.get("category", ""),
                    "abnormal": test.get("abnormal", False),
                    "created_at": datetime.now().isoformat()
                }
                
                supabase.table("doc_structured_test_results").insert(test_data).execute()
                
            logger.info(f"Сохранено {len(tests)} структурированных тестов")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения структурированных тестов: {e}")
    
    def _format_error_message(self, result: ProcessingResult) -> str:
        """Форматировать сообщение об ошибке"""
        base_error = "❌ Не удалось обработать документ"
        
        if "Неподдерживаемый тип документа" in str(result.error):
            return "❌ Неподдерживаемый тип документа"
        elif "Не удалось извлечь текст" in str(result.error):
            return "❌ Не удалось извлечь текст из документа. Попробуйте сделать фото более четким."
        elif "Function calling" in str(result.extraction_method):
            return "❌ Ошибка при извлечении данных с помощью ИИ. Попробуйте еще раз."
        elif result.confidence < 0.3:
            return f"❌ Низкое качество извлечения данных (уверенность: {result.confidence:.1f}). Попробуйте более четкий документ."
        else:
            return f"{base_error}: {result.error or 'Неизвестная ошибка'}"
    
    async def get_system_status(self, message: types.Message):
        """Получить статус системы"""
        try:
            # Получаем статистику обработки
            stats = self.processor.get_processing_statistics()
            
            # Получаем статус здоровья
            health = self.processor.get_health_status()
            
            # Формируем ответ
            response = "🔧 **Статус системы:**\n\n"
            
            # Общая статистика
            response += "📊 **Статистика обработки:**\n"
            response += f"• Всего обработано: {stats['total_processed']}\n"
            response += f"• Успешно: {stats['successful_processed']}\n"
            response += f"• С ошибками: {stats['failed_processed']}\n"
            response += f"• Успешность: {stats['success_rate']:.1f}%\n"
            response += f"• Среднее время: {stats['average_processing_time']:.2f}с\n"
            response += f"• Аптайм: {stats['uptime_minutes']:.1f} мин\n\n"
            
            # Статус здоровья
            health_status = health['overall_status']
            status_icon = "✅" if health_status == "healthy" else "⚠️" if health_status == "warning" else "🚨"
            
            response += f"{status_icon} **Здоровье системы:** {health_status.upper()}\n"
            
            for check_name, check_result in health['checks'].items():
                check_icon = "✅" if check_result['status'] == "healthy" else "⚠️" if check_result['status'] == "warning" else "🚨"
                response += f"• {check_icon} {check_result['message']}\n"
            
            # Метрики по типам документов
            if stats['by_type']:
                response += "\n📄 **По типам документов:**\n"
                for doc_type, type_stats in stats['by_type'].items():
                    success_rate = type_stats['successful'] / type_stats['total'] * 100 if type_stats['total'] > 0 else 0
                    response += f"• {doc_type}: {type_stats['total']} ({success_rate:.1f}% усп.)\n"
            
            # Метрики по методам извлечения
            if stats['by_method']:
                response += "\n🔬 **По методам извлечения:**\n"
                for method, method_stats in stats['by_method'].items():
                    success_rate = method_stats['successful'] / method_stats['total'] * 100 if method_stats['total'] > 0 else 0
                    avg_conf = method_stats.get('avg_confidence', 0)
                    response += f"• {method}: {method_stats['total']} ({success_rate:.1f}% усп., {avg_conf:.2f} увер.)\n"
            
            await message.answer(response, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Ошибка получения статуса системы: {e}")
            await message.answer("😔 Не удалось получить статус системы.")
    
    async def get_config_info(self, message: types.Message):
        """Получить информацию о конфигурации"""
        try:
            from config.medical_config import medical_config
            
            # Получаем статистику конфигурации
            config_stats = medical_config.get_statistics()
            
            # Валидация конфигурации
            validation = medical_config.validate_config()
            
            response = "⚙️ **Конфигурация системы:**\n\n"
            
            # Основная информация
            response += f"📋 **Версия:** {config_stats['version']}\n"
            response += f"📂 **Категорий:** {config_stats['categories_count']}\n"
            response += f"🔍 **Паттернов:** {config_stats['patterns_count']}\n"
            response += f"🤖 **LLM функций:** {config_stats['llm_functions_count']}\n\n"
            
            # Статус валидации
            validation_icon = "✅" if validation['valid'] else "❌"
            response += f"{validation_icon} **Валидация конфигурации:** {validation['valid']}\n"
            
            if validation['warnings']:
                response += "⚠️ **Предупреждения:**\n"
                for warning in validation['warnings'][:3]:  # Показываем только первые 3
                    response += f"• {warning}\n"
            
            if validation['issues']:
                response += "🚨 **Проблемы:**\n"
                for issue in validation['issues'][:3]:  # Показываем только первые 3
                    response += f"• {issue}\n"
            
            # Категории
            categories = medical_config.get_categories()
            if categories:
                response += "\n📊 **Доступные категории:**\n"
                for cat_key, cat_config in list(categories.items())[:5]:  # Показываем только первые 5
                    response += f"• {cat_config['name']} (приоритет: {cat_config.get('priority', 'N/A')})\n"
            
            await message.answer(response, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о конфигурации: {e}")
            await message.answer("😔 Не удалось получить информацию о конфигурации.")

# Фабрика для создания обработчиков
def create_bot_handlers(bot):
    """Создать экземпляр обработчиков"""
    return BotHandlers(bot)
