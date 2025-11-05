"""
Тестирование новой Low-Code архитектуры обработки медицинских документов
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from core.universal_processor import universal_processor, ProcessingResult
from core.validators import medical_validator
from core.monitoring import processing_monitor
from config.medical_config import medical_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ArchitectureTester:
    """Тестировщик новой архитектуры"""
    
    def __init__(self):
        self.processor = universal_processor
        self.validator = medical_validator
        self.monitor = processing_monitor
        self.config = medical_config
    
    async def test_complete_architecture(self):
        """Комплексный тест архитектуры"""
        logger.info("🚀 Начинаю тестирование новой Low-Code архитектуры")
        
        test_results = {
            "config_test": await self.test_config_system(),
            "validation_test": await self.test_validation_system(),
            "monitoring_test": await self.test_monitoring_system(),
            "processor_test": await self.test_universal_processor(),
            "integration_test": await self.test_integration()
        }
        
        # Формируем отчет
        await self.generate_test_report(test_results)
        
        return test_results
    
    async def test_config_system(self):
        """Тестирование конфигурационной системы"""
        logger.info("📋 Тестирование конфигурационной системы")
        
        try:
            # Проверка загрузки конфигурации
            stats = self.config.get_statistics()
            assert stats["version"] is not None
            assert stats["categories_count"] > 0
            assert stats["patterns_count"] > 0
            
            # Проверка категорий
            categories = self.config.get_categories()
            assert len(categories) > 0
            assert "biochemical" in categories
            
            # Проверка паттернов
            test_patterns = self.config.get_patterns("test_name")
            assert len(test_patterns) > 0
            
            # Проверка LLM функций
            llm_functions = self.config.get_llm_functions()
            assert len(llm_functions) > 0
            assert "extract_medical_data" in llm_functions
            
            # Проверка валидации конфигурации
            validation = self.config.validate_config()
            assert validation["valid"] is True
            
            logger.info("✅ Конфигурационная система работает корректно")
            return {"status": "success", "stats": stats, "validation": validation}
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования конфигурационной системы: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_validation_system(self):
        """Тестирование системы валидации"""
        logger.info("🔍 Тестирование системы валидации")
        
        try:
            # Тест валидации названия анализа
            test_name_result = self.validator.validate_test_name("АЛТ")
            assert test_name_result.is_valid
            assert test_name_result.confidence > 0
            
            # Тест валидации результата
            result_result = self.validator.validate_result("45.5")
            assert result_result.is_valid
            assert result_result.confidence > 0
            
            # Тест валидации даты
            date_result = self.validator.validate_date("01.01.2024")
            assert date_result.is_valid
            assert "01.01.2024" in date_result.cleaned_data["date"]
            
            # Тест валидации полного медицинского теста
            test_data = {
                "test_name": "Билирубин общий",
                "result": "15.2",
                "reference_values": "5-21",
                "units": "мкмоль/л"
            }
            test_result = self.validator.validate_medical_test(test_data)
            assert test_result.is_valid
            assert test_result.confidence > 0
            
            # Тест пакетной валидации
            batch_results = self.validator.batch_validate([test_data])
            assert len(batch_results) == 1
            assert batch_results[0].is_valid
            
            logger.info("✅ Система валидации работает корректно")
            return {"status": "success", "validation_results": len(batch_results)}
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования системы валидации: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_monitoring_system(self):
        """Тестирование системы мониторинга"""
        logger.info("📊 Тестирование системы мониторинга")
        
        try:
            # Проверка получения статистики
            stats = self.monitor.get_statistics()
            assert "total_processed" in stats
            assert "success_rate" in stats
            
            # Тест начала обработки
            session_id = self.monitor.start_processing("test")
            assert session_id is not None
            
            # Тест завершения обработки
            import time
            start_time = time.time()
            metrics = self.monitor.end_processing(
                session_id=session_id,
                document_type="test",
                start_time=start_time,
                success=True,
                extraction_method="test_method",
                tests_count=5,
                confidence_score=0.85
            )
            assert metrics.success is True
            assert metrics.document_type == "test"
            
            # Тест проверки здоровья
            from core.monitoring import health_checker
            health = health_checker.check_health()
            assert "overall_status" in health
            assert "checks" in health
            
            logger.info("✅ Система мониторинга работает корректно")
            return {"status": "success", "health_status": health["overall_status"]}
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования системы мониторинга: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_universal_processor(self):
        """Тестирование универсального процессора"""
        logger.info("🔧 Тестирование универсального процессора")
        
        try:
            # Проверка поддерживаемых типов
            supported_types = self.processor.get_supported_types()
            assert len(supported_types) > 0
            assert "pdf" in supported_types
            assert "image" in supported_types
            
            # Текстовый тест обработки (без реальных файлов)
            test_text = """
            Анализ крови
            Билирубин общий: 15.2 мкмоль/л (норма: 5-21)
            АЛТ: 45.5 Ед/л (норма: 7-55)
            АСТ: 38.2 Ед/л (норма: 5-40)
            """
            
            # Создаем тестовый результат
            test_result = ProcessingResult(
                success=True,
                data={
                    "medical_tests": [
                        {
                            "test_name": "Билирубин общий",
                            "result": "15.2",
                            "reference_values": "5-21",
                            "units": "мкмоль/л"
                        }
                    ],
                    "summary": "Тестовый анализ"
                },
                confidence=0.9,
                extraction_method="test"
            )
            
            # Тест форматирования ответа
            formatted = self.processor.response_formatter.format_response(test_result)
            assert len(formatted) > 0
            assert "Билирубин" in formatted
            
            # Тест получения статистики
            stats = self.processor.get_processing_statistics()
            assert isinstance(stats, dict)
            
            # Тест проверки здоровья
            health = self.processor.get_health_status()
            assert "overall_status" in health
            
            logger.info("✅ Универсальный процессор работает корректно")
            return {"status": "success", "supported_types": len(supported_types)}
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования универсального процессора: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_integration(self):
        """Интеграционный тест"""
        logger.info("🔗 Тестирование интеграции компонентов")
        
        try:
            # Текст для анализа
            sample_text = """
            Лабораторные исследования
            Дата: 01.11.2024
            
            Биохимический анализ:
            Общий белок: 75 г/л (норма: 65-85)
            Глюкоза: 5.2 ммоль/л (норма: 3.9-5.9)
            Мочевина: 6.1 ммоль/л (норма: 2.8-8.2)
            
            Гормональный профиль:
            ТТГ: 2.1 мЕд/л (норма: 0.4-4.0)
            """
            
            # Тест процессора данных
            data_processor = self.processor.data_processor
            
            # Тест извлечения с помощью регулярных выражений
            result = await data_processor._extract_with_regex(sample_text)
            assert result.success
            assert len(result.data.get("medical_tests", [])) > 0
            
            # Тест валидации извлеченных данных
            validation = data_processor._validate_extracted_data(result.data)
            assert validation is not None
            
            # Тест категоризации
            tests = result.data.get("medical_tests", [])
            categorized = data_processor._categorize_tests(tests)
            assert len(categorized) > 0
            
            logger.info("✅ Интеграционный тест пройден успешно")
            return {"status": "success", "extracted_tests": len(categorized)}
            
        except Exception as e:
            logger.error(f"❌ Ошибка интеграционного теста: {e}")
            return {"status": "error", "error": str(e)}
    
    async def generate_test_report(self, test_results):
        """Генерация отчета о тестировании"""
        logger.info("📋 Генерация отчета о тестировании")
        
        try:
            total_tests = len(test_results)
            successful_tests = sum(1 for result in test_results.values() if result.get("status") == "success")
            failed_tests = total_tests - successful_tests
            
            print("\n" + "="*60)
            print("🧪 ОТЧЕТ О ТЕСТИРОВАНИИ НОВОЙ ARCHITECTURE")
            print("="*60)
            
            print(f"📊 Общая статистика:")
            print(f"   • Всего тестов: {total_tests}")
            print(f"   • Успешно: {successful_tests}")
            print(f"   • С ошибками: {failed_tests}")
            print(f"   • Успешность: {successful_tests/total_tests*100:.1f}%")
            
            print(f"\n📋 Детальные результаты:")
            
            for test_name, result in test_results.items():
                status_icon = "✅" if result.get("status") == "success" else "❌"
                print(f"   {status_icon} {test_name}: {result.get('status', 'unknown')}")
                
                if result.get("error"):
                    print(f"      Ошибка: {result['error']}")
                
                # Дополнительная информация
                if test_name == "config_test" and result.get("stats"):
                    stats = result["stats"]
                    print(f"      Категорий: {stats['categories_count']}, Паттернов: {stats['patterns_count']}")
                
                elif test_name == "validation_test" and result.get("validation_results"):
                    print(f"      Проверено валидаций: {result['validation_results']}")
                
                elif test_name == "monitoring_test" and result.get("health_status"):
                    print(f"      Статус здоровья: {result['health_status']}")
                
                elif test_name == "processor_test" and result.get("supported_types"):
                    print(f"      Поддерживаемых типов: {result['supported_types']}")
                
                elif test_name == "integration_test" and result.get("extracted_tests"):
                    print(f"      Извлечено тестов: {result['extracted_tests']}")
            
            print(f"\n🔧 Техническая информация:")
            
            # Статистика системы
            stats = self.monitor.get_statistics()
            print(f"   • Обработано документов: {stats['total_processed']}")
            print(f"   • Успешность обработки: {stats['success_rate']:.1f}%")
            
            # Статус здоровья
            health = self.processor.get_health_status()
            print(f"   • Здоровье системы: {health['overall_status']}")
            
            # Конфигурация
            config_stats = self.config.get_statistics()
            print(f"   • Версия конфигурации: {config_stats['version']}")
            print(f"   • Категорий: {config_stats['categories_count']}")
            print(f"   • LLM функций: {config_stats['llm_functions_count']}")
            
            print("\n" + "="*60)
            
            if failed_tests == 0:
                print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Новая архитектура готова к использованию.")
            else:
                print(f"⚠️ {failed_tests} тест(ов) не пройдены. Требуется доработка.")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")

async def main():
    """Главная функция тестирования"""
    logger.info("🚀 Запуск тестирования новой Low-Code архитектуры")
    
    try:
        # Инициализация и тестирование
        tester = ArchitectureTester()
        results = await tester.test_complete_architecture()
        
        # Возвращаем общий статус
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        total_count = len(results)
        
        if success_count == total_count:
            logger.info("🎉 Все тесты пройдены! Новая архитектура готова к развертыванию.")
            return True
        else:
            logger.warning(f"⚠️ {total_count - success_count} из {total_count} тестов не пройдены.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при тестировании: {e}")
        return False

if __name__ == "__main__":
    # Запуск тестирования
    success = asyncio.run(main())
    
    # Выход с соответствующим кодом
    sys.exit(0 if success else 1)
