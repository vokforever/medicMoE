"""
Универсальный процессор медицинских документов с архитектурой Low-Code
"""
import json
import logging
import asyncio
import time
import re
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from config.medical_config import medical_config
from core.validators import medical_validator, ValidationResult
from core.monitoring import processing_monitor, ProcessingMetrics
from models import call_model_with_failover
from utils import extract_text_from_pdf, analyze_image

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Результат обработки документа"""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    extraction_method: str = "unknown"
    processing_time: float = 0.0

class DocumentExtractor(ABC):
    """Абстрактный класс для извлечения текста"""
    
    @abstractmethod
    async def extract_text(self, source: str) -> Optional[str]:
        """Извлечь текст из источника"""
        pass

class PDFExtractor(DocumentExtractor):
    """Извлечение текста из PDF"""
    
    async def extract_text(self, source: str) -> Optional[str]:
        try:
            logger.info(f"Начало извлечения текста из PDF: {source}")
            text = await extract_text_from_pdf(source)
            
            if text and len(text.strip()) > 10:
                logger.info(f"Успешно извлечено {len(text)} символов из PDF")
                return text
            else:
                logger.warning("PDF не содержит текста или текст слишком короткий")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из PDF: {e}")
            return None

class ImageExtractor(DocumentExtractor):
    """Извлечение текста из изображения"""
    
    async def extract_text(self, source: str) -> Optional[str]:
        try:
            logger.info(f"Начало извлечения текста из изображения: {source}")
            text = await analyze_image(source, "Извлеки весь текст с этого медицинского документа. Верни только текст без комментариев.")
            
            if text and len(text.strip()) > 10:
                logger.info(f"Успешно извлечено {len(text)} символов из изображения")
                return text
            else:
                logger.warning("Изображение не содержит текста или текст слишком короткий")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из изображения: {e}")
            return None

class DataProcessor:
    """Процессор данных с использованием LLM"""
    
    def __init__(self, config=None):
        self.config = config or medical_config
        self.validator = medical_validator
        self.processing_config = self.config.get_processing_config()
    
    async def extract_structured_data(self, text: str, document_type: str) -> ProcessingResult:
        """Извлечение структурированных данных с помощью LLM"""
        start_time = time.time()
        
        try:
            # Проверяем лимиты
            max_length = self.processing_config.get("max_text_length", 50000)
            if len(text) > max_length:
                text = text[:max_length] + "..."
                logger.warning(f"Текст обрезан до {max_length} символов")
            
            # Основной метод: Function Calling
            result = await self._extract_with_function_calling(text, document_type)
            
            if result.success:
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                return result
            
            # Fallback метод: Регулярные выражения
            logger.info("Function calling не удался, используем fallback с регулярными выражениями")
            result = await self._extract_with_regex(text)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Ошибка извлечения структурированных данных: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e),
                processing_time=processing_time,
                extraction_method="llm_error"
            )
    
    async def _extract_with_function_calling(self, text: str, document_type: str) -> ProcessingResult:
        """Извлечение данных с использованием function calling"""
        try:
            functions = self.config.get_llm_functions()
            
            # Формируем системный промпт
            system_prompt = f"""Ты - медицинский аналитик. Проанализируй медицинский документ и извлеки все релевантные данные.

ДОКУМЕНТ ТИПА: {document_type}

ТЕКУЩАЯ ДАТА: {time.strftime('%d.%m.%Y')}

ВАЖНО:
- Используй точные данные из документа
- Не придумывай данные, которых нет в тексте
- Определяй категорию каждого анализа
- Проверяй референсные значения
- Указывай единицы измерения

Используй функцию extract_medical_data для извлечения структурированных данных."""
            
            # Формируем пользовательский промпт
            user_prompt = f"""Проанализируй этот медицинский документ и извлеки всю информацию:

{text}

Используй функцию extract_medical_data для извлечения структурированных данных."""
            
            # Вызываем модель с function calling
            response = await call_model_with_failover(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model_type="text",
                functions=list(functions.values()),
                function_call={"name": "extract_medical_data"}
            )
            
            if response and isinstance(response, tuple):
                content = response[0]
                metadata = response[2] if len(response) > 2 else {}
                
                # Проверяем, является ли ответ function call
                if isinstance(content, dict) and "function_call" in content:
                    function_args = json.loads(content["function_call"]["arguments"])
                    
                    # Валидация извлеченных данных
                    validation_result = self._validate_extracted_data(function_args)
                    
                    if validation_result.is_valid:
                        return ProcessingResult(
                            success=True,
                            data=function_args,
                            metadata={
                                "extraction_method": "function_calling",
                                "model_provider": metadata.get("provider"),
                                "model_name": metadata.get("model"),
                                "tokens_used": metadata.get("usage", {}).get("total_tokens"),
                                "validation": validation_result.cleaned_data
                            },
                            confidence=validation_result.confidence,
                            extraction_method="function_calling"
                        )
                    else:
                        logger.warning(f"Данные не прошли валидацию: {validation_result.errors}")
                        # Продолжаем с данными, но с предупреждениями
                        return ProcessingResult(
                            success=True,
                            data=function_args,
                            metadata={
                                "extraction_method": "function_calling",
                                "validation_warnings": validation_result.warnings,
                                "validation_errors": validation_result.errors
                            },
                            confidence=validation_result.confidence * 0.5,  # Снижаем уверенность
                            extraction_method="function_calling_with_warnings"
                        )
                else:
                    logger.warning("Модель не вернула function call")
                    return await self._extract_with_text_fallback(text, content)
            
            # Если response не в ожидаемом формате
            logger.warning("Неверный формат ответа от модели")
            return await self._extract_with_text_fallback(text)
            
        except Exception as e:
            logger.error(f"Ошибка в function calling: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=f"Function calling error: {str(e)}",
                extraction_method="function_calling_error"
            )
    
    async def _extract_with_regex(self, text: str) -> ProcessingResult:
        """Извлечение данных с использованием регулярных выражений"""
        try:
            logger.info("Начало извлечения данных с помощью регулярных выражений")
            
            medical_tests = []
            lines = text.split('\n')
            
            # Получаем паттерны из конфигурации
            test_name_patterns = self.config.get_patterns("test_name")
            result_patterns = self.config.get_patterns("result")
            reference_patterns = self.config.get_patterns("reference")
            units_patterns = self.config.get_patterns("units")
            
            for line in lines:
                test = self._extract_test_from_line(
                    line, test_name_patterns, result_patterns, 
                    reference_patterns, units_patterns
                )
                if test:
                    # Валидируем тест
                    validation = self.validator.validate_medical_test(test)
                    if validation.is_valid and validation.cleaned_data:
                        medical_tests.append(validation.cleaned_data)
            
            # Категоризируем тесты
            categorized_tests = self._categorize_tests(medical_tests)
            
            result_data = {
                "medical_tests": categorized_tests,
                "summary": f"Извлечено {len(categorized_tests)} анализов с помощью регулярных выражений"
            }
            
            return ProcessingResult(
                success=True,
                data=result_data,
                metadata={"extraction_method": "regex_fallback"},
                confidence=0.6,  # Средняя уверенность для regex
                extraction_method="regex_fallback"
            )
            
        except Exception as e:
            logger.error(f"Ошибка извлечения с помощью regex: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=f"Regex extraction error: {str(e)}",
                extraction_method="regex_error"
            )
    
    async def _extract_with_text_fallback(self, text: str, llm_response: str = None) -> ProcessingResult:
        """Fallback метод извлечения на основе текстового ответа LLM"""
        try:
            logger.info("Используем текстовый fallback метод")
            
            if llm_response:
                # Пытаемся извлечь структуру из текстового ответа
                structured_data = self._parse_text_response(llm_response)
                if structured_data:
                    validation_result = self._validate_extracted_data(structured_data)
                    
                    return ProcessingResult(
                        success=True,
                        data=structured_data,
                        metadata={"extraction_method": "text_fallback"},
                        confidence=validation_result.confidence * 0.7,
                        extraction_method="text_fallback"
                    )
            
            # Если не удалось, возвращаем базовый результат
            return ProcessingResult(
                success=True,
                data={
                    "medical_tests": [],
                    "summary": "Не удалось извлечь структурированные данные. Текст документа требует ручной обработки.",
                    "raw_text": text[:500] + "..." if len(text) > 500 else text
                },
                metadata={"extraction_method": "basic_fallback"},
                confidence=0.2,
                extraction_method="basic_fallback"
            )
            
        except Exception as e:
            logger.error(f"Ошибка в текстовом fallback: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=f"Text fallback error: {str(e)}",
                extraction_method="text_fallback_error"
            )
    
    def _extract_test_from_line(self, line: str, test_patterns, result_patterns, 
                             reference_patterns, units_patterns) -> Optional[Dict[str, Any]]:
        """Извлечь тест из строки"""
        try:
            line = line.strip()
            if not line:
                return None
            
            test_name = None
            result = None
            reference = None
            units = None
            
            # Ищем название теста
            for pattern in test_patterns:
                match = pattern.search(line)
                if match:
                    test_name = match.group(1).strip()
                    break
            
            if not test_name:
                return None
            
            # Ищем результат
            for pattern in result_patterns:
                match = pattern.search(line)
                if match:
                    result = match.group(1).strip()
                    break
            
            # Ищем референсные значения
            for pattern in reference_patterns:
                match = pattern.search(line)
                if match:
                    reference = match.group(1).strip()
                    break
            
            # Ищем единицы измерения
            for pattern in units_patterns:
                match = pattern.search(line)
                if match:
                    units = match.group(1).strip()
                    break
            
            if test_name and result:
                return {
                    "test_name": test_name,
                    "result": result,
                    "reference_values": reference or "",
                    "units": units or ""
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка извлечения теста из строки: {e}")
            return None
    
    def _categorize_tests(self, tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Категоризировать тесты"""
        categories = self.config.get_categories()
        
        for test in tests:
            test_name = test.get("test_name", "").lower()
            category = "Другие анализы"
            
            # Ищем подходящую категорию
            for cat_key, cat_config in categories.items():
                keywords = cat_config.get("keywords", [])
                if any(keyword.lower() in test_name for keyword in keywords):
                    category = cat_config["name"]
                    test["category"] = cat_key  # Сохраняем ключ категории
                    break
            
            test["category_name"] = category  # Сохраняем отображаемое имя
        
        return tests
    
    def _validate_extracted_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Валидация извлеченных данных"""
        try:
            medical_tests = data.get("medical_tests", [])
            if not medical_tests:
                return ValidationResult(
                    is_valid=False,
                    errors=["Нет медицинских тестов в данных"],
                    warnings=[],
                    confidence=0.0
                )
            
            # Валидация каждого теста
            validation_results = self.validator.batch_validate(medical_tests)
            
            # Собираем ошибки и предупреждения
            all_errors = []
            all_warnings = []
            total_confidence = 0.0
            
            for result in validation_results:
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                total_confidence += result.confidence
            
            # Обновляем данные валидированными
            cleaned_tests = []
            for i, result in enumerate(validation_results):
                if result.cleaned_data:
                    cleaned_tests.append(result.cleaned_data)
                elif medical_tests[i]:  # Если валидация провалена, но есть исходные данные
                    cleaned_tests.append(medical_tests[i])
            
            data["medical_tests"] = cleaned_tests
            
            return ValidationResult(
                is_valid=len(all_errors) == 0,
                errors=all_errors,
                warnings=all_warnings,
                cleaned_data=data,
                confidence=total_confidence / len(validation_results) if validation_results else 0.0
            )
            
        except Exception as e:
            logger.error(f"Ошибка валидации извлеченных данных: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                confidence=0.0
            )
    
    def _parse_text_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Парсинг текстового ответа LLM"""
        try:
            # Простой парсер - ищем структуру в тексте
            lines = text.split('\n')
            medical_tests = []
            
            for line in lines:
                line = line.strip()
                # Ищем паттерны вроде "Анализ: результат (норма)"
                if ':' in line and any(char in line for char in ['0', 'отриц', 'положител']):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        test_name = parts[0].strip()
                        result_part = parts[1].strip()
                        
                        # Извлекаем результат и референс
                        result = result_part
                        reference = ""
                        
                        if '(' in result_part and ')' in result_part:
                            result = result_part.split('(')[0].strip()
                            reference = result_part.split('(')[1].split(')')[0].strip()
                        
                        medical_tests.append({
                            "test_name": test_name,
                            "result": result,
                            "reference_values": reference
                        })
            
            if medical_tests:
                return {
                    "medical_tests": medical_tests,
                    "summary": "Извлечено из текстового ответа LLM"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка парсинга текстового ответа: {e}")
            return None

class ResponseFormatter:
    """Форматировщик ответов"""
    
    def __init__(self, config=None):
        self.config = config or medical_config
    
    def format_response(self, processing_result: ProcessingResult) -> str:
        """Форматировать ответ для пользователя"""
        if not processing_result.success:
            return self._format_error(processing_result.error, processing_result.extraction_method)
        
        data = processing_result.data
        template = self.config.get_template("success")
        
        # Формируем ответ на основе шаблона
        response = template.get("header", "📊 **Анализ медицинских документов:**\n\n")
        
        # Добавляем информацию о пациенте
        patient_info = data.get("patient_info", {})
        if patient_info:
            patient_section = template.get("patient_section", "")
            if patient_section:
                response += patient_section.format(**patient_info)
        
        # Добавляем информацию о документе
        document_info = data.get("document_info", {})
        if document_info:
            document_section = template.get("document_section", "")
            if document_section:
                response += document_section.format(**document_info)
        
        # Добавляем анализы по категориям
        medical_tests = data.get("medical_tests", [])
        if medical_tests:
            categorized_tests = self._group_tests_by_category(medical_tests)
            
            status_icons = template.get("status", {
                "normal": "✅",
                "abnormal": "⚠️",
                "critical": "🚨"
            })
            
            for category, tests in categorized_tests.items():
                if not tests:
                    continue
                    
                category_section = template.get("category_section", "")
                tests_text = "\n".join([
                    self._format_test_item(test, template.get("test_item", ""), status_icons)
                    for test in tests
                ])
                
                response += category_section.format(
                    category_name=category,
                    tests=tests_text
                )
        
        # Добавляем резюме
        summary = data.get("summary", "")
        if summary:
            summary_section = template.get("summary_section", "")
            if summary_section:
                response += summary_section.format(summary=summary)
        
        # Добавляем рекомендации
        recommendations = self._generate_recommendations(medical_tests, processing_result.confidence)
        recommendations_section = template.get("recommendations_section", "")
        if recommendations_section:
            response += recommendations_section.format(recommendations=recommendations)
        
        # Добавляем футер
        footer = template.get("footer", "")
        if footer:
            response += footer
        
        # Добавляем метаданные обработки
        if processing_result.metadata:
            metadata_info = self._format_metadata(processing_result.metadata)
            response += f"\n\n---\n🔧 **Техническая информация:**\n{metadata_info}"
        
        return response
    
    def _format_error(self, error: str, extraction_method: str) -> str:
        """Форматировать ошибку"""
        template = self.config.get_template("error", {})
        
        # Выбираем подходящее сообщение об ошибке
        if "extraction_failed" in error.lower():
            return template.get("extraction_failed", "❌ Не удалось извлечь данные из документа")
        elif "ai_error" in error.lower():
            return template.get("ai_error", f"❌ Ошибка ИИ-обработки: {error}")
        else:
            return template.get("processing_failed", f"❌ Ошибка обработки документа: {error}").format(error=error)
    
    def _group_tests_by_category(self, tests: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Сгруппировать тесты по категориям"""
        categorized = {}
        categories = self.config.get_categories()
        
        for test in tests:
            category_name = test.get("category_name", "Другие анализы")
            
            if category_name not in categorized:
                categorized[category_name] = []
            categorized[category_name].append(test)
        
        # Сортируем категории по приоритету
        sorted_categories = {}
        for cat_key, cat_config in sorted(categories.items(), key=lambda x: x[1].get("priority", 999)):
            cat_name = cat_config["name"]
            if cat_name in categorized:
                sorted_categories[cat_name] = categorized[cat_name]
        
        # Добавляем категорию "Другие" в конец
        if "Другие анализы" in categorized:
            sorted_categories["Другие анализы"] = categorized["Другие анализы"]
        
        return sorted_categories
    
    def _format_test_item(self, test: Dict[str, Any], template: str, status_icons: Dict[str, str]) -> str:
        """Отформатировать отдельный тест"""
        test_name = test.get("test_name", "Неизвестный анализ")
        result = test.get("result", "Не указан")
        reference = test.get("reference_values", "")
        units = test.get("units", "")
        
        # Определяем статус
        status = "normal"  # По умолчанию
        if reference and result:
            status = self._determine_test_status(result, reference)
        
        status_icon = status_icons.get(status, "✅")
        
        return template.format(
            status=status_icon,
            test_name=test_name,
            result=result,
            units=units,
            reference=reference
        )
    
    def _determine_test_status(self, result: str, reference: str) -> str:
        """Определить статус теста (норма/отклонение)"""
        try:
            # Для качественных результатов
            if result.lower() in ["отрицательно", "negative", "не обнаружено"]:
                return "normal"
            elif result.lower() in ["положительно", "positive", "обнаружено"]:
                return "abnormal"
            
            # Для числовых результатов
            result_num = self._extract_number(result)
            if result_num is None:
                return "normal"
            
            # Проверяем диапазон
            if '-' in reference or '—' in reference:
                # Диапазон вида "10-20"
                parts = re.split(r'[-—]', reference)
                if len(parts) == 2:
                    min_val = self._extract_number(parts[0].strip())
                    max_val = self._extract_number(parts[1].strip())
                    if min_val is not None and max_val is not None:
                        if min_val <= result_num <= max_val:
                            return "normal"
                        else:
                            return "abnormal"
            
            # Проверяем пороговые значения
            elif reference.startswith('<'):
                max_val = self._extract_number(reference[1:].strip())
                if max_val is not None and result_num < max_val:
                    return "normal"
                else:
                    return "abnormal"
            elif reference.startswith('>'):
                min_val = self._extract_number(reference[1:].strip())
                if min_val is not None and result_num > min_val:
                    return "normal"
                else:
                    return "abnormal"
            
            return "normal"
            
        except Exception as e:
            logger.error(f"Ошибка определения статуса теста: {e}")
            return "normal"
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Извлечь число из текста"""
        import re
        match = re.search(r'([0-9]+\.?[0-9]*)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def _generate_recommendations(self, tests: List[Dict[str, Any]], confidence: float) -> str:
        """Сгенерировать рекомендации"""
        recommendations = []
        
        # Общие рекомендации
        recommendations.append("• Проконсультируйтесь с врачом для детальной интерпретации результатов")
        recommendations.append("• При необходимости повторите анализы через рекомендованный промежуток времени")
        recommendations.append("• Сохраните результаты для отслеживания динамики показателей")
        
        # Рекомендации на основе уверенности
        if confidence < 0.5:
            recommendations.append("• ⚠️ Низкая уверенность извлечения данных - проверьте результаты вручную")
        elif confidence < 0.7:
            recommendations.append("• ⚠️ Средняя уверенность извлечения - возможны ошибки")
        
        # Рекомендации на основе аномальных результатов
        abnormal_tests = [t for t in tests if self._has_abnormal_result(t)]
        if abnormal_tests:
            recommendations.append(f"• Обратите особое внимание на {len(abnormal_tests)} показателей, выходящих за пределы нормы")
        
        return "\n".join(recommendations)
    
    def _has_abnormal_result(self, test: Dict[str, Any]) -> bool:
        """Проверить, есть ли аномальные результаты"""
        result = test.get("result", "")
        reference = test.get("reference_values", "")
        
        if not reference:
            return False
        
        status = self._determine_test_status(result, reference)
        return status != "normal"
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Отформатировать метаданные"""
        info = []
        
        if "extraction_method" in metadata:
            info.append(f"• Метод извлечения: {metadata['extraction_method']}")
        
        if "model_provider" in metadata and "model_name" in metadata:
            info.append(f"• Модель: {metadata['model_provider']} - {metadata['model_name']}")
        
        if "tokens_used" in metadata:
            info.append(f"• Использовано токенов: {metadata['tokens_used']}")
        
        if "validation_warnings" in metadata and metadata["validation_warnings"]:
            info.append(f"• Предупреждения валидации: {len(metadata['validation_warnings'])}")
        
        return "\n".join(info)

class UniversalDocumentProcessor:
    """Универсальный процессор документов"""
    
    def __init__(self, config=None):
        self.config = config or medical_config
        self.extractors = {
            "pdf": PDFExtractor(),
            "image": ImageExtractor()
        }
        self.data_processor = DataProcessor(self.config)
        self.response_formatter = ResponseFormatter(self.config)
        self.monitor = processing_monitor
        self.logger = logging.getLogger(__name__)
    
    async def process_document(self, source: str, document_type: str) -> ProcessingResult:
        """Универсальный метод обработки документа"""
        session_id = self.monitor.start_processing(document_type)
        start_time = time.time()
        
        try:
            self.logger.info(f"Начинаю обработку {document_type} документа")
            
            # Шаг 1: Извлечение текста
            extractor = self.extractors.get(document_type)
            if not extractor:
                return ProcessingResult(
                    success=False,
                    data={},
                    error=f"Неподдерживаемый тип документа: {document_type}",
                    extraction_method="unsupported_type"
                )
            
            extracted_text = await extractor.extract_text(source)
            if not extracted_text:
                return ProcessingResult(
                    success=False,
                    data={},
                    error="Не удалось извлечь текст из документа",
                    extraction_method="text_extraction_failed"
                )
            
            # Шаг 2: Обработка данных
            processing_result = await self.data_processor.extract_structured_data(
                extracted_text, document_type
            )
            
            # Шаг 3: Форматирование ответа
            if processing_result.success:
                formatted_response = self.response_formatter.format_response(processing_result)
                processing_result.data["formatted_response"] = formatted_response
            
            # Шаг 4: Логирование метрик
            self.monitor.end_processing(
                session_id=session_id,
                document_type=document_type,
                start_time=start_time,
                success=processing_result.success,
                extraction_method=processing_result.extraction_method,
                tests_count=len(processing_result.data.get("medical_tests", [])),
                confidence_score=processing_result.confidence,
                error=processing_result.error,
                model_provider=processing_result.metadata.get("model_provider") if processing_result.metadata else None,
                model_name=processing_result.metadata.get("model_name") if processing_result.metadata else None,
                tokens_used=processing_result.metadata.get("tokens_used") if processing_result.metadata else None
            )
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки документа: {e}")
            
            # Логируем ошибку
            self.monitor.end_processing(
                session_id=session_id,
                document_type=document_type,
                start_time=start_time,
                success=False,
                extraction_method="error",
                tests_count=0,
                confidence_score=0.0,
                error=str(e)
            )
            
            return ProcessingResult(
                success=False,
                data={},
                error=str(e),
                processing_time=time.time() - start_time,
                extraction_method="error"
            )
    
    def get_supported_types(self) -> List[str]:
        """Получить поддерживаемые типы документов"""
        return list(self.extractors.keys())
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Получить статистику обработки"""
        return self.monitor.get_statistics()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Получить статус здоровья системы"""
        from core.monitoring import health_checker
        return health_checker.check_health()

# Глобальный экземпляр универсального процессора
universal_processor = UniversalDocumentProcessor()
