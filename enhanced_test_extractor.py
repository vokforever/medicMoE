"""
Enhanced Medical Test Extractor
Использует продвинутые LLM модели для точного распознавания и структурирования медицинских анализов из изображений
"""

import logging
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from models import call_model_with_failover

class EnhancedTestExtractor:
    """Улучшенный экстрактор медицинских анализов с использованием LLM"""
    
    def __init__(self):
        self.max_retries = 3
        self.medical_test_patterns = {
            'hepatitis': [
                'anti-hbs', 'hbsag', 'anti-hcv', 'anti-hbc', 'anti-hev', 'hbeag',
                'anti-hbs total', 'anti-hcv total', 'anti-hb core total', 'anti-hev igg',
                'anti-hev igm', 'гепатит', 'hepatitis'
            ],
            'parasites': [
                'anti-opisthorchis', 'anti-toxocara', 'anti-lamblia', 'anti-ascaris',
                'описторхоз', 'токсокароз', 'лямблиоз', 'аскаридоз'
            ],
            'allergy': [
                'ige', 'ige total', 'общий ige', 'аллергия', 'allergy',
                'эозинофилы', 'эозинофилы'
            ],
            'biochemistry': [
                'билирубин', 'алат', 'асат', 'ггт', 'щф', 'холестерин', 'глюкоза',
                'креатинин', 'мочевина', 'мочевая кислота', 'белок', 'альбумин'
            ],
            'hormones': [
                'ттг', 'т3', 'т4', 'т3 свободный', 'т4 свободный', 'tsh', 't3', 't4'
            ],
            'blood': [
                'гемоглобин', 'эритроциты', 'лейкоциты', 'тромбоциты', 'соэ',
                'гематокрит', 'mcv', 'mch', 'mchc', 'rdw', 'pdw', 'mpv'
            ]
        }
    
    async def extract_tests_from_image(self, image_url: str, user_query: str = "") -> Dict[str, Any]:
        """
        Извлекает и структурирует медицинские анализы из изображения
        
        Args:
            image_url: URL изображения с анализами
            user_query: Дополнительный запрос пользователя
            
        Returns:
            Dict с результатами извлечения
        """
        try:
            logging.info(f"Начинаю извлечение анализов из изображения: {image_url}")
            
            # Шаг 1: Базовый анализ изображения для получения текста
            basic_analysis = await self._analyze_image_for_text(image_url)
            
            # Шаг 2: Продвинутое извлечение структурированных данных
            structured_data = await self._extract_structured_tests(basic_analysis, user_query)
            
            # Шаг 3: Валидация и очистка данных
            validated_data = await self._validate_and_clean_tests(structured_data)
            
            # Шаг 4: Дополнительное извлечение метаданных
            metadata = await self._extract_metadata(basic_analysis)
            
            result = {
                "success": True,
                "raw_analysis": basic_analysis,
                "structured_tests": validated_data,
                "metadata": metadata,
                "tests_count": len(validated_data),
                "extraction_time": datetime.now().isoformat()
            }
            
            logging.info(f"Успешно извлечено {len(validated_data)} анализов из изображения")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении анализов из изображения: {e}")
            return {
                "success": False,
                "error": str(e),
                "structured_tests": [],
                "tests_count": 0
            }
    
    async def _analyze_image_for_text(self, image_url: str) -> str:
        """
        Анализирует изображение и извлекает текстовую информацию
        """
        try:
            logging.info("Выполняю базовый анализ изображения для извлечения текста")
            
            prompt = """
            Проанализируй это медицинское изображение и извлеки ВСЮ текстовую информацию.
            
            ВАЖНО:
            1. Извлеки ПОЛНОСТЬЮ весь текст с изображения, включая:
               - Названия анализов
               - Результаты (числовые значения, "отрицательно", "положительно")
               - Референсные значения (нормы)
               - Единицы измерения
               - Дату анализа
               - Информацию о пациенте
               - Название лаборатории/оборудования
               - Любые другие медицинские данные
            
            2. Сохраняй ОРИГИНАЛЬНОЕ форматирование и структуру текста
            3. Не пропускай никакие детали, даже если они кажутся незначительными
            4. Если текст нечеткий, попробуй реконструировать его по контексту
            
            Верни извлеченный текст в точном виде, как он представлен на изображении.
            """
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ]
            
            response, provider, metadata = await call_model_with_failover(
                messages=messages,
                model_type="vision",
                system_prompt="Ты — медицинский эксперт по извлечению текстовой информации из изображений анализов."
            )
            
            logging.info(f"Базовый анализ изображения завершен, длина текста: {len(response)}")
            return response
            
        except Exception as e:
            logging.error(f"Ошибка при базовом анализе изображения: {e}")
            raise
    
    async def _extract_structured_tests(self, text_content: str, user_query: str = "") -> List[Dict[str, Any]]:
        """
        Извлекает структурированные данные анализов из текста
        """
        try:
            logging.info("Начинаю извлечение структурированных данных анализов")
            
            # Используем function calling для структурированного извлечения
            functions = [
                {
                    "name": "extract_medical_tests",
                    "description": "Извлекает структурированные данные медицинских анализов из текста",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tests": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "test_name": {
                                            "type": "string",
                                            "description": "Полное название анализа (на русском и/или английском)"
                                        },
                                        "result": {
                                            "type": "string", 
                                            "description": "Точный результат (числовое значение, 'отрицательно', 'положительно', 'в норме' и т.д.)"
                                        },
                                        "reference_values": {
                                            "type": "string",
                                            "description": "Референсные значения (норма) если указаны"
                                        },
                                        "units": {
                                            "type": "string",
                                            "description": "Единицы измерения (МЕ/мл, мг/л, г/л, % и т.д.)"
                                        },
                                        "test_system": {
                                            "type": "string",
                                            "description": "Тест-система или метод исследования если указана"
                                        },
                                        "equipment": {
                                            "type": "string",
                                            "description": "Оборудование/анализатор если указано"
                                        },
                                        "test_date": {
                                            "type": "string",
                                            "description": "Дата сдачи анализа если указана"
                                        }
                                    },
                                    "required": ["test_name", "result"]
                                }
                            }
                        },
                        "required": ["tests"]
                    }
                }
            ]
            
            prompt = f"""
            Проанализируй следующий текст с медицинскими анализами и извлеки структурированную информацию.
            
            ТЕКСТ АНАЛИЗОВ:
            {text_content}
            
            ДОПОЛНИТЕЛЬНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
            {user_query}
            
            ПРАВИЛА ИЗВЛЕЧЕНИЯ:
            - НЕ используй символы "**", "*" или другие маркеры вместо реальных значений
            - Если результат "отрицательно" или "положительно" - используй эти слова
            - Если есть числовые значения - извлеки точные числа
            - Если референсные значения указаны в формате "0-10" или "< 5" - извлеки как есть
            - Сохраняй оригинальные названия анализов, включая аббревиатуры
            - Ищи дату анализа в тексте (форматы: ДД.ММ.ГГГГ, ДД/ММ/ГГГГ)
            - Ищи информацию о тест-системах (Abbott, Roche, Siemens и т.д.)
            - Ищи информацию об оборудовании (Alinity i, Cobas e602 и т.д.)
            
            ТИПЫ АНАЛИЗОВ для распознавания:
            - Анализы на гепатиты: Anti-HBs, HBsAg, Anti-HCV, Anti-HBc, Anti-HEV
            - Паразитологические: Anti-Opisthorchis, Anti-Toxocara, Anti-Lamblia
            - Аллергологические: IgE общий, эозинофилы
            - Биохимические: билирубин, АЛТ, АСТ, ГГТ, холестерин
            - Гормональные: ТТГ, Т3, Т4
            
            Используй функцию extract_medical_tests для извлечения данных.
            """
            
            messages = [
                {
                    "role": "system",
                    "content": "Ты — эксперт по медицинским анализам. Твоя задача - точно извлекать структурированные данные из медицинских документов."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response, provider, metadata = await call_model_with_failover(
                messages=messages,
                model_type="text"
            )
            
            # Пытаемся извлечь JSON из ответа
            try:
                logging.info(f"Парсинг ответа модели, длина: {len(response)} символов")
                logging.info(f"Начало ответа: {response[:500]}...")
                
                # Если модель поддерживает function calling, ответ будет в формате JSON
                if "extract_medical_tests" in response:
                    data = json.loads(response)
                    tests = data.get("extract_medical_tests", {}).get("tests", [])
                else:
                    # Fallback - ищем JSON в тексте ответа
                    # Ищем JSON объект с массивом tests
                    json_patterns = [
                        r'\{[^{}]*"tests"\s*:\s*\[[^\]]*\][^{}]*\}',  # JSON с массивом tests
                        r'\{[^{}]*"extract_medical_tests"[^{}]*\}',  # JSON с extract_medical_tests
                        r'\{.*?\}',  # Любой JSON объект
                    ]
                    
                    tests = []
                    for pattern in json_patterns:
                        json_match = re.search(pattern, response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            logging.info(f"Найден JSON: {json_str[:200]}...")
                            
                            try:
                                data = json.loads(json_str)
                                
                                # Проверяем разные возможные структуры
                                if "tests" in data:
                                    tests = data.get("tests", [])
                                elif "extract_medical_tests" in data:
                                    tests = data.get("extract_medical_tests", {}).get("tests", [])
                                elif isinstance(data, list):
                                    tests = data
                                
                                if tests:
                                    break
                                    
                            except json.JSONDecodeError as e:
                                logging.warning(f"Не удалось распарсить JSON паттерном {pattern}: {e}")
                                continue
                    
                    # Если все еще не нашли, пробуем извлечь тесты вручную
                    if not tests:
                        logging.info("JSON не найден, пробуем ручное извлечение")
                        tests = self._extract_tests_manually(response)
                
                logging.info(f"Извлечено {len(tests)} структурированных анализов")
                return tests
                
            except json.JSONDecodeError as e:
                logging.error(f"Ошибка парсинга JSON: {e}")
                logging.error(f"Ответ модели: {response}")
                # Пробуем ручное извлечение как fallback
                return self._extract_tests_manually(response)
            except Exception as e:
                logging.error(f"Неожиданная ошибка при парсинге: {e}")
                logging.error(f"Ответ модели: {response}")
                return []
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении структурированных данных: {e}")
            return []
    
    async def _validate_and_clean_tests(self, tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Валидирует и очищает извлеченные данные анализов
        """
        try:
            logging.info("Начинаю валидацию и очистку данных анализов")
            
            cleaned_tests = []
            
            for test in tests:
                cleaned_test = {
                    "test_name": self._clean_test_name(test.get("test_name", "")),
                    "result": self._clean_result(test.get("result", "")),
                    "reference_values": self._clean_reference_values(test.get("reference_values", "")),
                    "units": self._clean_units(test.get("units", "")),
                    "test_system": self._clean_test_system(test.get("test_system", "")),
                    "equipment": self._clean_equipment(test.get("equipment", "")),
                    "test_date": self._clean_date(test.get("test_date", "")),
                    "notes": None
                }
                
                # Дополнительная валидация
                if self._is_valid_test(cleaned_test):
                    cleaned_tests.append(cleaned_test)
                    logging.info(f"Валидный тест: {cleaned_test['test_name']} = {cleaned_test['result']}")
                else:
                    logging.warning(f"Невалидный тест пропущен: {test}")
            
            logging.info(f"После валидации: {len(cleaned_tests)} корректных анализов")
            return cleaned_tests
            
        except Exception as e:
            logging.error(f"Ошибка при валидации данных: {e}")
            return []
    
    def _clean_test_name(self, name: str) -> str:
        """Очищает название анализа"""
        if not name:
            return ""
        
        # Убираем лишние символы и нормализуем
        cleaned = re.sub(r'[\*\*]', '', name)
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)  # Убираем номера в начале
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _clean_result(self, result: str) -> str:
        """Очищает результат анализа"""
        if not result:
            return ""
        
        # Убираем лишние символы
        cleaned = re.sub(r'[\*\*]', '', result)
        cleaned = cleaned.strip()
        
        # Нормализуем распространенные значения
        cleaned_lower = cleaned.lower()
        if 'отриц' in cleaned_lower:
            return "отрицательно"
        elif 'полож' in cleaned_lower:
            return "положительно"
        elif 'норм' in cleaned_lower:
            return "в норме"
        
        return cleaned
    
    def _clean_reference_values(self, ref_values: str) -> str:
        """Очищает референсные значения"""
        if not ref_values:
            return ""
        
        cleaned = re.sub(r'[\*\*]', '', ref_values)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _clean_units(self, units: str) -> str:
        """Очищает единицы измерения"""
        if not units:
            return ""
        
        cleaned = re.sub(r'[\*\*]', '', units)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _clean_test_system(self, test_system: str) -> str:
        """Очищает название тест-системы"""
        if not test_system:
            return ""
        
        cleaned = re.sub(r'[\*\*]', '', test_system)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _clean_equipment(self, equipment: str) -> str:
        """Очищает название оборудования"""
        if not equipment:
            return ""
        
        cleaned = re.sub(r'[\*\*]', '', equipment)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _clean_date(self, date_str: str) -> str:
        """Очищает и нормализует дату"""
        if not date_str:
            return ""
        
        cleaned = re.sub(r'[\*\*]', '', date_str)
        cleaned = cleaned.strip()
        
        # Пытаемся распарсить дату в различных форматах
        date_patterns = [
            r'(\d{2})\.(\d{2})\.(\d{4})',  # DD.MM.YYYY
            r'(\d{2})/(\d{2})/(\d{4})',  # DD/MM/YYYY
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, cleaned)
            if match:
                if pattern == date_patterns[0]:  # DD.MM.YYYY
                    day, month, year = match.groups()
                    return f"{year}-{month}-{day}"
                elif pattern == date_patterns[1]:  # DD/MM/YYYY
                    day, month, year = match.groups()
                    return f"{year}-{month}-{day}"
                elif pattern == date_patterns[2]:  # YYYY-MM-DD
                    return cleaned
        
        return cleaned
    
    def _is_valid_test(self, test: Dict[str, Any]) -> bool:
        """Проверяет валидность данных анализа"""
        # Минимальные требования: название и результат
        if not test.get("test_name") or not test.get("result"):
            return False
        
        # Проверяем, что результат не содержит только символы форматирования
        result = test.get("result", "")
        if not result or result in ["**", "*", "Не указан"]:
            return False
        
        return True
    
    async def _extract_metadata(self, text_content: str) -> Dict[str, Any]:
        """
        Извлекает метаданные из текста анализа
        """
        try:
            metadata = {
                "patient_name": None,
                "patient_birth_date": None,
                "laboratory": None,
                "doctor": None,
                "analysis_date": None
            }
            
            lines = text_content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Ищем имя пациента
                if any(keyword in line.lower() for keyword in ['пациент', 'фио', 'имя']):
                    name_match = re.search(r':\s*([А-Яа-я\s]+)', line)
                    if name_match:
                        metadata["patient_name"] = name_match.group(1).strip()
                
                # Ищем дату рождения
                if any(keyword in line.lower() for keyword in ['дата рождения', 'др', 'родился']):
                    date_match = re.search(r'(\d{2}[\.\/\-]\d{2}[\.\/\-]\d{4})', line)
                    if date_match:
                        metadata["patient_birth_date"] = date_match.group(1)
                
                # Ищем лабораторию
                if any(keyword in line.lower() for keyword in ['лаборатория', 'кдл', 'инвитро']):
                    metadata["laboratory"] = line
                
                # Ищем врача
                if any(keyword in line.lower() for keyword in ['врач', 'доктор']):
                    metadata["doctor"] = line
            
            return metadata
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении метаданных: {e}")
            return {}
    
    def _extract_tests_manually(self, response: str) -> List[Dict[str, Any]]:
        """
        Ручное извлечение тестов из текстового ответа модели как fallback
        """
        try:
            logging.info("Начинаю ручное извлечение тестов из текстового ответа")
            
            tests = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Ищем паттерны медицинских анализов в тексте
                for category, keywords in self.medical_test_patterns.items():
                    for keyword in keywords:
                        if keyword.lower() in line.lower():
                            # Пытаемся извлечь данные из строки
                            test_data = self._parse_test_line(line)
                            if test_data:
                                tests.append(test_data)
                                logging.info(f"Ручное извлечение: {test_data['test_name']} = {test_data['result']}")
                            break
            
            # Если не нашли по ключевым словам, пробуем общий парсинг
            if not tests:
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Ищем строки с двоеточием (название: результат)
                    if ':' in line and len(line) > 10:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            test_name = parts[0].strip()
                            result = parts[1].strip()
                            
                            # Проверяем, что это похоже на медицинский анализ
                            if self._looks_like_medical_test(test_name, result):
                                test_data = {
                                    "test_name": test_name,
                                    "result": result,
                                    "reference_values": "",
                                    "units": "",
                                    "test_system": "",
                                    "equipment": "",
                                    "test_date": ""
                                }
                                tests.append(test_data)
                                logging.info(f"Общее извлечение: {test_name} = {result}")
            
            logging.info(f"Ручное извлечение завершено, найдено {len(tests)} тестов")
            return tests
            
        except Exception as e:
            logging.error(f"Ошибка при ручном извлечении тестов: {e}")
            return []
    
    def _parse_test_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Парсит строку с медицинским анализом
        """
        try:
            # Ищем паттерн "Название: Результат"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    test_name = parts[0].strip()
                    result = parts[1].strip()
                    
                    # Очищаем данные
                    test_name = self._clean_test_name(test_name)
                    result = self._clean_result(result)
                    
                    if test_name and result:
                        return {
                            "test_name": test_name,
                            "result": result,
                            "reference_values": "",
                            "units": "",
                            "test_system": "",
                            "equipment": "",
                            "test_date": ""
                        }
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка при парсинге строки анализа: {e}")
            return None
    
    def _looks_like_medical_test(self, test_name: str, result: str) -> bool:
        """
        Проверяет, похожа ли строка на медицинский анализ
        """
        try:
            # Проверяем ключевые слова в названии
            medical_keywords = [
                'anti-', 'igg', 'igm', 'ige', 'гепатит', 'hepatitis', 'аллергия', 'allergy',
                'билирубин', 'алат', 'асат', 'ггт', 'щф', 'холестерин', 'глюкоза',
                'креатинин', 'мочевина', 'белок', 'ттг', 'т3', 'т4', 'tsh', 't3', 't4',
                'гемоглобин', 'эритроциты', 'лейкоциты', 'тромбоциты', 'соэ'
            ]
            
            test_name_lower = test_name.lower()
            if any(keyword in test_name_lower for keyword in medical_keywords):
                return True
            
            # Проверяем результат
            result_lower = result.lower()
            if any(keyword in result_lower for keyword in ['отрицательно', 'положительно', 'норма', ' negative', 'positive']):
                return True
            
            # Проверяем числовые значения
            if re.search(r'\d+\.?\d*\s*(ме/мл|мг/л|г/л|ммоль/л|%)', result_lower):
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Ошибка при проверке медицинского анализа: {e}")
            return False

    async def extract_specific_test(self, text_content: str, test_name: str) -> Optional[Dict[str, Any]]:
        """
        Извлекает данные конкретного анализа по названию
        """
        try:
            prompt = f"""
            Найди в следующем тексте информацию об анализе "{test_name}" и извлеки все данные.
            
            ТЕКСТ:
            {text_content}
            
            ИЗВЛЕКИ:
            - Точный результат
            - Референсные значения
            - Единицы измерения
            - Дату анализа
            
            Верни результат в формате JSON:
            {{
                "test_name": "{test_name}",
                "result": "результат",
                "reference_values": "норма",
                "units": "единицы",
                "test_date": "дата"
            }}
            
            Если анализ не найден, верни {{"found": false}}.
            """
            
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response, provider, metadata = await call_model_with_failover(
                messages=messages,
                model_type="text"
            )
            
            try:
                data = json.loads(response)
                if data.get("found") is False:
                    return None
                return data
            except:
                return None
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении конкретного анализа: {e}")
            return None

# Глобальная функция для удобного использования
async def extract_medical_tests_from_image(image_url: str, user_query: str = "") -> Dict[str, Any]:
    """
    Удобная функция для извлечения медицинских анализов из изображения
    
    Args:
        image_url: URL изображения
        user_query: Дополнительный запрос пользователя
        
    Returns:
        Dict с результатами извлечения
    """
    extractor = EnhancedTestExtractor()
    return await extractor.extract_tests_from_image(image_url, user_query)
