"""
Модуль для интеллектуального определения медицинских терминов с использованием LLM function calling
Заменяет хардкод списков медицинских терминов на динамическое определение через ИИ
"""

import logging
import json
from typing import List, Dict, Any, Optional
from models import call_model_with_failover

class MedicalTermsAgent:
    """Агент для интеллектуального определения медицинских терминов"""
    
    def __init__(self):
        self.cache = {}  # Кэш для определений терминов
        self.cache_ttl = 3600  # Время жизни кэша в секундах
        
    async def extract_medical_keywords(self, text: str) -> List[str]:
        """
        Извлекает медицинские ключевые слова из текста с помощью LLM
        """
        try:
            logging.info(f"Извлечение медицинских ключевых слов из текста длиной {len(text)} символов")
            
            # Проверяем кэш
            cache_key = hash(text[:500])  # Используем хэш первых 500 символов
            if cache_key in self.cache:
                logging.info("Найдены ключевые слова в кэше")
                return self.cache[cache_key]
            
            # Используем LLM для извлечения медицинских терминов
            system_prompt = """Ты - медицинский эксперт. Извлеки из текста все медицинские термины, названия анализов, 
            заболеваний, биомаркеров, лекарств и процедур. Верни ответ в формате JSON списком строк.

            Примеры медицинских терминов:
            - Названия анализов: АЛТ, АСТ, билирубин, глюкоза, холестерин
            - Инфекции: гепатит, HCV, HBV, описторхоз, токсокароз
            - Антитела: Anti-HCV, Anti-HBs, IgG, IgM, IgE
            - Маркеры: С-реактивный белок, ферритин, ТТГ
            - Другое: аллергия, паразиты, гормоны

            Правила:
            1. Извлекай только медицински значимые термины
            2. Включай синонимы и вариации написания
            3. Игнорируй общие слова и служебную информацию
            4. Возвращай только термины на русском и английском языках"""
            
            user_prompt = f"Извлеки медицинские термины из этого текста:\n\n{text}"
            
            # Вызываем LLM с функцией
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await call_model_with_failover(
                messages=messages,
                model_type="text"
            )
            
            if isinstance(response, tuple):
                response_text = response[0]
            else:
                response_text = response
            
            # Парсим JSON ответ
            try:
                # Ищем JSON в ответе
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    medical_terms = json.loads(json_str)
                    
                    # Фильтруем и нормализуем термины
                    filtered_terms = []
                    for term in medical_terms:
                        if isinstance(term, str) and len(term.strip()) > 1:
                            filtered_terms.append(term.strip().lower())
                    
                    # Удаляем дубликаты
                    unique_terms = list(set(filtered_terms))
                    
                    # Сохраняем в кэш
                    self.cache[cache_key] = unique_terms
                    
                    logging.info(f"Извлечено {len(unique_terms)} уникальных медицинских терминов")
                    return unique_terms
                else:
                    logging.warning("Не найден JSON в ответе LLM")
                    return []
                    
            except json.JSONDecodeError as e:
                logging.error(f"Ошибка парсинга JSON ответа LLM: {e}")
                # Fallback: извлекаем термины вручную
                return self._fallback_extraction(response_text)
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении медицинских ключевых слов: {e}")
            return []
    
    def _fallback_extraction(self, text: str) -> List[str]:
        """Fallback метод извлечения терминов"""
        try:
            # Базовый список медицинских паттернов для fallback
            medical_patterns = [
                r'\b(?:анти-|anti-)?[a-zA-Z]{2,5}\b',
                r'\b(?:алт|аст|ттг|иг[егм]|ige|igg|igm)\b',
                r'\b(?:гепатит|hcv|hbv|hbsag)\b',
                r'\b(?:билирубин|глюкоза|холестерин|ферритин)\b',
                r'\b(?:описторхоз|токсокароз|лямблиоз)\b',
                r'\b(?:с-реактивный|церулоплазмин)\b'
            ]
            
            import re
            found_terms = []
            
            for pattern in medical_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found_terms.extend([match.lower() for match in matches])
            
            return list(set(found_terms))
            
        except Exception as e:
            logging.error(f"Ошибка в fallback извлечении: {e}")
            return []
    
    async def categorize_medical_test(self, test_name: str) -> Dict[str, Any]:
        """
        Категоризирует медицинский анализ с помощью LLM
        """
        try:
            logging.info(f"Категоризация анализа: {test_name}")
            
            # Проверяем кэш
            cache_key = f"cat_{hash(test_name.lower())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            system_prompt = """Ты - медицинский эксперт. Определи категорию медицинского анализа и верни ответ в формате JSON:

            {
                "category": "название категории",
                "subcategory": "подкатегория (если применимо)",
                "description": "краткое описание",
                "keywords": ["список", "ключевых", "слов"],
                "is_medical": true/false
            }

            Возможные категории:
            - Анализы на гепатиты
            - Аллергологические анализы
            - Паразитологические анализы
            - Биохимические анализы
            - Общий анализ крови
            - Гормональные анализы
            - Иммунологические анализы
            - Другие анализы"""
            
            user_prompt = f"Определи категорию для анализа: {test_name}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await call_model_with_failover(
                messages=messages,
                model_type="text"
            )
            
            if isinstance(response, tuple):
                response_text = response[0]
            else:
                response_text = response
            
            # Парсим JSON
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    category_data = json.loads(json_str)
                    
                    # Валидация данных
                    if not isinstance(category_data, dict):
                        category_data = {}
                    
                    # Добавляем значения по умолчанию
                    category_data.setdefault('category', 'Другие анализы')
                    category_data.setdefault('subcategory', '')
                    category_data.setdefault('description', '')
                    category_data.setdefault('keywords', [])
                    category_data.setdefault('is_medical', True)
                    
                    # Сохраняем в кэш
                    self.cache[cache_key] = category_data
                    
                    logging.info(f"Анализ '{test_name}' отнесен к категории: {category_data.get('category')}")
                    return category_data
                else:
                    logging.warning("Не найден JSON в ответе LLM для категоризации")
                    return self._fallback_categorization(test_name)
                    
            except json.JSONDecodeError as e:
                logging.error(f"Ошибка парсинга JSON для категоризации: {e}")
                return self._fallback_categorization(test_name)
            
        except Exception as e:
            logging.error(f"Ошибка при категоризации анализа: {e}")
            return self._fallback_categorization(test_name)
    
    def _fallback_categorization(self, test_name: str) -> Dict[str, Any]:
        """Fallback категоризация"""
        test_lower = test_name.lower()
        
        category_map = {
            "Анализы на гепатиты": ["гепатит", "hcv", "hbv", "hbsag", "anti-hcv", "anti-hbs", "anti-hev"],
            "Аллергологические анализы": ["ige", "аллерг", "аллергия"],
            "Паразитологические анализы": ["opisthorchis", "toxocara", "lamblia", "ascaris", "описторхоз", "токсокароз"],
            "Биохимические анализы": ["алт", "аст", "билирубин", "глюкоза", "холестерин", "креатинин", "мочевина"],
            "Общий анализ крови": ["гемоглобин", "лейкоциты", "эритроциты", "тромбоциты"],
            "Гормональные анализы": ["ттг", "т3", "т4", "tsh", "пролактин"],
            "Иммунологические анализы": ["igg", "igm", "цитокины", "интерлейкины"]
        }
        
        for category, keywords in category_map.items():
            if any(keyword in test_lower for keyword in keywords):
                return {
                    "category": category,
                    "subcategory": "",
                    "description": f"Анализ из категории {category}",
                    "keywords": keywords,
                    "is_medical": True
                }
        
        return {
            "category": "Другие анализы",
            "subcategory": "",
            "description": "Другой медицинский анализ",
            "keywords": [],
            "is_medical": True
        }
    
    async def extract_test_parameters(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлекает параметры анализов с помощью LLM
        """
        import re
        try:
            logging.info(f"Извлечение параметров анализов из текста")
            
            system_prompt = """Ты - медицинский эксперт. Извлеки из текста все медицинские анализы с их параметрами.
            Верни ответ в формате JSON списком объектов:

            [
                {
                    "test_name": "название анализа",
                    "result": "результат",
                    "units": "единицы измерения",
                    "reference_values": "референсные значения",
                    "test_date": "дата анализа",
                    "laboratory": "лаборатория",
                    "equipment": "оборудование",
                    "test_system": "тест-система"
                }
            ]

            Правила:
            1. Извлекай только полную информацию об анализе
            2. Единицы измерения могут быть в скобках или после значения
            3. Референсные значения могут быть помечены как "норма", "референс"
            4. Дата может быть в различных форматах
            5. Если поле не найдено, оставь его пустой строкой"""
            
            user_prompt = f"Извлеки параметры анализов из текста:\n\n{text}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await call_model_with_failover(
                messages=messages,
                model_type="text"
            )
            
            if isinstance(response, tuple):
                response_text = response[0]
            else:
                response_text = response
            
            # Парсим JSON с улучшенной обработкой ошибок
            try:
                # Ищем JSON в различных форматах
                json_str = None
                
                # Способ 1: Ищем массив [ ... ]
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                
                # Способ 2: Ищем объект { ... } и оборачиваем в массив
                if not json_str:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = "[" + response_text[json_start:json_end] + "]"
                
                # Способ 3: Ищем JSON с помощью регулярного выражения
                if not json_str:
                    import re
                    json_pattern = r'\[.*?\]|\{.*?\}(?=\s*[,;\n]|$)'
                    matches = re.findall(json_pattern, response_text, re.DOTALL)
                    if matches:
                        json_str = matches[0]
                
                if not json_str:
                    logging.warning("Не найден JSON в ответе LLM для параметров")
                    logging.debug(f"Ответ LLM: {response_text[:500]}...")
                    return []
                
                logging.debug(f"Найден JSON: {json_str[:200]}...")
                
                # Очищаем JSON от проблемных символов
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                json_str = re.sub(r'\s+', ' ', json_str)  # Удаляем лишние пробелы
                
                test_parameters = json.loads(json_str)
                
                # Если получили объект, преобразуем в массив
                if isinstance(test_parameters, dict):
                    test_parameters = [test_parameters]
                
                if isinstance(test_parameters, list):
                    # Валидация и очистка данных
                    cleaned_tests = []
                    for test in test_parameters:
                        if isinstance(test, dict):
                            # Очистка каждого поля
                            cleaned_test = {}
                            for key, value in test.items():
                                if isinstance(value, str):
                                    cleaned_test[key] = value.strip().replace('**', '').replace('*', '').replace('\n', ' ').strip()
                                elif value is None:
                                    cleaned_test[key] = ""
                                else:
                                    cleaned_test[key] = value
                            
                            # Проверяем обязательные поля
                            if cleaned_test.get('test_name') and cleaned_test.get('result'):
                                cleaned_tests.append(cleaned_test)
                    
                    logging.info(f"Извлечено {len(cleaned_tests)} параметров анализов")
                    return cleaned_tests
                else:
                    logging.warning("LLM вернула не список для параметров анализов")
                    return []
                    
            except json.JSONDecodeError as e:
                logging.error(f"Ошибка парсинга JSON для параметров: {e}")
                logging.debug(f"Проблемный JSON: {response_text[:500]}...")
                return []
            except Exception as e:
                logging.error(f"Непредвиденная ошибка при обработке параметров: {e}")
                return []
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении параметров анализов: {e}")
            return []
    
    async def is_medical_query(self, query: str) -> Dict[str, Any]:
        """
        Определяет, является ли запрос медицинским, с помощью LLM
        """
        try:
            logging.info(f"Проверка медицинского характера запроса: {query[:100]}...")
            
            system_prompt = """Ты - медицинский ассистент. Определи, является ли запрос медицинским и на какую тему.
            Верни ответ в формате JSON:

            {
                "is_medical": true/false,
                "category": "тип запроса",
                "keywords": ["ключевые", "слова"],
                "confidence": 0.95,
                "requires_doctor": true/false
            }

            Типы запросов:
            - medical_question: медицинский вопрос
            - test_analysis: анализ анализов
            - symptom_check: проверка симптомов
            - general_info: общая медицинская информация
            - non_medical: немедицинский запрос"""
            
            user_prompt = f"Проанализируй запрос: {query}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await call_model_with_failover(
                messages=messages,
                model_type="text"
            )
            
            if isinstance(response, tuple):
                response_text = response[0]
            else:
                response_text = response
            
            # Парсим JSON
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    analysis = json.loads(json_str)
                    
                    # Валидация и значения по умолчанию
                    analysis.setdefault('is_medical', False)
                    analysis.setdefault('category', 'non_medical')
                    analysis.setdefault('keywords', [])
                    analysis.setdefault('confidence', 0.0)
                    analysis.setdefault('requires_doctor', False)
                    
                    return analysis
                else:
                    return self._fallback_medical_check(query)
                    
            except json.JSONDecodeError as e:
                logging.error(f"Ошибка парсинга JSON для анализа запроса: {e}")
                return self._fallback_medical_check(query)
            
        except Exception as e:
            logging.error(f"Ошибка при анализе запроса: {e}")
            return self._fallback_medical_check(query)
    
    def _fallback_medical_check(self, query: str) -> Dict[str, Any]:
        """Fallback проверка медицинского характера"""
        query_lower = query.lower()
        
        medical_keywords = [
            'анализ', 'болит', 'симптом', 'лечение', 'диагноз', 'врач',
            'гепатит', 'аллергия', 'температура', 'давление', 'пульс',
            'лекарство', 'препарат', 'побочный эффект'
        ]
        
        is_medical = any(keyword in query_lower for keyword in medical_keywords)
        
        return {
            "is_medical": is_medical,
            "category": "medical_question" if is_medical else "non_medical",
            "keywords": [kw for kw in medical_keywords if kw in query_lower],
            "confidence": 0.7 if is_medical else 0.3,
            "requires_doctor": is_medical
        }
    
    def clear_cache(self):
        """Очищает кэш агента"""
        self.cache.clear()
        logging.info("Кэш медицинского агента очищен")

# Глобальный экземпляр агента
medical_terms_agent = MedicalTermsAgent()
