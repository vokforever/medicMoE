import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
from models import call_model_with_failover
from config import supabase, AGENT_CACHE_EXPIRE_HOURS

# Агент для уточнения информации и переключения режимов ИИ
class ClarificationAgent:
    def __init__(self):
        self.max_clarifications = 3  # Максимальное количество уточняющих вопросов
    
    async def analyze_and_ask(self, user_message: str, history: List[Dict[str, str]] = None, 
                             patient_data: Dict[str, Any] = None, clarification_count: int = 0,
                             has_medical_records: bool = False) -> Tuple[bool, str, str]:
        """
        Анализирует сообщение пользователя и определяет режим работы ИИ.
        
        Возвращает:
            - is_enough: True, если информации достаточно, иначе False
            - response: Уточняющий вопрос или None
            - ai_mode: "assistant" (ИИ-ассистент врача для сбора данных) или "doctor" (ИИ-врач главный)
        """
        # Определяем режим работы ИИ на основе наличия медицинских записей
        if has_medical_records:
            # Если есть загруженные анализы, переключаемся на режим ИИ-ассистента врача
            ai_mode = "assistant"
        else:
            # Если анализов нет, используем обычный режим
            ai_mode = "assistant"
        
        # Формируем контекст для модели
        context = f"""
        Ты - медицинский ассистент, который помогает собрать информацию для ответа на медицинский вопрос.
        Твоя задача - проанализировать вопрос пользователя и историю диалога (если есть) и определить, достаточно ли информации для ответа.
        
        Если информации недостаточно, задай ОДИН уточняющий вопрос, который поможет получить недостающие данные.
        Если информации достаточно, верни только слово "ДА".
        
        Вопрос пользователя: {user_message}
        
        История диалога:
        """
        
        # Добавляем историю диалога
        if history:
            for msg in history[-5:]:  # Берем последние 5 сообщений
                context += f"{msg['role']}: {msg['content']}\n"
        
        # Добавляем данные пациента
        if patient_data:
            context += f"\nДанные пациента:\n"
            if patient_data.get("name"):
                context += f"Имя: {patient_data['name']}\n"
            if patient_data.get("age"):
                context += f"Возраст: {patient_data['age']}\n"
            if patient_data.get("gender"):
                context += f"Пол: {patient_data['gender']}\n"
        
        # Добавляем информацию о количестве уточнений
        context += f"\nКоличество уже заданных уточняющих вопросов: {clarification_count}\n"
        
        context += """
        Отвечай только одним уточняющим вопросом, если информации недостаточно, или словом 'ДА', если достаточно.
        Не задавай более 3 уточняющих вопросов. Если уже было задано 3 вопроса, верни 'ДА'.
        """
        
        try:
            messages = [
                {"role": "system", "content": "Ты - медицинский ассистент, который помогает собрать информацию."},
                {"role": "user", "content": context}
            ]
            
            # Используем общий механизм выбора моделей по приоритету
            response, provider, metadata = await call_model_with_failover(
                messages=messages,
                system_prompt="Ты - медицинский ассистент, который помогает собрать информацию для ответа на медицинский вопрос.",
                model_type="text"
            )
            
            response = response.strip()
            
            if response == "ДА" or clarification_count >= self.max_clarifications:
                # Если данных достаточно и есть медицинские записи, переключаемся на ИИ-врача главного
                if has_medical_records:
                    ai_mode = "doctor"
                return True, None, ai_mode
            else:
                return False, response, ai_mode
        except Exception as e:
            logging.error(f"Ошибка в ClarificationAgent: {e}")
            # В случае ошибки считаем, что данных достаточно
            if has_medical_records:
                ai_mode = "doctor"
            return True, None, ai_mode

# Агент для анализа анализов на основе horizon-beta
class TestAnalysisAgent:
    def __init__(self):
        pass  # Убираем жестко заданную модель

    async def analyze_test_results(self, text: str) -> List[Dict[str, Any]]:
        """Анализ текста анализов и извлечение структурированных данных"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""Ты — медицинский эксперт по анализам. Извлеки из текста все результаты анализов в структурированном формате.
                    
                    ТЕКУЩАЯ ДАТА: {datetime.now().strftime('%d.%m.%Y')} (год: {datetime.now().year})
                    
                    Для каждого анализа укажи:
                    1. Название анализа (на русском)
                    2. Значение
                    3. Референсные значения (норма)
                    4. Единицы измерения
                    5. Дату анализа (если есть)
                    6. Отклонение от нормы (если есть)
                    
                    При извлечении возраста пациента учитывай текущую дату и корректируй возраст соответствующим образом.
                    
                    Верни ответ в формате JSON массива объектов:
                    [
                        {{
                            "test_name": "Название анализа",
                            "value": "Значение",
                            "reference_range": "Референсные значения",
                            "unit": "Единицы измерения",
                            "test_date": "ГГГГ-ММ-ДД",
                            "is_abnormal": true/false,
                            "notes": "Примечания"
                        }}
                    ]
                    Если даты нет, укажи null. Если референсные значения не указаны, укажи null."""
                },
                {
                    "role": "user",
                    "content": text[:4000]  # Ограничиваем длину текста
                }
            ]
            
            # Используем общий механизм выбора моделей по приоритету
            response_text, _, _ = await call_model_with_failover(
                messages=messages,
                system_prompt="Ты — медицинский эксперт по анализам. Извлеки из текста все результаты анализов в структурированном формате.",
                model_type="text"
            )
            
            # Извлекаем JSON из ответа
            import json
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    return data
                except json.JSONDecodeError:
                    pass
            return []
        except Exception as e:
            logging.error(f"Ошибка при анализе результатов анализов: {e}")
            return []

    async def get_test_summary(self, user_id: str, test_names: List[str] = None) -> str:
        """Получение сводки по анализам пациента"""
        try:
            # Проверяем кэш
            cache_key = f"summary_{user_id}_{'_'.join(test_names) if test_names else 'all'}"
            cached = supabase.table("doc_agent_cache").select("*").eq("user_id", user_id).eq("query",
                                                                                             cache_key).execute()
            if cached.data and datetime.now() < datetime.fromisoformat(cached.data[0]["expires_at"]):
                return cached.data[0]["result"]["summary"]

            # Получаем анализы из базы
            query = supabase.table("doc_test_results").select("*").eq("user_id", user_id)
            if test_names:
                # Фильтруем по названиям анализов
                conditions = []
                for name in test_names:
                    conditions.append(f"test_name.ilike.%{name}%")
                query = query.or_(*conditions)
            results = query.order("test_date", desc=True).limit(50).execute()

            if not results.data:
                return "У пациента нет сохраненных анализов."

            # Формируем текст для анализа
            tests_text = "Анализы пациента:\n"
            for test in results.data:
                tests_text += f"- {test['test_name']}: {test['value']} {test['unit'] or ''} (норма: {test['reference_range'] or 'не указана'}) от {test['test_date'] or 'дата не указана'}\n"
                if test.get('is_abnormal'):
                    tests_text += f"  Отклонение от нормы: {test.get('notes', 'есть')}\n"

            # Анализируем анализы
            messages = [
                {
                    "role": "system",
                    "content": """Ты — медицинский ассистент. Проанализируй предоставленные анализы и дай краткую сводку:
                    1. Выдели основные показатели и их значения
                    2. Укажи, какие показатели выходят за пределы нормы
                    3. Дай общую оценку состояния пациента
                    4. Рекомендуй дополнительные обследования или консультации специалистов, если необходимо
                    Отвечай кратко и по делу, на русском языке."""
                },
                {
                    "role": "user",
                    "content": tests_text
                }
            ]
            
            # Используем общий механизм выбора моделей по приоритету
            summary, _, _ = await call_model_with_failover(
                messages=messages,
                system_prompt="Ты — медицинский ассистент. Проанализируй предоставленные анализы и дай краткую сводку.",
                model_type="text"
            )

            # Сохраняем в кэш
            supabase.table("doc_agent_cache").insert({
                "user_id": user_id,
                "query": cache_key,
                "result": {"summary": summary},
                "expires_at": (datetime.now() + timedelta(hours=AGENT_CACHE_EXPIRE_HOURS)).isoformat()
            }).execute()

            return summary
        except Exception as e:
            logging.error(f"Ошибка при получении сводки анализов: {e}")
            return "Не удалось получить сводку анализов."

# Интеллектуальный агент для определения типа запроса
class IntelligentQueryAnalyzer:
    def __init__(self):
        self.analysis_patterns = [
            "анализ", "анализы", "результат", "показатель", "кровь", "моча", 
            "биохимия", "общий анализ", "тест", "лаборатория", "норма", "значение",
            "anti-hev", "anti-hev igg", "гепатит", "igg", "igm", "антитела"
        ]
        
        # Паттерны для конкретных показателей
        self.specific_indicator_patterns = [
            "какие у меня", "что показывает", "мой", "мои", "результаты", "значения",
            "показатели", "анализы по", "тест на", "уровень", "концентрация"
        ]
    
    async def analyze_query_type(self, question: str, user_id: str) -> Dict[str, Any]:
        """
        Интеллектуально анализирует тип запроса пользователя.
        Возвращает словарь с информацией о типе запроса.
        """
        try:
            logging.info(f"Анализирую тип запроса: {question} для пользователя: {user_id}")
            
            # Получаем медицинские записи пользователя
            from database import get_medical_records
            medical_records = get_medical_records(user_id)
            has_medical_data = len(medical_records) > 0
            
            logging.info(f"Найдено медицинских записей: {len(medical_records)}")
            
            # Базовый анализ вопроса
            question_lower = question.lower()
            
            # Определяем, является ли это вопросом об анализах
            is_analysis_question = any(pattern in question_lower for pattern in self.analysis_patterns)
            
            # Определяем, спрашивает ли пользователь о конкретных показателях
            is_specific_indicator_question = await self._is_specific_indicator_question(question, medical_records)
            
            # Определяем, нужен ли режим врача
            needs_doctor_mode = await self._needs_doctor_mode(question, medical_records)
            
            result = {
                "is_analysis_question": is_analysis_question,
                "is_specific_indicator_question": is_specific_indicator_question,
                "needs_doctor_mode": needs_doctor_mode,
                "has_medical_data": has_medical_data,
                "medical_records": medical_records
            }
            
            logging.info(f"Результат анализа: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при анализе типа запроса: {e}")
            return {
                "is_analysis_question": False,
                "is_specific_indicator_question": False,
                "needs_doctor_mode": False,
                "has_medical_data": False,
                "medical_records": []
            }
    
    async def _is_specific_indicator_question(self, question: str, medical_records: List[Dict]) -> bool:
        """
        Определяет, спрашивает ли пользователь о конкретных показателях.
        Использует комбинацию паттернов и ИИ для надежного определения.
        """
        try:
            question_lower = question.lower()
            
            # Сначала проверяем по паттернам - это быстрее и надежнее
            has_specific_patterns = any(pattern in question_lower for pattern in self.specific_indicator_patterns)
            has_analysis_patterns = any(pattern in question_lower for pattern in self.analysis_patterns)
            
            logging.info(f"Проверка паттернов для вопроса: {question}")
            logging.info(f"Специфичные паттерны: {has_specific_patterns}, паттерны анализов: {has_analysis_patterns}")
            
            # Если есть паттерны конкретных показателей И паттерны анализов - это точно вопрос об анализах
            if has_specific_patterns and has_analysis_patterns:
                logging.info("Вопрос определен как конкретный показатель по паттернам")
                return True
            
            # Если нет медицинских записей, но есть паттерны анализов - это тоже вопрос об анализах
            if not medical_records and has_analysis_patterns:
                logging.info("Вопрос определен как анализ (нет медицинских записей, но есть паттерны)")
                return True
            
            # Если есть медицинские записи, используем ИИ для дополнительной проверки
            if medical_records:
                logging.info("Использую ИИ для дополнительной проверки типа вопроса")
                return await self._ai_check_specific_question(question, medical_records)
            
            logging.info("Вопрос не определен как конкретный показатель")
            return False
            
        except Exception as e:
            logging.error(f"Ошибка при определении типа вопроса: {e}")
            # В случае ошибки, если есть паттерны анализов, считаем что это вопрос об анализах
            question_lower = question.lower()
            fallback_result = any(pattern in question_lower for pattern in self.analysis_patterns)
            logging.info(f"Fallback результат: {fallback_result}")
            return fallback_result
    
    async def _needs_doctor_mode(self, question: str, medical_records: List[Dict]) -> bool:
        """
        Определяет, нужен ли режим врача для ответа на вопрос.
        """
        try:
            if not medical_records:
                return False
            
            # Формируем контекст для анализа
            context = "Доступные медицинские записи пациента:\n"
            for record in medical_records[:2]:
                if record.get('record_type') in ['analysis', 'image_analysis']:
                    content = record.get('content', '')[:300]
                    context += f"- {content[:100]}...\n"
            
            # Используем ИИ для определения сложности вопроса
            messages = [
                {
                    "role": "system",
                    "content": """Определи, нужен ли режим врача для ответа на вопрос пользователя.
                    
                    Верни "doctor" если:
                    - Вопрос требует глубокой медицинской экспертизы
                    - Нужно интерпретировать сложные анализы
                    - Вопрос требует постановки предположительного диагноза
                    - Нужно дать медицинские рекомендации
                    
                    Верни "assistant" если:
                    - Вопрос простой и требует только извлечения данных
                    - Нужно просто предоставить информацию из анализов
                    - Вопрос не требует сложной медицинской интерпретации
                    - Нужно только показать результаты
                    """
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nДоступные медицинские данные: {context}"
                }
            ]
            
            response, _, _ = await call_model_with_failover(
                messages=messages,
                system_prompt="Определи режим работы ИИ",
                model_type="text"
            )
            
            return "doctor" in response.lower()
            
        except Exception as e:
            logging.error(f"Ошибка при определении режима ИИ: {e}")
            return False
    
    async def get_relevant_medical_context(self, question: str, medical_records: List[Dict]) -> str:
        """
        Извлекает релевантный медицинский контекст на основе вопроса.
        Улучшенная версия с более надежным поиском.
        """
        try:
            if not medical_records:
                logging.info("get_relevant_medical_context: нет медицинских записей")
                return ""
            
            logging.info(f"get_relevant_medical_context: анализирую {len(medical_records)} записей для вопроса: {question}")
            
            question_lower = question.lower()
            context_parts = []
            
            # Сначала ищем по ключевым словам - это быстрее и надежнее
            for record in medical_records:
                if record.get('record_type') in ['analysis', 'image_analysis']:
                    content = record.get('content', '')
                    content_lower = content.lower()
                    
                    # Проверяем, содержит ли запись ключевые слова из вопроса
                    question_words = [word for word in question_lower.split() if len(word) > 3]
                    has_relevant_keywords = any(word in content_lower for word in question_words)
                    
                    logging.info(f"Проверяю запись {record.get('record_type')}: ключевые слова {question_words}, найдены: {has_relevant_keywords}")
                    
                    # Если есть ключевые слова, добавляем запись
                    if has_relevant_keywords:
                        context_parts.append(content)
                        logging.info(f"Добавлена запись по ключевым словам: {len(content)} символов")
                        continue
                    
                    # Если нет ключевых слов, но есть паттерны анализов, проверяем с помощью ИИ
                    if any(pattern in question_lower for pattern in self.analysis_patterns):
                        logging.info("Проверяю релевантность с помощью ИИ")
                        is_relevant = await self._ai_check_relevance(question, content)
                        if is_relevant:
                            context_parts.append(content)
                            logging.info(f"Добавлена запись по ИИ-анализу: {len(content)} символов")
            
            # Если не нашли по ключевым словам, используем ИИ для всех записей
            if not context_parts:
                logging.info("Не найдено по ключевым словам, проверяю все записи с помощью ИИ")
                for record in medical_records:
                    if record.get('record_type') in ['analysis', 'image_analysis']:
                        content = record.get('content', '')
                        is_relevant = await self._ai_check_relevance(question, content)
                        if is_relevant:
                            context_parts.append(content)
                            logging.info(f"Добавлена запись по ИИ-анализу (второй проход): {len(content)} символов")
            
            # Объединяем релевантные записи
            if context_parts:
                full_context = "\n\n📊 ВАШИ МЕДИЦИНСКИЕ ДАННЫЕ:\n"
                for i, part in enumerate(context_parts, 1):
                    full_context += f"\n--- Запись {i} ---\n{part}\n"
                
                # Ограничиваем общую длину контекста
                if len(full_context) > 8000:
                    full_context = full_context[:8000] + "\n\n... (данные обрезаны из-за ограничений длины)"
                
                logging.info(f"Создан контекст из {len(context_parts)} записей, общая длина: {len(full_context)} символов")
                return full_context
            
            logging.info("Релевантные записи не найдены")
            return ""
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении медицинского контекста: {e}")
            return ""
    
    async def _ai_check_relevance(self, question: str, content: str) -> bool:
        """Проверка релевантности медицинской записи с помощью ИИ"""
        try:
            logging.info(f"Проверяю релевантность записи длиной {len(content)} символов для вопроса: {question}")
            
            messages = [
                {
                    "role": "system",
                    "content": """Определи, содержит ли эта медицинская запись информацию, релевантную вопросу пользователя.
                    
                    Верни "ДА" если запись содержит:
                    - Упоминание анализов или показателей из вопроса
                    - Данные, которые могут помочь ответить на вопрос
                    - Результаты, о которых спрашивает пользователь
                    
                    Верни "НЕТ" если запись не релевантна вопросу."""
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nМедицинская запись: {content[:1000]}"
                }
            ]
            
            relevance_response, _, _ = await call_model_with_failover(
                messages=messages,
                system_prompt="Определи релевантность медицинской записи",
                model_type="text"
            )
            
            is_relevant = "ДА" in relevance_response.strip().upper()
            logging.info(f"ИИ определил релевантность: {is_relevant} (ответ: {relevance_response.strip()})")
            
            return is_relevant
            
        except Exception as e:
            logging.error(f"Ошибка при проверке релевантности: {e}")
            return False
    
    async def _ai_check_specific_question(self, question: str, medical_records: List[Dict]) -> bool:
        """Дополнительная проверка с помощью ИИ"""
        try:
            logging.info(f"ИИ-проверка типа вопроса: {question}")
            
            # Формируем контекст для анализа
            context = "Доступные медицинские записи пациента:\n"
            for record in medical_records[:3]:  # Берем первые 3 записи для анализа
                if record.get('record_type') in ['analysis', 'image_analysis']:
                    content = record.get('content', '')[:500]  # Ограничиваем длину
                    context += f"- {content[:100]}...\n"
            
            # Если контекст пуст, возвращаем False
            if "Доступные медицинские записи пациента:\n" == context:
                logging.info("Контекст пуст, возвращаю False")
                return False
            
            logging.info(f"Контекст для ИИ-анализа: {len(context)} символов")
            
            # Используем ИИ для определения типа вопроса
            messages = [
                {
                    "role": "system",
                    "content": """Ты - медицинский ассистент. Определи, спрашивает ли пользователь о конкретных показателях или результатах анализов.
                    
                    Верни только "ДА" если вопрос:
                    - Содержит упоминание конкретных анализов, тестов или показателей
                    - Спрашивает о результатах, значениях или нормах
                    - Относится к данным из медицинских записей пациента
                    
                    Верни "НЕТ" если вопрос:
                    - Общий медицинский вопрос без упоминания конкретных показателей
                    - Запрашивает общую медицинскую информацию
                    - Не относится к конкретным данным пациента
                    
                    Примеры вопросов, на которые нужно ответить "ДА":
                    - "Какие у меня анализы по anti-HEV IgG?"
                    - "Что показывает мой гемоглобин?"
                    - "Мой сахар в норме?"
                    - "Покажи результаты моих анализов"
                    
                    Примеры вопросов, на которые нужно ответить "НЕТ":
                    - "Что такое гепатит?"
                    - "Какие бывают виды анализов крови?"
                    - "Как питаться при диабете?"
                    - "Объясни, что такое аллергия"
                    """
                },
                {
                    "role": "user",
                    "content": f"Вопрос пользователя: {question}\n\n{context}"
                }
            ]
            
            response, _, _ = await call_model_with_failover(
                messages=messages,
                system_prompt="Определи тип вопроса пользователя",
                model_type="text"
            )
            
            is_specific = "ДА" in response.strip().upper()
            logging.info(f"ИИ определил тип вопроса: {is_specific} (ответ: {response.strip()})")
            
            return is_specific
            
        except Exception as e:
            logging.error(f"Ошибка при ИИ-проверке типа вопроса: {e}")
            # В случае ошибки ИИ, возвращаем True если есть паттерны анализов
            question_lower = question.lower()
            fallback_result = any(pattern in question_lower for pattern in self.analysis_patterns)
            logging.info(f"Fallback результат при ошибке ИИ: {fallback_result}")
            return fallback_result
