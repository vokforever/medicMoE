import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Импорты из наших модулей
from config import bot_token, supabase
from models import call_model_with_failover, reset_provider_blocks
from agents import ClarificationAgent, TestAnalysisAgent, IntelligentQueryAnalyzer
from database import (
    generate_user_uuid, create_patient_profile, get_patient_profile, save_medical_record, get_user_successful_responses,
    delete_test_result, delete_all_test_results, delete_test_results_by_period, delete_test_results_before_date
)
from utils import (
    escape_html, escape_markdown, search_medical_sources, analyze_image, extract_text_from_pdf,
    check_duplicate_medical_record_ai_enhanced
)
from keyboards import (
    get_feedback_keyboard, get_main_keyboard, get_manage_tests_keyboard, 
    get_delete_test_keyboard, get_delete_medical_record_keyboard, get_confirm_delete_keyboard, 
    get_confirm_delete_all_keyboard, get_confirm_delete_medical_record_keyboard,
    get_confirm_delete_all_medical_records_keyboard, get_date_range_keyboard, 
    get_confirm_delete_period_keyboard
)

# Импорт и инициализация агента для структурированных данных
from structured_tests_agent import TestExtractionAgent

# Класс для управления сессиями
class SessionManager:
    """Менеджер сессий для управления контекстом пользователей"""
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.active_sessions = {}  # user_id: session_data
        self.max_history_length = 50  # Максимальное количество сообщений в истории
        
    async def load_session_history(self, user_id: str) -> List[Dict]:
        """Загрузка истории диалога из Supabase"""
        try:
            logging.info(f"Загрузка истории диалога для пользователя: {user_id}")
            
            response = self.supabase.table("doc_conversation_history").select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(self.max_history_length) \
                .execute()
            
            history = response.data if response.data else []
            logging.info(f"Загружено {len(history)} сообщений из истории")
            
            # Сортируем по времени создания (от старых к новым)
            history.sort(key=lambda x: x.get("created_at", ""))
            
            return history
            
        except Exception as e:
            logging.error(f"Ошибка загрузки истории: {e}")
            return []
    
    async def save_session_message(self, user_id: str, message: Dict):
        """Сохранение сообщения в историю"""
        try:
            logging.info(f"Сохранение сообщения в историю для пользователя: {user_id}")
            
            # Подготавливаем данные для сохранения
            message_data = {
                "user_id": user_id,
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
                "message_type": message.get("type", "text"),
                "created_at": datetime.now().isoformat()
            }
            
            # Сохраняем в базу
            self.supabase.table("doc_conversation_history").insert(message_data).execute()
            
            # Обновляем активную сессию
            if user_id not in self.active_sessions:
                self.active_sessions[user_id] = {"history": [], "context": {}}
            
            self.active_sessions[user_id]["history"].append(message_data)
            
            # Ограничиваем размер истории в памяти
            if len(self.active_sessions[user_id]["history"]) > self.max_history_length:
                self.active_sessions[user_id]["history"] = self.active_sessions[user_id]["history"][-self.max_history_length:]
            
            logging.info("Сообщение успешно сохранено в историю")
            
        except Exception as e:
            logging.error(f"Ошибка сохранения сообщения: {e}")
    
    async def get_session_context(self, user_id: str) -> str:
        """Получение полного контекста сессии"""
        try:
            logging.info(f"Формирование контекста сессии для пользователя: {user_id}")
            
            # Загружаем историю, если её нет в памяти
            if user_id not in self.active_sessions or not self.active_sessions[user_id].get("history"):
                history = await self.load_session_history(user_id)
                if user_id not in self.active_sessions:
                    self.active_sessions[user_id] = {"history": [], "context": {}}
                self.active_sessions[user_id]["history"] = history
            
            history = self.active_sessions[user_id]["history"]
            
            if not history:
                logging.info("История диалога пуста")
                return "История диалога пуста."
            
            # Формируем контекст из последних сообщений
            recent_messages = history[-10:]  # Берем последние 10 сообщений
            
            context = "История диалога:\n"
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]  # Ограничиваем длину
                context += f"{role}: {content}\n"
            
            logging.info(f"Сформирован контекст длиной {len(context)} символов")
            return context
            
        except Exception as e:
            logging.error(f"Ошибка формирования контекста сессии: {e}")
            return "Ошибка формирования контекста сессии."
    
    async def get_user_profile_context(self, user_id: str) -> str:
        """Получение контекста профиля пользователя"""
        try:
            logging.info(f"Получение контекста профиля для пользователя: {user_id}")
            
            response = self.supabase.table("doc_patient_profiles").select("*").eq("user_id", user_id).execute()
            
            if response.data:
                profile = response.data[0]
                context = f"Профиль пациента: {profile.get('name', 'Не указан')}, "
                context += f"возраст: {profile.get('age', 'Не указан')}, "
                context += f"пол: {profile.get('gender', 'Не указан')}"
                
                if profile.get('birth_date'):
                    context += f", дата рождения: {profile.get('birth_date')}"
                
                logging.info(f"Контекст профиля сформирован: {context}")
                return context
            else:
                logging.info("Профиль пациента не найден")
                return "Профиль пациента не найден."
                
        except Exception as e:
            logging.error(f"Ошибка получения контекста профиля: {e}")
            return "Ошибка загрузки профиля пациента."
    
    async def get_medical_records_context(self, user_id: str) -> str:
        """Получение контекста медицинских записей пользователя"""
        try:
            logging.info(f"Получение контекста медицинских записей для пользователя: {user_id}")
            
            response = self.supabase.table("doc_medical_records").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(5).execute()
            
            if response.data:
                records = response.data
                context = f"Медицинские записи: найдено {len(records)} записей\n"
                
                for i, record in enumerate(records[:3]):  # Показываем только последние3
                    record_type = record.get("record_type", "неизвестно")
                    created_at = record.get("created_at", "")
                    content = record.get("content", "")[:300]  # Ограничиваем длину
                    
                    context += f"\n--- Запись {i+1} ---\n"
                    context += f"Тип: {record_type}\n"
                    context += f"Дата: {created_at}\n"
                    context += f"Содержание: {content}\n"
                
                logging.info(f"Контекст медицинских записей сформирован: {len(context)} символов")
                return context
            else:
                logging.info("Медицинские записи не найдены")
                return "Медицинские записи не найдены."
                
        except Exception as e:
            logging.error(f"Ошибка получения контекста медицинских записей: {e}")
            return "Ошибка загрузки медицинских записей."
    
    async def update_session_context(self, user_id: str, context_data: Dict[str, Any]):
        """Обновление контекста сессии"""
        try:
            logging.info(f"Обновление контекста сессии для пользователя: {user_id}")
            
            if user_id not in self.active_sessions:
                self.active_sessions[user_id] = {"history": [], "context": {}}
            
            # Обновляем контекст
            self.active_sessions[user_id]["context"].update(context_data)
            
            logging.info(f"Контекст сессии обновлен: {list(context_data.keys())}")
            
        except Exception as e:
            logging.error(f"Ошибка обновления контекста сессии: {e}")

# Класс улучшенной RAG системы
class EnhancedRAGSystem:
    """Улучшенная RAG система с управлением контекстом"""
    
    def __init__(self, session_manager: SessionManager, supabase_client):
        self.session_manager = session_manager
        self.supabase = supabase_client
        self.max_context_length = 4000  # Максимальная длина контекста для модели
        
    async def get_enhanced_context(self, user_id: str, query: str) -> str:
        """Получение расширенного контекста для ответа"""
        try:
            logging.info(f"Формирование расширенного контекста для пользователя: {user_id}")
            
            # 1. Получаем историю диалога
            conversation_context = await self.session_manager.get_session_context(user_id)
            
            # 2. Получаем профиль и медицинские записи
            profile_context = await self.session_manager.get_user_profile_context(user_id)
            medical_context = await self.session_manager.get_medical_records_context(user_id)
            
            # 3. Ищем релевантные документы в базе знаний
            knowledge_context = await self._search_knowledge_base(query)
            
            # 4. Ищем в медицинских источниках
            medical_sources_context = await self._search_medical_sources(query)
            
            # 5. Объединяем контексты
            enhanced_context = f"""
{profile_context}

{medical_context}

{conversation_context}

Релевантная информация из базы знаний:
{knowledge_context}

Медицинские источники:
{medical_sources_context}
            """.strip()
            
            # Ограничиваем длину контекста
            if len(enhanced_context) > self.max_context_length:
                enhanced_context = enhanced_context[:self.max_context_length] + "..."
            
            logging.info(f"Расширенный контекст сформирован: {len(enhanced_context)} символов")
            return enhanced_context
            
        except Exception as e:
            logging.error(f"Ошибка формирования расширенного контекста: {e}")
            return "Ошибка формирования контекста."
    
    async def _search_knowledge_base(self, query: str) -> str:
        """Поиск в базе знаний"""
        try:
            logging.info(f"Поиск в базе знаний для запроса: {query}")
            
            # Ищем в структурированных тестах
            test_results = await self._search_test_results(query)
            
            # Ищем в векторной базе знаний
            vector_results = await self._search_vector_knowledge(query)
            
            context = ""
            if test_results:
                context += f"Результаты анализов:\n{test_results}\n\n"
            
            if vector_results:
                context += f"База знаний:\n{vector_results}\n\n"
            
            if not context:
                context = "Релевантной информации в базе знаний не найдено."
            
            return context
            
        except Exception as e:
            logging.error(f"Ошибка поиска в базе знаний: {e}")
            return "Ошибка поиска в базе знаний."
    
    async def _search_test_results(self, query: str) -> str:
        """Поиск результатов анализов по запросу"""
        try:
            # Ищем ключевые слова в названиях анализов
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return ""
            
            # Формируем поисковый запрос
            search_query = " OR ".join([f"test_name.ilike.%{kw}%" for kw in keywords])
            
            # Выполняем поиск
            response = self.supabase.table("doc_structured_test_results").select("*").or_(search_query).limit(10).execute()
            
            if not response.data:
                return ""
            
            # Форматируем результаты
            results = []
            for test in response.data:
                test_name = test.get("test_name", "Не указан")
                result = test.get("result", "Не указан")
                ref_values = test.get("reference_values", "Не указаны")
                units = test.get("units", "")
                
                result_text = f"• {test_name}: {result}"
                if units:
                    result_text += f" {units}"
                if ref_values and ref_values != "Не указаны":
                    result_text += f" (норма: {ref_values})"
                
                results.append(result_text)
            
            return "\n".join(results)
            
        except Exception as e:
            logging.error(f"Ошибка поиска результатов анализов: {e}")
            return ""
    
    async def _search_vector_knowledge(self, query: str) -> str:
        """Поиск в векторной базе знаний"""
        try:
            # Здесь будет реализован векторный поиск
            # Пока возвращаем пустой результат
            return ""
            
        except Exception as e:
            logging.error(f"Ошибка векторного поиска: {e}")
            return ""
    
    async def _search_medical_sources(self, query: str) -> str:
        """Поиск в медицинских источниках"""
        try:
            logging.info(f"Поиск в медицинских источниках для запроса: {query}")
            
            # Используем существующую функцию поиска
            results = await search_medical_sources(query)
            
            if results:
                return f"Найдено {len(results)} релевантных источников:\n" + "\n".join(results[:3])
            else:
                return "Релевантных медицинских источников не найдено."
                
        except Exception as e:
            logging.error(f"Ошибка поиска в медицинских источниках: {e}")
            return "Ошибка поиска в медицинских источниках."
    
    async def _extract_keywords(self, query: str) -> List[str]:
        """Извлечение ключевых слов из запроса с использованием LLM"""
        try:
            # Используем медицинский агент для извлечения ключевых слов
            from medical_terms_agent import medical_terms_agent
            
            medical_keywords = await medical_terms_agent.extract_medical_keywords(query)
            
            # Если медицинские термины не найдены, используем общие слова
            if not medical_keywords:
                words = query.lower().split()
                medical_keywords = [word for word in words if len(word) > 3][:3]
            
            logging.info(f"Извлечены ключевые слова: {medical_keywords}")
            return medical_keywords
            
        except Exception as e:
            logging.error(f"Ошибка извлечения ключевых слов: {e}")
            # Fallback к простому методу
            words = query.lower().split()
            return [word for word in words if len(word) > 3][:3]
    
    async def process_query(self, user_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """Обработка запроса с полным контекстом"""
        try:
            logging.info(f"Обработка запроса для пользователя: {user_id}")
            
            # 1. Сохраняем пользовательский запрос
            await self.session_manager.save_session_message(user_id, {
                "role": "user",
                "content": query,
                "type": "text"
            })
            
            # 2. Получаем расширенный контекст
            context = await self.get_enhanced_context(user_id, query)
            
            # 3. Генерируем ответ с помощью ИИ
            response = await self._generate_ai_response(query, context, user_id)
            
            # 4. Сохраняем ответ ассистента
            await self.session_manager.save_session_message(user_id, {
                "role": "assistant",
                "content": response,
                "type": "text"
            })
            
            # 5. Обновляем контекст сессии
            await self.session_manager.update_session_context(user_id, {
                "last_query": query,
                "last_response": response,
                "last_interaction": datetime.now().isoformat()
            })
            
            logging.info("Запрос успешно обработан")
            return response, {"context_length": len(context), "success": True}
            
        except Exception as e:
            logging.error(f"Ошибка обработки запроса: {e}")
            error_response = "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз."
            return error_response, {"error": str(e), "success": False}
    
    async def _generate_ai_response(self, query: str, context: str, user_id: str) -> str:
        """Генерация ответа с помощью ИИ"""
        try:
            logging.info(f"Генерация ИИ-ответа для запроса длиной {len(query)} символов")
            
            # Формируем системный промпт
            system_prompt = f"""Ты — ИИ-ассистент врача. Твоя задача — помогать пользователям с медицинскими вопросами, 
            анализировать их анализы и предоставлять информацию о здоровье. Отвечай максимально точно и информативно, 
            используя предоставленный контекст. Учитывай историю диалога и данные пациента, если они доступны.
            
            ТЕКУЩАЯ ДАТА: {datetime.now().strftime('%d.%m.%Y')} (год: {datetime.now().year})
            
            ВАЖНО: 
            - Ты не ставишь диагноз и не заменяешь консультацию врача
            - Всегда рекомендуй консультацию со специалистом для точной диагностики и лечения
            - Если в контексте есть точный ответ из авторитетных медицинских источников — используй его
            - Всегда указывай источник информации, если он известен
            - Отвечай на русском языке
            - Структурируй ответ с использованием эмодзи для лучшего восприятия
            - При работе с возрастом пациента учитывай текущую дату и корректируй возраст
            
            КОНТЕКСТ ПОЛЬЗОВАТЕЛЯ:
            {context}
            
            ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
            {query}
            
            Сформируй подробный и полезный ответ, используя всю доступную информацию."""
            
            # Вызываем ИИ-модель
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            ai_response = await call_model_with_failover(
                messages=messages,
                model_type="text"
            )
            
            if ai_response and isinstance(ai_response, tuple):
                response_text = ai_response[0]
                logging.info(f"Получен ИИ-ответ длиной {len(response_text)} символов")
                return response_text
            else:
                logging.warning("ИИ-модель не вернула ответ")
                return "Извините, не удалось сгенерировать ответ. Попробуйте переформулировать вопрос."
                
        except Exception as e:
            logging.error(f"Ошибка генерации ИИ-ответа: {e}")
            return "Извините, произошла ошибка при генерации ответа. Попробуйте еще раз."

# Инициализация компонентов
session_manager = SessionManager(supabase)
enhanced_rag_system = EnhancedRAGSystem(session_manager, supabase)
structured_test_agent = TestExtractionAgent(supabase)

# Инициализация бота и диспетчера
bot = Bot(token=bot_token)
dp = Dispatcher()
scheduler = AsyncIOScheduler()

# Инициализация агентов
clarification_agent = ClarificationAgent()
test_agent = TestAnalysisAgent()
intelligent_analyzer = IntelligentQueryAnalyzer()

# Состояния для FSM
class DoctorStates(StatesGroup):
    waiting_for_feedback = State()
    waiting_for_clarification = State()
    waiting_for_file = State()
    waiting_for_patient_id = State()
    viewing_history = State()
    confirming_profile = State()
    updating_profile = State()
    waiting_for_test_data = State()  # Ожидание дополнения данных анализов
    managing_tests = State()  # Состояние управления анализами
    confirming_delete = State()  # Состояние подтверждения удаления
    confirming_delete_all = State()  # Состояние подтверждения удаления всех
    choosing_delete_period = State()  # Состояние выбора периода удаления
    waiting_for_date = State()  # Ожидание ввода даты для удаления

# Функция для генерации ответа с failover между провайдерами
async def generate_answer_with_failover(
        question: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        patient_data: Dict[str, Any] = None,
        user_id: int = None,
        system_prompt: str = None,
        model_type: str = "text"
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Генерирует ответ с использованием failover между провайдерами и моделями.
    Возвращает кортеж: (ответ, провайдер, дополнительная информация)
    """
    logging.info(f"Генерация ответа с failover для вопроса: {question[:100]}...")
    logging.info(f"Тип модели: {model_type}, длина контекста: {len(context)} символов")
    logging.info(f"История диалога: {len(history) if history else 0} сообщений")
    logging.info(f"Данные пациента: {'есть' if patient_data else 'нет'}")
    
    # Используем переданный системный промпт или стандартный
    if system_prompt is None:
        system_prompt = f"""Ты — ИИ-ассистент врача. Твоя задача — помогать пользователям с медицинскими вопросами, 
        анализировать их анализы и предоставлять информацию о здоровье. Отвечай максимально точно и информативно, 
        используя предоставленный контекст. Учитывай историю диалога и данные пациента, если они доступны.
        
        ТЕКУЩАЯ ДАТА: {datetime.now().strftime('%d.%m.%Y')} (год: {datetime.now().year})
        
        ВАЖНО: Ты не ставишь диагноз и не заменяешь консультацию врача. Всегда рекомендуй консультацию 
        со специалистом для точной диагностики и лечения.
        Если в контексте есть точный ответ из авторитетных медицинских источников — используй его.
        Всегда указывай источник информации, если он известен.
        Отвечай на русском языке.
        Структурируй ответ с использованием эмодзи для лучшего восприятия.
        
        При работе с возрастом пациента учитывай текущую дату и корректируй возраст соответствующим образом."""
        logging.info("Используется стандартный системный промпт")
    else:
        logging.info("Используется переданный системный промпт")
    
    # Формируем сообщения для модели
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    if context:
        messages.append({"role": "system", "content": f"Медицинская информация:\n{context}"})
        logging.info("Добавлен медицинский контекст в сообщения")

    # Добавляем информацию о пациенте
    if patient_data:
        patient_info = f"Информация о пациенте:\n"
        if patient_data.get("name"):
            patient_info += f"Имя: {patient_data['name']}\n"
        if patient_data.get("age"):
            patient_info += f"Возраст: {patient_data['age']}\n"
        if patient_data.get("gender"):
            patient_info += f"Пол: {patient_data['gender']}\n"
        messages.append({"role": "system", "content": patient_info})
        logging.info("Добавлена информация о пациенте в сообщения")

    # Добавляем историю диалога
    if history:
        from config import MAX_CONTEXT_MESSAGES
        recent_history = history[-MAX_CONTEXT_MESSAGES:] if len(history) > MAX_CONTEXT_MESSAGES else history
        for msg in recent_history:
            messages.append(msg)
        logging.info(f"Добавлена история диалога: {len(recent_history)} сообщений")

    messages.append({"role": "user", "content": question})
    logging.info(f"Всего сообщений для модели: {len(messages)}")

    # Используем универсальную функцию с failover
    logging.info("Вызываю call_model_with_failover")
    return await call_model_with_failover(
        messages=messages,
        system_prompt=system_prompt,
        model_type=model_type
    )

# Функция для генерации ответа с MOE подходом (старая версия, оставлена для совместимости)
async def generate_answer(question: str, context: str = "", history: List[Dict[str, str]] = None,
                          patient_data: Dict[str, Any] = None, user_id: int = None) -> str:
    answer, _, _ = await generate_answer_with_failover(question, context, history, patient_data, user_id)
    return answer

# Функция для очистки состояния
async def clear_conversation_state(state: FSMContext, chat_id: int):
    try:
        logging.info(f"Очистка состояния для чата {chat_id}")
        
        # Удаляем напоминание
        try:
            scheduler.remove_job(f"reminder_{chat_id}")
            logging.info(f"Напоминание для чата {chat_id} удалено")
        except Exception as e:
            logging.warning(f"Не удалось удалить напоминание для чата {chat_id}: {e}")
        
        # Очищаем состояние
        await state.clear()
        logging.info(f"Состояние для чата {chat_id} очищено")
        
    except Exception as e:
        logging.error(f"Ошибка при очистке состояния для чата {chat_id}: {e}")

# Функция для отложенного напоминания
async def send_reminder(chat_id: int):
    try:
        logging.info(f"Отправка напоминания в чат {chat_id}")
        
        await bot.send_message(
            chat_id,
            "🔔 Напоминаю: помог ли вам мой предыдущий ответ?",
            reply_markup=get_feedback_keyboard()
        )
        
        logging.info(f"Напоминание успешно отправлено в чат {chat_id}")
        
    except Exception as e:
        logging.error(f"Ошибка при отправке напоминания в чат {chat_id}: {e}")

# Обработчик команды /start
@dp.message(Command("start"))
async def start_command(message: types.Message, state: FSMContext):
    logging.info(f"Команда /start от пользователя {message.from_user.id}")
    
    await clear_conversation_state(state, message.chat.id)
    profile = get_patient_profile(generate_user_uuid(message.from_user.id))
    
    if profile:
        logging.info(f"Профиль пациента найден: {profile.get('name', 'N/A')}")
        await message.answer(
            f"👋 Здравствуйте, {profile['name']}! Я ваш ИИ-ассистент врача.\n\n"
            f"📊 Я могу помочь вам с анализом анализов, ответить на медицинские вопросы и хранить ваш анамнез.\n\n"
            f"💡 Просто задайте ваш вопрос, или загрузите анализы для анализа.\n\n"
                         f"📊 Доступные команды:\n"
             f"• /start - перезапуск бота\n"
             f"• /help - справка по использованию\n"
             f"• /profile - управление профилем пациента\n"
             f"• /tests - загрузка и анализ анализов\n"
             f"• /history - история диалогов\n"
             f"• /manage_tests - управление анализами (удаление)\n\n"
             f"🔍 Что вас интересует?",
            reply_markup=get_main_keyboard()
        )
    else:
        logging.info("Профиль пациента не найден, предлагаю создать")
        await message.answer(
            "👋 Здравствуйте! Я ваш ИИ-ассистент врача.\n\n"
            f"📊 Я могу помочь вам с анализом анализов, ответить на медицинские вопросы и хранить ваш анамнез.\n\n"
            f"💡 Для начала работы создайте профиль пациента или загрузите анализы для анализа.\n\n"
                         f"📊 Доступные команды:\n"
             f"• /start - перезапуск бота\n"
             f"• /help - справка по использованию\n"
             f"• /profile - создание профиля пациента\n"
             f"• /tests - загрузка и анализ анализов\n"
             f"• /manage_tests - управление анализами (удаление)\n\n"
            f"🔍 Что вас интересует?",
            reply_markup=get_main_keyboard()
        )

# Обработчик команды /manage_tests
@dp.message(Command("manage_tests"))
async def manage_tests_command(message: types.Message, state: FSMContext):
    """Команда для управления анализами"""
    try:
        from database import get_latest_test_results, get_medical_records
        user_id = generate_user_uuid(message.from_user.id)
        logging.info(f"Команда управления анализами от пользователя {message.from_user.id}")
        
        # Получаем последние анализы пользователя
        tests = get_latest_test_results(user_id, limit=10)
        
        # Получаем медицинские записи (включая неудачные анализы изображений)
        medical_records = get_medical_records(user_id, "image_analysis")
        
        if not tests and not medical_records:
            await message.answer(
                "📊 У вас пока нет сохраненных анализов.\n\n"
                "Загрузите анализы через фото или PDF файлы, чтобы начать работу с ними.",
                reply_markup=get_main_keyboard()
            )
            return
        
        await state.set_state(DoctorStates.managing_tests)
        
        # Формируем сообщение со списком анализов
        response_text = "📊 **Ваши последние данные:**\n\n"
        
        # Добавляем успешные анализы
        if tests:
            response_text += "🔬 **Структурированные анализы:**\n"
            for i, test in enumerate(tests):
                test_name = test.get("test_name", "Неизвестный анализ")
                test_date = test.get("created_at", "")[:10] if test.get("created_at") else "Не указана"
                result = test.get("result", "Не указан")
                
                response_text += f"**{i+1}. {test_name}**\n"
                response_text += f"📅 Дата: {test_date}\n"
                response_text += f"🔬 Результат: {result}\n\n"
        
        # Добавляем медицинские записи (включая неудачные попытки)
        if medical_records:
            response_text += "📋 **Медицинские записи (включая изображения):**\n"
            for i, record in enumerate(medical_records[:5]):  # Показываем последние 5
                content = record.get("content", "")
                created_at = record.get("created_at", "")[:10] if record.get("created_at") else "Не указана"
                record_id = record.get("id", "N/A")
                
                # Определяем тип записи по содержимому
                if "не удалось извлечь" in content.lower() or len(content.strip()) < 100:
                    record_type = "❌ Неудачный анализ"
                else:
                    record_type = "✅ Успешный анализ"
                
                # Обрезаем контент для отображения
                display_content = content[:100] + "..." if len(content) > 100 else content
                
                response_text += f"**{i+1}. {record_type}** (ID: {record_id})\n"
                response_text += f"📅 Дата: {created_at}\n"
                response_text += f"📝 Содержание: {display_content}\n\n"
        
        response_text += "💡 **Выберите действие:**"
        
        await message.answer(
            response_text,
            parse_mode="Markdown",
            reply_markup=get_manage_tests_keyboard()
        )
        
    except Exception as e:
        logging.error(f"Ошибка при управлении анализами: {e}")
        await message.answer("😔 Произошла ошибка при загрузке данных. Попробуйте еще раз.")

# Обработчик команды /models для проверки статуса моделей
@dp.message(Command("models"))
async def models_command(message: types.Message):
    from config import MODEL_CONFIG, TOKEN_LIMITS
    from models import check_model_availability, is_provider_blocked
    
    status_text = "🤖 <b>Статус моделей:</b>\n\n"
    
    # Проверяем заблокированные провайдеры
    blocked_providers = [p for p in MODEL_CONFIG.keys() if is_provider_blocked(p)]
    if blocked_providers:
        status_text += f"🚫 <b>Заблокированные провайдеры:</b> {', '.join(blocked_providers)}\n\n"

    for provider, config in MODEL_CONFIG.items():
        # Проверяем статус провайдера
        if is_provider_blocked(provider):
            status_text += f"<b>{provider.upper()}:</b> 🚫 <i>Заблокирован</i>\n"
        else:
            status_text += f"<b>{provider.upper()}:</b>\n"

        for model in config["models"]:
            model_name = model["name"]
            is_available = await check_model_availability(provider, model_name)
            status = "✅ Доступна" if is_available else "❌ Недоступна"
            status_text += f"  • {model_name}: {status}\n"

        # Добавляем информацию об использовании токенов
        token_info = TOKEN_LIMITS.get(provider, {})
        if token_info.get("daily_limit", 0) > 0:
            used = token_info.get("used_today", 0)
            limit = token_info["daily_limit"]
            percentage = (used / limit) * 100 if limit > 0 else 0
            status_text += f"  📊 Токены: {used}/{limit} ({percentage:.1f}%)\n"

    status_text += "\n"

    await message.answer(status_text, parse_mode="HTML")

# Обработчик команды /profile
@dp.message(Command("profile"))
async def profile_command(message: types.Message, state: FSMContext):
    profile = get_patient_profile(generate_user_uuid(message.from_user.id))
    if profile:
        await message.answer(
            f"👤 <b>Ваш профиль:</b>\n\n"
            f"🆔 ID: {profile['id']}\n"
            f"📝 Имя: {profile['name']}\n"
            f"🎂 Возраст: {profile['age']}\n"
            f"⚧️ Пол: {profile['gender']}\n"
            f"📅 Создан: {profile['created_at'][:10]}",
            parse_mode="HTML",
            reply_markup=get_main_keyboard()
        )
    else:
        await message.answer(
            "📝 Для создания профиля пациента, пожалуйста, предоставьте следующую информацию:\n\n"
            "1. Ваше имя\n"
            "2. Возраст\n"
            "3. Пол (М/Ж)\n\n"
            "Пожалуйста, отправьте информацию в формате:\n"
            "<b>Имя: [ваше имя]</b>\n"
            "<b>Возраст: [ваш возраст]</b>\n"
            "<b>Пол: [М/Ж]</b>",
            parse_mode="HTML"
        )
    await state.set_state(DoctorStates.waiting_for_patient_id)

# Обработчик команды /stats
@dp.message(Command("stats"))
async def stats_command(message: types.Message):
    try:
        response = supabase.table("doc_user_feedback").select("*").eq("user_id", generate_user_uuid(message.from_user.id)).execute()
        total = len(response.data)
        helped = sum(1 for item in response.data if item["helped"])

        # Получаем статистику по успешным ответам
        successful_responses = get_user_successful_responses(generate_user_uuid(message.from_user.id))

        await message.answer(
            f"📊 Ваша статистика:\n"
            f"Всего вопросов: {total}\n"
            f"Помогло ответов: {helped}\n"
            f"Успешность: {helped / total * 100:.1f}%" if total > 0 else "📊 У вас пока нет статистики",
            reply_markup=get_main_keyboard()
        )
    except Exception as e:
        logging.error(f"Ошибка при получении статистики: {e}")
        await message.answer("😔 Не удалось загрузить статистику")

# Обработчик команды /history
@dp.message(Command("history"))
async def history_command(message: types.Message):
    try:
        response = supabase.table("doc_user_feedback").select("*").eq("user_id", generate_user_uuid(message.from_user.id)).order(
            "created_at", desc=True).limit(5).execute()
        if response.data:
            history_text = "📝 Последние вопросы:\n\n"
            for item in response.data:
                status = "✅" if item["helped"] else "❌"
                history_text += f"{status} {item['question'][:50]}...\n"
            await message.answer(history_text)
        else:
            await message.answer("📝 У вас пока нет истории обращений")
    except Exception as e:
        logging.error(f"Ошибка при получении истории: {e}")
        await message.answer("😔 Не удалось загрузить историю")

# Обработчик команды /clear
@dp.message(Command("clear"))
async def clear_command(message: types.Message, state: FSMContext):
    try:
        await clear_conversation_state(state, message.chat.id)
        supabase.table("doc_user_feedback").delete().eq("user_id", generate_user_uuid(message.from_user.id)).execute()
        await message.answer("🗑️ Ваша история очищена")
    except Exception as e:
        logging.error(f"Ошибка при очистке истории: {e}")
        await message.answer("😔 Не удалось очистить историю")

# Обработчик команды /reset_providers
@dp.message(Command("reset_providers"))
async def reset_providers_command(message: types.Message):
    """Сбрасывает блокировки провайдеров (только для администраторов)"""
    try:
        # Проверяем, является ли пользователь администратором (можно настроить по user_id)
        admin_user_ids = [1298530968]  # Добавьте сюда ID администраторов
        
        if message.from_user.id not in admin_user_ids:
            await message.answer("❌ У вас нет прав для выполнения этой команды.")
            return
        
        # Сбрасываем блокировки провайдеров
        from models import reset_provider_blocks
        reset_provider_blocks()
        
        await message.answer("✅ Блокировки провайдеров сброшены. Все провайдеры снова доступны.")
        logging.info(f"Администратор {message.from_user.id} сбросил блокировки провайдеров")
        
    except Exception as e:
        logging.error(f"Ошибка при сбросе блокировок провайдеров: {e}")
        await message.answer("😔 Не удалось сбросить блокировки провайдеров.")

# Обработчики callback для управления анализами
@dp.callback_query(F.data.startswith("delete_tests"))
async def delete_tests_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик кнопки удаления анализов"""
    try:
        from database import get_latest_test_results
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} выбрал удаление анализов")
        
        # Получаем последние анализы пользователя
        tests = get_latest_test_results(user_id, limit=10)
        
        if not tests:
            await callback.message.edit_text(
                "📊 У вас пока нет сохраненных анализов для удаления.",
                reply_markup=get_main_keyboard()
            )
            await state.clear()
            return
        
        await state.set_state(DoctorStates.confirming_delete)
        
        # Формируем сообщение со списком анализов для удаления
        response_text = "🗑️ **Выберите анализы для удаления:**\n\n"
        
        await callback.message.edit_text(
            response_text,
            parse_mode="Markdown",
            reply_markup=get_delete_test_keyboard(tests)
        )
        
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при обработке удаления анализов: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith("delete_all_tests"))
async def delete_all_tests_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик кнопки удаления всех анализов"""
    try:
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} выбрал удаление всех анализов")
        
        await state.set_state(DoctorStates.confirming_delete_all)
        
        await callback.message.edit_text(
            "⚠️ **Подтвердите удаление ВСЕХ анализов!**\n\n"
            "⚠️ Это действие нельзя отменить! Все ваши анализы будут безвозвратно удалены.\n\n"
            "Вы уверены, что хотите удалить все анализы?",
            parse_mode="Markdown",
            reply_markup=get_confirm_delete_all_keyboard()
        )
        
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при обработке удаления всех анализов: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith("delete_medical_records"))
async def delete_medical_records_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик кнопки удаления медицинских записей"""
    try:
        from database import get_medical_records
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} выбрал удаление медицинских записей")
        
        # Получаем медицинские записи пользователя
        medical_records = get_medical_records(user_id, "image_analysis")
        
        if not medical_records:
            await callback.message.edit_text(
                "📊 У вас пока нет медицинских записей для удаления.",
                reply_markup=get_main_keyboard()
            )
            await state.clear()
            return
        
        # Формируем сообщение со списком медицинских записей для удаления
        response_text = "🗑️ **Выберите медицинские записи для удаления:**\n\n"
        
        await callback.message.edit_text(
            response_text,
            parse_mode="Markdown",
            reply_markup=get_delete_medical_record_keyboard(medical_records)
        )
        
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при обработке удаления медицинских записей: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith("delete_by_date"))
async def delete_by_date_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик кнопки удаления по дате"""
    try:
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} выбрал удаление по дате")
        
        await state.set_state(DoctorStates.choosing_delete_period)
        
        await callback.message.edit_text(
            "📅 **Выберите период для удаления:**\n\n"
            "Будут удалены все анализы, созданные в выбранный период.",
            parse_mode="Markdown",
            reply_markup=get_date_range_keyboard()
        )
        
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при обработке удаления по дате: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith("delete_test_"))
async def delete_test_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик выбора конкретного анализа для удаления"""
    try:
        from database import get_latest_test_results
        test_id = int(callback.data.split("_")[-1])
        user_id = generate_user_uuid(callback.from_user.id)
        
        logging.info(f"Пользователь {callback.from_user.id} выбрал удаление анализа {test_id}")
        
        # Получаем информацию об анализе
        tests = get_latest_test_results(user_id, limit=50)
        test_to_delete = None
        for test in tests:
            if test.get('id') == test_id:
                test_to_delete = test
                break
        
        if test_to_delete:
            test_name = test_to_delete.get('test_name', 'Неизвестный анализ')
            await state.set_state(DoctorStates.confirming_delete)
            await state.update_data({"test_id_to_delete": test_id, "test_name_to_delete": test_name})
            
            await callback.message.edit_text(
                f"🗑️ **Подтвердите удаление:**\n\n"
                f"Анализ: {test_name}\n\n"
                f"⚠️ Это действие нельзя отменить!",
                parse_mode="Markdown",
                reply_markup=get_confirm_delete_keyboard(test_id, test_name)
            )
        else:
            await callback.message.edit_text("❌ Анализ не найден.")
            await state.clear()
        
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при выборе анализа для удаления: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith("delete_medical_record_"))
async def delete_medical_record_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик выбора медицинской записи для удаления"""
    try:
        from database import get_medical_records, delete_medical_record
        record_id = int(callback.data.split("_")[-1])
        user_id = generate_user_uuid(callback.from_user.id)
        
        logging.info(f"Пользователь {callback.from_user.id} выбрал удаление медицинской записи {record_id}")
        
        # Получаем информацию о записи
        medical_records = get_medical_records(user_id, "image_analysis")
        record_to_delete = None
        for record in medical_records:
            if record.get('id') == record_id:
                record_to_delete = record
                break
        
        if record_to_delete:
            content = record_to_delete.get("content", "")
            created_at = record_to_delete.get("created_at", "")[:10] if record_to_delete.get("created_at") else "Не указана"
            
            # Определяем тип записи
            if "не удалось извлечь" in content.lower() or len(content.strip()) < 100:
                record_type = "❌ Неудачный анализ"
            else:
                record_type = "✅ Успешный анализ"
            
            # Обрезаем контент для отображения
            display_content = content[:100] + "..." if len(content) > 100 else content
            
            await state.set_state(DoctorStates.confirming_delete)
            await state.update_data({
                "medical_record_id_to_delete": record_id, 
                "medical_record_content": display_content,
                "medical_record_type": record_type
            })
            
            await callback.message.edit_text(
                f"🗑️ **Подтвердите удаление медицинской записи:**\n\n"
                f"Тип: {record_type}\n"
                f"ID: {record_id}\n"
                f"Дата: {created_at}\n"
                f"Содержание: {display_content}\n\n"
                f"⚠️ Это действие нельзя отменить!",
                parse_mode="Markdown",
                reply_markup=get_confirm_delete_medical_record_keyboard(record_id, record_type)
            )
        else:
            await callback.message.edit_text("❌ Медицинская запись не найдена.")
            await state.clear()
        
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при выборе медицинской записи для удаления: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith("confirm_delete_medical_record_"))
async def confirm_delete_medical_record_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик подтверждения удаления медицинской записи"""
    try:
        from database import delete_medical_record
        record_id = int(callback.data.split("_")[-1])
        user_id = generate_user_uuid(callback.from_user.id)
        
        logging.info(f"Пользователь {callback.from_user.id} подтвердил удаление медицинской записи {record_id}")
        
        # Удаляем медицинскую запись
        success = await delete_medical_record(user_id, record_id)
        
        if success:
            await callback.message.edit_text(
                f"✅ Медицинская запись ID:{record_id} успешно удалена!"
            )
        else:
            await callback.message.edit_text("❌ Не удалось удалить медицинскую запись. Попробуйте еще раз.")
            
        await callback.answer()
        await state.clear()
        
    except Exception as e:
        logging.error(f"Ошибка при подтверждении удаления медицинской записи: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await callback.answer()

@dp.callback_query(F.data == "delete_all_medical_records")
async def delete_all_medical_records_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик кнопки удаления всех медицинских записей"""
    try:
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} выбрал удаление всех медицинских записей")
        
        await state.set_state(DoctorStates.confirming_delete_all)
        
        await callback.message.edit_text(
            "⚠️ **Подтвердите удаление ВСЕХ медицинских записей!**\n\n"
            "⚠️ Это действие нельзя отменить! Все ваши медицинские записи (включая изображения и PDF) будут безвозвратно удалены.\n\n"
            "Вы уверены, что хотите удалить все медицинские записи?",
            parse_mode="Markdown",
            reply_markup=get_confirm_delete_all_medical_records_keyboard()
        )
        
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при обработке удаления всех медицинских записей: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data == "confirm_delete_all_medical_records")
async def confirm_delete_all_medical_records_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик подтверждения удаления всех медицинских записей"""
    try:
        from database import delete_all_medical_records
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} подтвердил удаление всех медицинских записей")
        
        # Удаляем все медицинские записи
        deleted_count = await delete_all_medical_records(user_id)
        
        await callback.message.edit_text(
            f"✅ Удалено {deleted_count} медицинских записей!"
        )
        
        logging.info(f"Пользователь {user_id} удалил {deleted_count} медицинских записей")
        await state.clear()
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при удалении всех медицинских записей: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith("confirm_delete_"))
async def confirm_delete_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик подтверждения удаления анализа"""
    try:
        callback_data_parts = callback.data.split("_")
        
        # Проверяем, это удаление конкретного анализа или всех
        if callback_data_parts[2] == "all":
            # Это обработка удаления всех анализов - перенаправляем в нужный обработчик
            await confirm_delete_all_callback(callback, state)
            return
        
        # Удаление конкретного анализа
        test_id = int(callback_data_parts[2])
        user_id = generate_user_uuid(callback.from_user.id)
        
        # Получаем информацию об анализе для подтверждения
        from database import get_latest_test_results
        tests = get_latest_test_results(user_id, limit=50)
        test_to_delete = None
        for test in tests:
            if test.get('id') == test_id:
                test_to_delete = test
                break
        
        if test_to_delete:
            # Удаляем анализ
            success = await delete_test_result(user_id, test_id)
            if success:
                await callback.message.edit_text(
                    f"✅ Анализ **{test_to_delete.get('test_name', 'Неизвестный')}** успешно удален!"
                )
            else:
                await callback.message.edit_text("❌ Не удалось удалить анализ. Попробуйте еще раз.")
        else:
            await callback.message.edit_text("❌ Анализ не найден.")
            
        await callback.answer()
        await state.clear()
        
    except Exception as e:
        logging.error(f"Ошибка при подтверждении удаления анализа: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await callback.answer()

@dp.callback_query(F.data == "confirm_delete_all")
async def confirm_delete_all_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик подтверждения удаления всех анализов"""
    try:
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} подтвердил удаление всех анализов")
        
        # Удаляем все анализы
        deleted_count = await delete_all_test_results(user_id)
        
        await callback.message.edit_text(
            f"✅ Удалено {deleted_count} анализов!"
        )
        
        logging.info(f"Пользователь {user_id} удалил {deleted_count} анализов")
        await state.clear()
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при удалении всех анализов: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith("confirm_period_"))
async def confirm_period_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик подтверждения удаления по периоду"""
    try:
        period = callback.data.split("_")[1]
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} подтвердил удаление за период {period}")
        
        # Удаляем анализы за период
        deleted_count = await delete_test_results_by_period(user_id, period)
        
        await callback.message.edit_text(
            f"✅ Удалено {deleted_count} анализов за период {period}!"
        )
        
        logging.info(f"Пользователь {user_id} удалил {deleted_count} анализов за период {period}")
        await state.clear()
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при удалении анализов за период: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith(("today", "week", "month", "year")))
async def period_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик выбора периода удаления"""
    try:
        period_map = {
            "delete_today": "сегодня",
            "delete_week": "неделю", 
            "delete_month": "месяц",
            "delete_year": "год"
        }
        
        period = period_map.get(callback.data, "неизвестный период")
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} выбрал период {period}")
        
        await callback.message.edit_text(
            f"📅 **Подтвердите удаление за {period}:**\n\n"
            f"⚠️ Будут удалены все анализы за выбранный период!",
            parse_mode="Markdown",
            reply_markup=get_confirm_delete_period_keyboard(period)
        )
        
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при выборе периода удаления: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.startswith("delete_before_date"))
async def delete_before_date_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик удаления до определенной даты"""
    try:
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} выбрал удаление до определенной даты")
        
        await callback.message.edit_text(
            "📅 **Введите дату в формате ГГГГ-ММ-ДД:**\n\n"
            "Например: 2024-01-01\n\n"
            "Будут удалены все анализы, созданные до этой даты.",
            parse_mode="Markdown"
        )
        
        await state.set_state(DoctorStates.waiting_for_date)
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при обработке удаления до даты: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")
        await state.clear()

@dp.callback_query(F.data.in_(["cancel_manage", "cancel_delete"]))
async def cancel_manage_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик отмены управления анализами"""
    try:
        logging.info(f"Пользователь {callback.from_user.id} отменил управление анализами")
        
        await callback.message.edit_text(
            "❌ Операция отменена.",
            reply_markup=get_main_keyboard()
        )
        
        await state.clear()
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при отмене управления анализами: {e}")
        await state.clear()

@dp.callback_query(F.data == "view_all_tests")
async def view_all_tests_callback(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик просмотра всех анализов"""
    try:
        from database import get_latest_test_results
        user_id = generate_user_uuid(callback.from_user.id)
        logging.info(f"Пользователь {callback.from_user.id} запросил просмотр всех анализов")
        
        # Получаем все анализы пользователя
        tests = get_latest_test_results(user_id, limit=20)
        
        if not tests:
            await callback.message.edit_text(
                "📊 У вас пока нет сохраненных анализов.",
                reply_markup=get_main_keyboard()
            )
            return
        
        await state.clear()
        
        # Формируем сообщение со списком всех анализов
        response_text = f"📊 **Все ваши анализы ({len(tests)}):**\n\n"
        
        for i, test in enumerate(tests):
            test_name = test.get("test_name", "Неизвестный анализ")
            test_date = test.get("created_at", "")[:10] if test.get("created_at") else "Не указана"
            result = test.get("result", "Не указан")
            ref_values = test.get("reference_values", "")
            units = test.get("units", "")
            
            response_text += f"**{i+1}. {test_name}**\n"
            response_text += f"📅 Дата: {test_date}\n"
            response_text += f"🔬 Результат: {result}"
            if units:
                response_text += f" {units}"
            if ref_values:
                response_text += f" (норма: {ref_values})"
            response_text += "\n\n"
        
        response_text += "\n💡 Используйте команду /manage_tests для управления анализами."
        
        await callback.message.edit_text(
            response_text,
            parse_mode="Markdown"
        )
        
        await callback.answer()
        
    except Exception as e:
        logging.error(f"Ошибка при просмотре всех анализов: {e}")
        await callback.message.edit_text("😔 Произошла ошибка. Попробуйте еще раз.")

# Обработчик текстовых сообщений
@dp.message(F.text)
async def handle_text_message(message: types.Message, state: FSMContext):
    """Обработчик текстовых сообщений с использованием улучшенной RAG системы"""
    try:
        user_id = generate_user_uuid(message.from_user.id)
        query = message.text
        
        logging.info(f"Обработка текстового сообщения от пользователя {message.from_user.id}: {query[:100]}...")
        
        # Проверяем, не является ли это командой
        if query.startswith('/'):
            return
        
        # Обрабатываем запрос с помощью улучшенной RAG системы
        response, metadata = await enhanced_rag_system.process_query(user_id, query)
        
        # Экранируем специальные символы для Markdown
        clean_response = escape_markdown(response)
        
        # Отправляем ответ
        await message.answer(clean_response, parse_mode="Markdown")
        
        # Логируем успешную обработку
        logging.info(f"Запрос успешно обработан. Длина контекста: {metadata.get('context_length', 0)}")
        
    except Exception as e:
        logging.error(f"Ошибка при обработке текстового сообщения: {e}")
        await message.answer("Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз.")

# Обработчик фото - УПРОЩЕННАЯ ВЕРСИЯ
@dp.message(F.photo)
async def handle_photo_message(message: types.Message, state: FSMContext):
    """Упрощенный обработчик фото"""
    try:
        user_id = generate_user_uuid(message.from_user.id)
        logging.info(f"Получено фото от пользователя {message.from_user.id}")
        
        # Отправляем сообщение о начале обработки
        processing_msg = await message.answer("🔍 Анализирую изображение...")
        
        try:
            # Получаем URL файла
            photo = message.photo[-1]
            file_info = await bot.get_file(photo.file_id)
            file_url = f"https://api.telegram.org/file/bot{bot_token}/{file_info.file_path}"
            
            # Используем упрощенный процессор
            from photo_processor import SimplePhotoProcessor
            processor = SimplePhotoProcessor()
            
            result = await processor.process_photo(file_url)
            
            if result["success"]:
                # Сохраняем в базу данных
                await save_medical_record(
                    user_id, 
                    "image_analysis", 
                    result["response"], 
                    "simple_processor"
                )
                
                # Сохраняем структурированные данные
                if result["structured_data"]:
                    await save_structured_tests(user_id, result["structured_data"])
                
                # Отправляем ответ с использованием безопасной функции
                from utils import safe_send_message
                await safe_send_message(
                    message,
                    result["response"],
                    reply_markup=get_feedback_keyboard()
                )
                
                # Удаляем сообщение о обработке
                await processing_msg.delete()
                
            else:
                await processing_msg.edit_text(
                    f"❌ {result['error']}\n\n"
                    "💡 Попробуйте сделать фото более четким или отправьте PDF файл с анализами."
                )
            
        except Exception as e:
            logging.error(f"Ошибка при обработке фото: {e}")
            await processing_msg.edit_text(
                "😔 Не удалось обработать изображение. Попробуйте еще раз или отправьте PDF файл."
            )
            
    except Exception as e:
        logging.error(f"Ошибка при обработке фото: {e}")
        await message.answer("Извините, произошла ошибка. Попробуйте еще раз.")

async def save_structured_tests(user_id: str, tests: List[Dict]):
    """Сохранение структурированных тестов"""
    try:
        for test in tests:
            test_data = {
                "user_id": user_id,
                "test_name": test.get("test_name", ""),
                "result": test.get("result", ""),
                "reference_values": test.get("reference_values", ""),
                "units": test.get("units", ""),
                "category": test.get("category", ""),
                "created_at": datetime.now().isoformat()
            }
            
            supabase.table("doc_structured_test_results").insert(test_data).execute()
            
    except Exception as e:
        logging.error(f"Ошибка сохранения структурированных тестов: {e}")

# Обработчик документов (PDF)
@dp.message(F.document)
async def handle_document_message(message: types.Message, state: FSMContext):
    """Обработчик документов (PDF) с медицинскими анализами"""
    try:
        user_id = generate_user_uuid(message.from_user.id)
        document = message.document
        
        logging.info(f"Получен документ от пользователя {message.from_user.id}: {document.file_name}")
        
        # Проверяем, что это PDF
        if not document.file_name.lower().endswith('.pdf'):
            await message.answer("❌ Поддерживаются только PDF файлы. Пожалуйста, загрузите PDF документ.")
            return
        
        # Отправляем сообщение о начале обработки
        processing_msg = await message.answer("📄 Обрабатываю PDF документ... Пожалуйста, подождите.")
        
        try:
            # Получаем URL файла
            file_info = await bot.get_file(document.file_id)
            file_url = file_info.file_path
            full_url = f"https://api.telegram.org/file/bot{bot_token}/{file_url}"
            
            logging.info(f"URL PDF файла: {full_url}")
            
            # Извлекаем текст из PDF
            pdf_text = await extract_text_from_pdf(full_url)
            
            if not pdf_text:
                await processing_msg.edit_text("❌ Не удалось извлечь текст из PDF. Возможно, файл поврежден или защищен.")
                return
            
            # Используем медицинский агент для извлечения структурированных данных из PDF
            from medical_terms_agent import medical_terms_agent
            
            try:
                # Извлекаем параметры анализов с помощью LLM
                test_parameters = await medical_terms_agent.extract_test_parameters(pdf_text)
                
                if test_parameters:
                    # Генерируем анализ на основе структурированных данных
                    analysis_result = await generate_pdf_analysis_description(test_parameters, pdf_text)
                    
                    # Сохраняем структурированные данные
                    await save_structured_tests_from_pdf(user_id, test_parameters)
                    
                    logging.info(f"Извлечено {len(test_parameters)} параметров анализов из PDF")
                else:
                    # Fallback к обычному анализу, если структурированные данные не извлечены
                    analysis_response = await call_model_with_failover(
                        messages=[{"role": "user", "content": f"Проанализируй этот медицинский документ и выдели ключевую информацию:\n\n{pdf_text}"}],
                        model_type="text",
                        system_prompt="Ты — медицинский эксперт. Проанализируй документ и выдели ключевую информацию о пациенте, анализах, диагнозах и рекомендациях."
                    )
                    
                    # Извлекаем текст из кортежа
                    if isinstance(analysis_response, tuple) and len(analysis_response) > 0:
                        analysis_result = analysis_response[0]
                    else:
                        analysis_result = str(analysis_response)
                    
            except Exception as e:
                logging.error(f"Ошибка при извлечении параметров из PDF: {e}")
                # Fallback к обычному анализу
                analysis_result = await call_model_with_failover(
                    messages=[{"role": "user", "content": f"Проанализируй этот медицинский документ и выдели ключевую информацию:\n\n{pdf_text}"}],
                    model_type="text",
                    system_prompt="Ты — медицинский эксперт. Проанализируй документ и выдели ключевую информацию о пациенте, анализах, диагнозах и рекомендациях."
                )
            
            # Проверяем дубликаты
            is_duplicate = await check_duplicate_medical_record_ai_enhanced(
                user_id, analysis_result, "pdf_analysis"
            )
            
            if is_duplicate:
                await processing_msg.edit_text("⚠️ Похожий документ уже был проанализирован ранее.")
                return
            
            # Сохраняем результат анализа в базу данных
            await save_medical_record(user_id, "pdf_analysis", analysis_result, "telegram_pdf")
            
            # Отправляем результат анализа
            escaped_analysis = escape_html(analysis_result)
            await processing_msg.edit_text(
                f"📋 <b>Анализ PDF документа:</b>\n\n{escaped_analysis}",
                parse_mode="HTML",
                reply_markup=get_feedback_keyboard()
            )
            
            logging.info(f"PDF документ успешно проанализирован для пользователя {user_id}")
            
        except Exception as e:
            logging.error(f"Ошибка при анализе PDF: {e}")
            await processing_msg.edit_text(
                "😔 Не удалось обработать PDF документ. Возможно, файл поврежден или произошла ошибка."
            )
            
    except Exception as e:
        logging.error(f"Ошибка при обработке документа: {e}")
        await message.answer("Извините, произошла ошибка при обработке документа. Попробуйте еще раз.")

# Обработчик создания профиля
@dp.message(DoctorStates.waiting_for_patient_id)
async def handle_profile_creation(message: types.Message, state: FSMContext):
    try:
        text = message.text
        name = ""
        age = 0
        gender = ""

        for line in text.split('\n'):
            line = line.strip().lower()
            if line.startswith('имя:'):
                name = line.split(':', 1)[1].strip()
            elif line.startswith('возраст:'):
                try:
                    age = int(line.split(':', 1)[1].strip())
                except:
                    pass
            elif line.startswith('пол:'):
                gender = line.split(':', 1)[1].strip()

        if name and age > 0 and gender in ['м', 'ж']:
            if create_patient_profile(generate_user_uuid(message.from_user.id), name, age, gender, message.from_user.id):
                await message.answer(
                    f"✅ Профиль успешно создан!\n\n"
                    f"👤 <b>Ваш профиль:</b>\n"
                    f"📝 Имя: {name}\n"
                    f"🎂 Возраст: {age}\n"
                    f"⚧️ Пол: {'Мужской' if gender == 'м' else 'Женский'}\n\n"
                    f"Теперь вы можете задавать медицинские вопросы и загружать анализы.",
                    parse_mode="HTML",
                    reply_markup=get_main_keyboard()
                )
                await state.clear()
            else:
                await message.answer("😔 Не удалось создать профиль. Пожалуйста, попробуйте еще раз.")
        else:
            await message.answer(
                "❌ Некорректные данные. Пожалуйста, отправьте информацию в формате:\n"
                "<b>Имя: [ваше имя]</b>\n"
                "<b>Возраст: [ваш возраст]</b>\n"
                "<b>Пол: [М/Ж]</b>",
                parse_mode="HTML"
            )
    except Exception as e:
        logging.error(f"Ошибка при создании профиля: {e}")
        await message.answer("😔 Произошла ошибка. Пожалуйста, попробуйте еще раз.")

# Обработчик ввода даты для удаления
@dp.message(DoctorStates.waiting_for_date)
async def handle_date_input(message: types.Message, state: FSMContext):
    """Обработчик ввода даты для удаления"""
    try:
        date_text = message.text.strip()
        user_id = generate_user_uuid(message.from_user.id)
        
        # Проверяем формат даты
        import re
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'  # ГГГГ-ММ-ДД
        
        if not re.match(date_pattern, date_text):
            await message.answer(
                "❌ Неверный формат даты. Используйте формат ГГГГ-ММ-ДД\n"
                "Например: 2024-01-01"
            )
            return
        
        logging.info(f"Пользователь {message.from_user.id} ввел дату для удаления: {date_text}")
        
        # Удаляем анализы до указанной даты
        deleted_count = await delete_test_results_before_date(user_id, date_text)
        
        await message.answer(
            f"✅ Удалено {deleted_count} анализов до {date_text}!"
        )
        
        logging.info(f"Пользователь {user_id} удалил {deleted_count} анализов до {date_text}")
        await state.clear()
        
    except Exception as e:
        logging.error(f"Ошибка при удалении анализов до даты: {e}")
        await message.answer("😔 Произошла ошибка. Попробуйте еще раз.")

# Функция для сброса счетчиков токенов
def reset_token_usage():
    """Сбрасывает ежедневные счетчики использования токенов"""
    try:
        logging.info("Сброс ежедневных счетчиков токенов")
        # Импортируем функцию из models.py
        from models import reset_token_usage as reset_tokens
        reset_tokens()
        logging.info("Счетчики токенов сброшены")
    except Exception as e:
        logging.error(f"Ошибка при сбросе счетчиков токенов: {e}")

# Планировщик для отложенных напоминаний и сброса токенов
@dp.startup()
async def on_startup():
    logging.info("Запуск планировщика задач")
    scheduler.start()
    logging.info("Планировщик задач запущен")

    # Добавляем задачу для ежедневного сброса счетчиков токенов в полночь
    scheduler.add_job(
        reset_token_usage,
        "cron",
        hour=0,
        minute=0,
        id="reset_token_usage"
    )
    logging.info("Добавлена задача ежедневного сброса токенов")
    
    # Добавляем задачу для ежедневного сброса блокировок провайдеров в полночь
    scheduler.add_job(
        reset_provider_blocks,
        "cron",
        hour=0,
        minute=0,
        id="reset_provider_blocks"
    )
    logging.info("Добавлена задача ежедневного сброса блокировок провайдеров")

@dp.shutdown()
async def on_shutdown():
    logging.info("Остановка планировщика задач")
    scheduler.shutdown()
    logging.info("Планировщик задач остановлен")

async def generate_analysis_description(extraction_result: Dict[str, Any]) -> str:
    """
    Генерирует описательный анализ на основе структурированных данных
    """
    try:
        tests = extraction_result.get("structured_tests", [])
        metadata = extraction_result.get("metadata", {})
        
        if not tests:
            return "Не удалось извлечь данные анализов из изображения."
        
        # Формируем описание
        description = "📊 **Анализ медицинских результатов:**\n\n"
        
        # Добавляем информацию о пациенте если есть
        if metadata.get("patient_name"):
            description += f"👤 **Пациент:** {metadata['patient_name']}\n"
        
        # Добавляем дату анализа если есть
        test_dates = [test.get("test_date") for test in tests if test.get("test_date")]
        if test_dates:
            description += f"📅 **Дата анализа:** {test_dates[0]}\n"
        
        description += "\n**Результаты анализов:**\n\n"
        
        # Группируем анализы по категориям с использованием LLM
        categories = {}
        from medical_terms_agent import medical_terms_agent
        
        for test in tests:
            test_name = test.get("test_name", "")
            
            # Используем LLM для категоризации
            try:
                category_data = await medical_terms_agent.categorize_medical_test(test_name)
                category = category_data.get("category", "Другие анализы")
            except Exception as e:
                logging.error(f"Ошибка категоризации теста {test_name}: {e}")
                # Fallback к простому методу
                test_name_lower = test_name.lower()
                if any(keyword in test_name_lower for keyword in ['anti-', 'гепатит', 'hcv', 'hbv', 'hev']):
                    category = "Анализы на гепатиты"
                elif any(keyword in test_name_lower for keyword in ['opisthorchis', 'toxocara', 'lamblia', 'ascaris']):
                    category = "Паразитологические анализы"
                elif 'ige' in test_name_lower or 'аллерг' in test_name_lower:
                    category = "Аллергологические анализы"
                elif any(keyword in test_name_lower for keyword in ['билирубин', 'алат', 'асат', 'ггт']):
                    category = "Биохимические анализы"
                else:
                    category = "Другие анализы"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(test)
        
        # Формируем результат по категориям
        for category, category_tests in categories.items():
            description += f"🔬 **{category}:**\n"
            for test in category_tests:
                test_name = test.get("test_name", "")
                result = test.get("result", "")
                ref_values = test.get("reference_values", "")
                units = test.get("units", "")
                
                description += f"• **{test_name}:** {result}"
                if units:
                    description += f" {units}"
                if ref_values:
                    description += f" (норма: {ref_values})"
                description += "\n"
            description += "\n"
        
        # Добавляем информацию о лаборатории если есть
        if metadata.get("laboratory"):
            description += f"🏥 **Лаборатория:** {metadata['laboratory']}\n"
        
        # Добавляем рекомендации
        description += "\n💡 **Рекомендации:**\n"
        description += "• Проконсультируйтесь с врачом для детальной интерпретации результатов\n"
        description += "• При необходимости повторите анализы через рекомендованный промежуток времени\n"
        description += "• Сохраните результаты для отслеживания динамики показателей"
        
        return description
        
    except Exception as e:
        logging.error(f"Ошибка при генерации описания анализа: {e}")
        return "Произошла ошибка при анализе результатов. Попробуйте еще раз."

async def save_structured_tests_from_image(user_id: str, extraction_result: Dict[str, Any]) -> int:
    """
    Сохраняет структурированные данные анализов в базу данных
    """
    try:
        from database import save_medical_record
        from structured_tests_agent import TestExtractionAgent
        
        tests = extraction_result.get("structured_tests", [])
        if not tests:
            return 0
        
        # Сохраняем сырой анализ как медицинскую запись
        raw_analysis = extraction_result.get("raw_analysis", "")
        await save_medical_record(user_id, "image_analysis", raw_analysis, "enhanced_extraction")
        
        # Используем существующий агент для сохранения структурированных данных
        agent = TestExtractionAgent(supabase)
        saved_count = await agent._save_structured_tests(user_id, tests)
        
        logging.info(f"Сохранено {saved_count} структурированных анализов из изображения")
        return saved_count
        
    except Exception as e:
        logging.error(f"Ошибка при сохранении структурированных анализов: {e}")
        return 0

async def generate_pdf_analysis_description(test_parameters: List[Dict[str, Any]], pdf_text: str) -> str:
    """
    Генерирует описательный анализ PDF на основе структурированных данных
    """
    try:
        if not test_parameters:
            return "Не удалось извлечь структурированные данные из PDF документа."
        
        # Формируем описание
        description = "📋 **Анализ PDF документа с медицинскими анализами:**\n\n"
        
        # Группируем анализы по категориям с использованием LLM
        from medical_terms_agent import medical_terms_agent
        categories = {}
        
        for test in test_parameters:
            test_name = test.get("test_name", "")
            
            # Используем LLM для категоризации
            try:
                category_data = await medical_terms_agent.categorize_medical_test(test_name)
                category = category_data.get("category", "Другие анализы")
            except Exception as e:
                logging.error(f"Ошибка категоризации теста {test_name}: {e}")
                # Fallback к простому методу
                test_name_lower = test_name.lower()
                if any(keyword in test_name_lower for keyword in ['anti-', 'гепатит', 'hcv', 'hbv', 'hev']):
                    category = "Анализы на гепатиты"
                elif any(keyword in test_name_lower for keyword in ['opisthorchis', 'toxocara', 'lamblia', 'ascaris']):
                    category = "Паразитологические анализы"
                elif 'ige' in test_name_lower or 'аллерг' in test_name_lower:
                    category = "Аллергологические анализы"
                elif any(keyword in test_name_lower for keyword in ['билирубин', 'алат', 'асат', 'ггт']):
                    category = "Биохимические анализы"
                else:
                    category = "Другие анализы"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(test)
        
        # Формируем результат по категориям
        for category, category_tests in categories.items():
            description += f"🔬 **{category}:**\n"
            for test in category_tests:
                test_name = test.get("test_name", "")
                result = test.get("result", "")
                ref_values = test.get("reference_values", "")
                units = test.get("units", "")
                test_date = test.get("test_date", "")
                laboratory = test.get("laboratory", "")
                
                description += f"• **{test_name}:** {result}"
                if units:
                    description += f" {units}"
                if ref_values:
                    description += f" (норма: {ref_values})"
                if test_date:
                    description += f"\n  📅 Дата: {test_date}"
                if laboratory:
                    description += f"\n  🏥 Лаборатория: {laboratory}"
                description += "\n"
            description += "\n"
        
        # Добавляем рекомендации
        description += "\n💡 **Рекомендации:**\n"
        description += "• Проконсультируйтесь с врачом для детальной интерпретации результатов\n"
        description += "• При необходимости повторите анализы через рекомендованный промежуток времени\n"
        description += "• Обратите внимание на показатели, выходящие за пределы референсных значений"
        
        return description
        
    except Exception as e:
        logging.error(f"Ошибка при генерации описания PDF анализа: {e}")
        return "Произошла ошибка при анализе PDF документа. Попробуйте еще раз."

async def save_structured_tests_from_pdf(user_id: str, test_parameters: List[Dict[str, Any]]) -> int:
    """
    Сохраняет структурированные данные анализов из PDF в базу данных
    """
    try:
        from structured_tests_agent import TestExtractionAgent
        
        if not test_parameters:
            return 0
        
        # Используем существующий агент для сохранения структурированных данных
        agent = TestExtractionAgent(supabase)
        saved_count = await agent._save_structured_tests(user_id, test_parameters)
        
        logging.info(f"Сохранено {saved_count} структурированных анализов из PDF")
        return saved_count
        
    except Exception as e:
        logging.error(f"Ошибка при сохранении структурированных анализов из PDF: {e}")
        return 0

# Запуск бота
async def main():
    logging.info("Запуск бота")
    try:
        await dp.start_polling(bot)
        logging.info("Бот успешно запущен")
    except Exception as e:
        logging.error(f"Ошибка при запуске бота: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
