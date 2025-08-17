import os
import asyncio
import requests
import json
import uuid
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.utils.keyboard import InlineKeyboardBuilder
from openai import OpenAI
from tavily import TavilyClient
from supabase import create_client, Client
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import aiohttp
from bs4 import BeautifulSoup
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
import PyPDF2
import io
import base64
import re
from dateutil.parser import parse
from pytz import timezone

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация клиентов
bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()
scheduler = AsyncIOScheduler()

# OpenRouter
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Tavily API
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Supabase
supabase: Client = create_client(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)

# Медицинские источники
MEDICAL_SOURCES = [
    "https://www.who.int/ru",
    "https://medportal.ru",
    "https://www.webmd.com",
    "https://www.mayoclinic.org"
]

# Конфигурация моделей для failover
MODEL_CONFIG = {
    "openrouter": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            {"name": "qwen/qwen2.5-vl-72b-instruct:free", "priority": 1, "type": "vision"},
            {"name": "google/gemini-2.0-flash-exp:free", "priority": 2, "type": "vision"},
            {"name": "deepseek/deepseek-chat-v3-0324:free", "priority": 3, "type": "text"},
            {"name": "deepseek/deepseek-r1-0528:free", "priority": 4, "type": "text"},
            {"name": "openai/gpt-oss-20b:free", "priority": 5, "type": "text"},
            {"name": "z-ai/glm-4.5-air:free", "priority": 6, "type": "text"},
            {"name": "moonshotai/kimi-k2:free", "priority": 7, "type": "text"}
        ],
        "client": openrouter_client
    },
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            {"name": "meta-llama/llama-4-scout-17b-16e-instruct:free", "priority": 1, "type": "vision"},
            {"name": "meta-llama/llama-4-maverick-17b-128e-instruct:free", "priority": 2, "type": "vision"},
            {"name": "llama-3.2-90b-vision-preview:free", "priority": 3, "type": "vision"},
            {"name": "deepseek-r1-distill-llama-70b", "priority": 4, "type": "text"},
            {"name": "openai/gpt-oss-120b", "priority": 5, "type": "text"}
        ],
        "client": None  # Будет инициализирован позже
    },
    "cerebras": {
        "api_key": os.getenv("CEREBRAS_API_KEY"),
        "base_url": "https://api.cerebras.ai/v1",
        "models": [
            {"name": "qwen-3-235b-thinking", "priority": 1, "type": "text"}
        ],
        "client": None  # Будет инициализирован позже
    }
}

# Инициализация клиентов для Cerebras и Groq
if MODEL_CONFIG["cerebras"]["api_key"]:
    MODEL_CONFIG["cerebras"]["client"] = OpenAI(
        base_url=MODEL_CONFIG["cerebras"]["base_url"],
        api_key=MODEL_CONFIG["cerebras"]["api_key"]
    )

# Функция для генерации UUID на основе Telegram user ID
def generate_user_uuid(telegram_user_id: int) -> str:
    """
    Генерирует детерминированный UUID на основе Telegram user ID.
    Один и тот же Telegram user ID всегда будет генерировать один и тот же UUID.
    """
    # Создаем namespace UUID для Telegram (используем фиксированный UUID)
    telegram_namespace = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')
    
    # Создаем UUID на основе namespace и user_id
    return str(uuid.uuid5(telegram_namespace, str(telegram_user_id)))

if MODEL_CONFIG["groq"]["api_key"]:
    MODEL_CONFIG["groq"]["client"] = OpenAI(
        base_url=MODEL_CONFIG["groq"]["base_url"],
        api_key=MODEL_CONFIG["groq"]["api_key"]
    )

# Параметры токенов (если есть)
TOKEN_LIMITS = {
    "openrouter": {
        "daily_limit": int(os.getenv("OPENROUTER_DAILY_LIMIT", "100000")),
        "used_today": 0
    },
    "cerebras": {
        "daily_limit": int(os.getenv("CEREBRAS_DAILY_LIMIT", "50000")),
        "used_today": 0
    },
    "groq": {
        "daily_limit": int(os.getenv("GROQ_DAILY_LIMIT", "50000")),
        "used_today": 0
    }
}

# Константы
MAX_HISTORY_LENGTH = 10
MAX_CONTEXT_MESSAGES = 6
AGENT_CACHE_EXPIRE_HOURS = 24


# Состояния для FSM
class DoctorStates(StatesGroup):
    waiting_for_feedback = State()
    waiting_for_clarification = State()
    waiting_for_file = State()
    waiting_for_patient_id = State()
    viewing_history = State()
    confirming_profile = State()


# Функция для экранирования HTML
def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# Функция для проверки доступности модели
async def check_model_availability(provider: str, model_name: str) -> bool:
    """Проверяет доступность модели и наличие токенов"""
    try:
        config = MODEL_CONFIG.get(provider)
        if not config or not config.get("client"):
            logging.warning(f"Провайдер {provider} не настроен")
            return False

        # Проверка лимитов токенов
        token_limit = TOKEN_LIMITS.get(provider, {})
        if token_limit.get("daily_limit", 0) > 0 and token_limit.get("used_today", 0) >= token_limit["daily_limit"]:
            logging.warning(f"Достигнут лимит токенов для провайдера {provider}")
            return False

        # Для OpenRouter можно проверить доступность модели через API
        if provider == "openrouter":
            try:
                headers = {
                    "Authorization": f"Bearer {config['api_key']}"
                }
                response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    available_models = [m["id"] for m in models]
                    return model_name in available_models
            except Exception as e:
                logging.error(f"Ошибка при проверке доступности модели OpenRouter: {e}")

        # Для других провайдеров просто проверяем, что API ключ существует
        return True
    except Exception as e:
        logging.error(f"Ошибка при проверке доступности модели {model_name} у провайдера {provider}: {e}")
        return False


# Функция для обновления счетчика использованных токенов
def update_token_usage(provider: str, tokens_used: int):
    """Обновляет счетчик использованных токенов для провайдера"""
    if provider in TOKEN_LIMITS:
        TOKEN_LIMITS[provider]["used_today"] += tokens_used
        logging.info(
            f"Использовано токенов {provider}: {tokens_used}, всего сегодня: {TOKEN_LIMITS[provider]['used_today']}")


# Функция для сброса счетчиков токенов (можно вызывать раз в день)
def reset_token_usage():
    """Сбрасывает ежедневные счетчики токенов"""
    for provider in TOKEN_LIMITS:
        TOKEN_LIMITS[provider]["used_today"] = 0
    logging.info("Счетчики токенов сброшены")


# Универсальная функция для вызова моделей с failover
async def call_model_with_failover(
    messages: List[Dict[str, str]],
    model_preference: str = None,
    model_type: str = None,  # Новый параметр для указания типа модели
    system_prompt: str = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Универсальная функция для вызова моделей с failover.
    
    Args:
        messages: Список сообщений для модели
        model_preference: Предпочтительная модель (опционально)
        model_type: Тип модели (например, "vision" для анализа изображений)
        system_prompt: Системный промпт (опционально)
    
    Returns:
        (response, provider, metadata)
    """
    # Формируем список всех моделей с учетом приоритета
    all_models = []
    for provider, config in MODEL_CONFIG.items():
        for model in config["models"]:
            # Добавляем информацию о типе модели
            model_info = {
                "provider": provider,
                "name": model["name"],
                "priority": model["priority"],
                "type": model.get("type", "text"),  # По умолчанию text
                "client": config["client"]
            }
            all_models.append(model_info)
    
    # Фильтруем модели по типу, если указан
    if model_type:
        all_models = [m for m in all_models if m.get("type") == model_type]
        if not all_models:
            logging.warning(f"Нет доступных моделей типа '{model_type}', используем все модели")
            # Если нет моделей нужного типа, используем все
            all_models = []
            for provider, config in MODEL_CONFIG.items():
                for model in config["models"]:
                    model_info = {
                        "provider": provider,
                        "name": model["name"],
                        "priority": model["priority"],
                        "type": model.get("type", "text"),
                        "client": config["client"]
                    }
                    all_models.append(model_info)
    
    # Сортируем по приоритету
    all_models.sort(key=lambda x: x["priority"])
    
    # Если указана предпочтительная модель, перемещаем её в начало
    if model_preference:
        preferred_models = [m for m in all_models if m["name"] == model_preference]
        other_models = [m for m in all_models if m["name"] != model_preference]
        all_models = preferred_models + other_models
    
    last_error = None
    
    # Пробуем модели в порядке приоритета
    for model_info in all_models:
        provider = model_info["provider"]
        model_name = model_info["name"]
        client = model_info["client"]
        
        # Проверяем доступность модели
        if not await check_model_availability(provider, model_name):
            logging.info(f"Модель {model_name} провайдера {provider} недоступна, пробуем следующую")
            continue
        
        try:
            logging.info(f"Пробую модель {model_name} от провайдера {provider}")
            
            # Добавляем системный промпт, если он указан
            if system_prompt:
                # Проверяем, есть ли уже системный промпт в сообщениях
                has_system = any(msg.get("role") == "system" for msg in messages)
                if not has_system:
                    messages = [{"role": "system", "content": system_prompt}] + messages
            
            # Добавляем заголовки для OpenRouter
            extra_headers = {}
            if provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://github.com/vokforever/ai-doctor",
                    "X-Title": "AI Doctor Bot"
                }
            
            # Выполняем запрос
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **({"extra_headers": extra_headers} if extra_headers else {})
            )
            
            # Получаем ответ
            current_answer = completion.choices[0].message.content
            
            # Для некоторых моделей (например, Cerebras) может быть цепочка размышлений
            thinking_process = ""
            if provider == "cerebras" and hasattr(completion.choices[0], 'thinking'):
                thinking_process = completion.choices[0].thinking
            
            # Обновляем счетчик токенов (если есть информация)
            if hasattr(completion, 'usage') and completion.usage:
                tokens_used = completion.usage.total_tokens
                update_token_usage(provider, tokens_used)
            
            # Сохраняем информацию о модели
            metadata = {
                "provider": provider,
                "model": model_name,
                "type": model_info.get("type", "text"),
                "thinking": thinking_process,
                "usage": getattr(completion, 'usage', None)
            }
            
            # Возвращаем первый успешный ответ
            return current_answer, provider, metadata
        except Exception as e:
            last_error = e
            logging.warning(f"Ошибка при использовании модели {model_name} от провайдера {provider}: {e}")
            continue
    
    # Если все модели не сработали
    logging.error(f"Все модели недоступны. Последняя ошибка: {last_error}")
    error_message = "😔 К сожалению, произошла ошибка при генерации ответа. Все модели временно недоступны. Попробуйте повторить запрос позже."
    return error_message, "", {}


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
                system_prompt="Ты - медицинский ассистент, который помогает собрать информацию для ответа на медицинский вопрос."
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


# Инициализация агента уточнения
clarification_agent = ClarificationAgent()


# Функция для генерации ответа с failover между провайдерами
async def generate_answer_with_failover(
        question: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        patient_data: Dict[str, Any] = None,
        user_id: int = None,
        system_prompt: str = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Генерирует ответ с использованием failover между провайдерами и моделями.
    Возвращает кортеж: (ответ, провайдер, дополнительная информация)
    """
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
    
    # Формируем сообщения для модели
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    if context:
        messages.append({"role": "system", "content": f"Медицинская информация:\n{context}"})

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

    # Добавляем историю диалога
    if history:
        recent_history = history[-MAX_CONTEXT_MESSAGES:] if len(history) > MAX_CONTEXT_MESSAGES else history
        for msg in recent_history:
            messages.append(msg)

    messages.append({"role": "user", "content": question})

    # Используем универсальную функцию с failover
    return await call_model_with_failover(
        messages=messages,
        system_prompt=system_prompt
    )


# Функция для сохранения успешных ответов с цепочкой размышлений
async def save_successful_response(
        user_id: str,
        question: str,
        answer: str,
        provider: str,
        metadata: Dict[str, Any],
        conversation_history: List[Dict[str, str]] = None
):
    """Сохраняет успешный ответ и цепочку размышлений в базу данных"""
    try:
        # Формируем данные для сохранения
        save_data = {
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "provider": provider,
            "model": metadata.get("model", ""),
            "thinking": metadata.get("thinking", ""),
            "usage": json.dumps(metadata.get("usage", {})),
            "created_at": datetime.now().isoformat()
        }

        # Если есть история диалога, сохраняем ее
        if conversation_history:
            save_data["conversation_history"] = json.dumps(conversation_history)

        # Сохраняем в базу данных
        response = supabase.table("doc_successful_responses").insert(save_data).execute()

        if response.data:
            logging.info(f"Успешный ответ сохранен для пользователя {user_id}")
            return True
        else:
            logging.error(f"Ошибка при сохранении успешного ответа для пользователя {user_id}")
            return False

    except Exception as e:
        logging.error(f"Ошибка при сохранении успешного ответа: {e}")
        return False


# Функция для получения успешных ответов пользователя
def get_user_successful_responses(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Получает успешные ответы пользователя"""
    try:
        response = supabase.table("doc_successful_responses").select("*").eq("user_id", user_id).order("created_at",
                                                                                                       desc=True).limit(
            limit).execute()
        return response.data if response.data else []
    except Exception as e:
        logging.error(f"Ошибка при получении успешных ответов пользователя {user_id}: {e}")
        return []


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
                system_prompt="Ты — медицинский эксперт по анализам. Извлеки из текста все результаты анализов в структурированном формате."
            )
            
            # Извлекаем JSON из ответа
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
            if cached.data and datetime.now() < parse(cached.data[0]["expires_at"]):
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
                system_prompt="Ты — медицинский ассистент. Проанализируй предоставленные анализы и дай краткую сводку."
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


# Инициализация агента
test_agent = TestAnalysisAgent()


# Функция для извлечения даты из текста
def extract_date(text: str) -> Optional[str]:
    """Извлечение даты из текста"""
    date_patterns = [
        r'(\d{1,2})\.(\d{1,2})\.(\d{4})',  # DD.MM.YYYY
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # DD/MM/YYYY
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                if pattern == date_patterns[0]:  # DD.MM.YYYY
                    day, month, year = match.groups()
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                elif pattern == date_patterns[1]:  # YYYY-MM-DD
                    return match.group(0)
                elif pattern == date_patterns[2]:  # DD/MM/YYYY
                    day, month, year = match.groups()
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except:
                continue
    return None


# Функция для сохранения результатов анализов
async def save_test_results(user_id: str, test_results: List[Dict[str, Any]], source: str = ""):
    """Сохранение результатов анализов в базу данных"""
    try:
        for result in test_results:
            # Обработка даты
            test_date = None
            if result.get("test_date"):
                try:
                    test_date = datetime.strptime(result["test_date"], "%Y-%m-%d").date()
                except:
                    test_date = None

            # Определение отклонения от нормы
            is_abnormal = result.get("is_abnormal", False)

            # Сохранение в базу
            supabase.table("doc_test_results").insert({
                "user_id": user_id,
                "test_name": result.get("test_name", ""),
                "value": result.get("value", ""),
                "reference_range": result.get("reference_range"),
                "unit": result.get("unit"),
                "test_date": test_date,
                "is_abnormal": is_abnormal,
                "notes": result.get("notes", ""),
                "source": source
            }).execute()
        return True
    except Exception as e:
        logging.error(f"Ошибка при сохранении результатов анализов: {e}")
        return False


# Функция для получения анализов пациента
def get_patient_tests(user_id: str, test_names: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Получение анализов пациента"""
    try:
        query = supabase.table("doc_test_results").select("*").eq("user_id", user_id)
        if test_names:
            conditions = []
            for name in test_names:
                conditions.append(f"test_name.ilike.%{name}%")
            query = query.or_(*conditions)
        return query.order("test_date", desc=True).limit(limit).execute().data
    except Exception as e:
        logging.error(f"Ошибка при получении анализов пациента: {e}")
        return []


# Функция для создания клавиатуры обратной связи
def get_feedback_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, помогло",
        callback_data="feedback_yes"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Нет, не помогло",
        callback_data="feedback_no"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🔍 Найти больше информации",
        callback_data="search_more"
    ))
    builder.adjust(2, 1)
    return builder.as_markup()


# Функция для создания клавиатуры уточнения
def get_clarification_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="🔍 Уточнить вопрос",
        callback_data="clarify_question"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📊 Загрузить анализы",
        callback_data="upload_tests"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🔄 Попробовать еще раз",
        callback_data="try_again"
    ))
    builder.adjust(1)
    return builder.as_markup()


# Функция для создания главной клавиатуры
def get_main_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="📊 Мои анализы",
        callback_data="my_tests"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📝 Мой анамнез",
        callback_data="my_history"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🆔 Создать профиль пациента",
        callback_data="create_profile"
    ))
    builder.adjust(1)
    return builder.as_markup()


# Функция для получения эмбеддинга
def get_embedding(text: str) -> List[float]:
    """Получение эмбеддинга текста с помощью Mistral AI"""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-embed",
            "input": text
        }
        response = requests.post("https://api.mistral.ai/v1/embeddings", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        logging.error(f"Ошибка при получении эмбеддинга: {e}")
        return []


# Функция для косинусного сходства
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Вычисление косинусного сходства между двумя векторами"""
    try:
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except:
        return 0.0


# Функция для векторного поиска
def vector_search(query: str, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """Поиск похожих вопросов в векторной базе знаний"""
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []

        # Получаем все записи с эмбеддингами
        response = supabase.table("doc_knowledge_base_vector").select("*").execute()
        results = []
        for item in response.data:
            if item.get("embedding"):
                # Конвертируем строку JSON обратно в список
                try:
                    item_embedding = json.loads(item["embedding"])
                    similarity = cosine_similarity(query_embedding, item_embedding)
                    if similarity >= threshold:
                        results.append((item["question"], item["answer"], similarity))
                except (json.JSONDecodeError, TypeError):
                    continue

        # Сортируем по схожести
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:3]  # Возвращаем топ-3 результата
    except Exception as e:
        logging.error(f"Ошибка при векторном поиске: {e}")
        return []


# Функция для сохранения в векторную базу знаний
def save_to_vector_knowledge_base(question: str, answer: str, source: str = ""):
    """Сохранение вопроса и ответа с эмбеддингом"""
    try:
        embedding = get_embedding(question)
        if embedding:
            # Конвертируем эмбеддинг в строку JSON для сохранения
            embedding_json = json.dumps(embedding)
            supabase.table("doc_knowledge_base_vector").insert({
                "question": question,
                "answer": answer,
                "source": source,
                "embedding": embedding_json,
                "created_at": datetime.now().isoformat()
            }).execute()
    except Exception as e:
        logging.error(f"Ошибка при сохранении в векторную базу знаний: {e}")


# Функция для поиска в медицинских источниках
async def search_medical_sources(query: str) -> str:
    try:
        search_query = f"{query} медицина здоровье"
        response = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=3
        )
        results = []
        for result in response["results"]:
            if any(source in result["url"] for source in MEDICAL_SOURCES):
                results.append(f"Источник: {result['url']}\n{result['content']}")
        return "\n\n".join(results) if results else ""
    except Exception as e:
        logging.error(f"Ошибка при поиске в медицинских источниках: {e}")
        return ""


# Функция для извлечения текста из PDF
async def extract_text_from_pdf(file_path: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_path) as response:
                if response.status == 200:
                    pdf_data = await response.read()
                    pdf_file = io.BytesIO(pdf_data)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
        return ""
    except Exception as e:
        logging.error(f"Ошибка при извлечении текста из PDF: {e}")
        return ""


# Функция для извлечения данных пациента из текста
async def extract_patient_data_from_text(text: str) -> Dict[str, Any]:
    try:
        messages = [
            {
                "role": "system",
                "content": f"""Ты — помощник, который извлекает данные пациента из медицинских документов. 
                
                ТЕКУЩАЯ ДАТА: {datetime.now().strftime('%d.%m.%Y')} (год: {datetime.now().year})
                
                Извлеки имя, возраст и пол, если они есть. 
                
                ВАЖНО: При извлечении возраста учитывай текущую дату. Если в документе указан возраст 
                "33 года", а сейчас {datetime.now().year} год, то возраст пациента сейчас больше 33 лет.
                Корректируй возраст в зависимости от того, когда был создан документ.
                
                Верни ответ в формате JSON: 
                {{"name": "имя", "age": число, "gender": "М" или "Ж"}}. 
                Если каких-то данных нет, поставь null."""
            },
            {
                "role": "user",
                "content": text[:2000]
            }
        ]
        
        response_text, _, _ = await call_model_with_failover(
            messages=messages,
            system_prompt="Ты — помощник, который извлекает данные пациента из медицинских документов."
        )
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Вычисляем текущий возраст на основе извлеченного возраста
                extracted_age = data.get("age")
                current_age = None
                if extracted_age and isinstance(extracted_age, int):
                    current_age = calculate_current_age(extracted_age)
                
                return {
                    "name": data.get("name"),
                    "age": current_age,
                    "gender": data.get("gender")
                }
        except json.JSONDecodeError:
            pass

        # Если не удалось извлечь JSON, пробуем простой парсинг
        name_match = re.search(r'(?:Пациент|ФИО|Имя):\s*([А-Яа-я\s]+)', text)
        age_match = re.search(r'(?:Возраст|Лет):\s*(\d+)', text)
        gender_match = re.search(r'(?:Пол):\s*([МЖ])', text)

        extracted_age = int(age_match.group(1)) if age_match else None
        current_age = calculate_current_age(extracted_age) if extracted_age else None

        return {
            "name": name_match.group(1).strip() if name_match else None,
            "age": current_age,
            "gender": gender_match.group(1) if gender_match else None
        }
    except Exception as e:
        logging.error(f"Ошибка при извлечении данных пациента: {e}")
        return {}


# Функция для вычисления текущего возраста на основе указанного возраста
def calculate_current_age(extracted_age: int) -> int:
    """
    Вычисляет текущий возраст на основе указанного возраста в документе.
    Предполагается, что указанный возраст был актуален на момент создания документа.
    """
    try:
        current_year = datetime.now().year
        # Предполагаем, что документ мог быть создан в течение последних 5 лет
        # и вычисляем примерный год рождения
        estimated_birth_year = current_year - extracted_age
        
        # Если год рождения кажется нереалистичным (до 1900), 
        # считаем что возраст указан для текущего года
        if estimated_birth_year < 1900:
            return extracted_age
        
        # Вычисляем текущий возраст
        current_age = current_year - estimated_birth_year
        
        # Проверяем разумность результата
        if current_age < 0 or current_age > 120:
            return extracted_age
        
        return current_age
    except Exception as e:
        logging.error(f"Ошибка при вычислении возраста: {e}")
        return extracted_age


# Функция для анализа изображения
async def analyze_image(image_url: str, query: str = "Что показано на этом медицинском изображении?") -> str:
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
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
        
        # Для анализа изображений используем модели с поддержкой изображений
        response, _, _ = await call_model_with_failover(
            messages=messages,
            model_type="vision",  # Указываем тип модели для анализа изображений
            system_prompt="Ты — медицинский эксперт по анализу изображений."
        )
        
        return response
    except Exception as e:
        logging.error(f"Ошибка при анализе изображения: {e}")
        return "Не удалось проанализировать изображение. Попробуйте еще раз."


# Функция для поиска в базе знаний
def search_knowledge_base(query: str) -> str:
    try:
        vector_results = vector_search(query)
        if vector_results:
            return "\n\n".join([f"Вопрос: {q}\nОтвет: {a}" for q, a, _ in vector_results])

        response = supabase.table("doc_knowledge_base").select("*").execute()
        results = [item["answer"] for item in response.data if query.lower() in item["question"].lower()]
        return "\n".join(results) if results else ""
    except Exception as e:
        logging.error(f"Ошибка при поиске в базе знаний: {e}")
        return ""


# Функция для создания профиля пациента
def create_patient_profile(user_id: str, name: str, age: int, gender: str, telegram_id: int = None) -> bool:
    try:
        profile_data = {
            "user_id": user_id,
            "name": name,
            "age": age,
            "gender": gender,
            "created_at": datetime.now().isoformat()
        }
        
        # Добавляем telegram_id если он передан
        if telegram_id:
            profile_data["telegram_id"] = telegram_id
            
        response = supabase.table("doc_patient_profiles").insert(profile_data).execute()
        return len(response.data) > 0
    except Exception as e:
        logging.error(f"Ошибка при создании профиля пациента: {e}")
        return False


# Функция для получения профиля пациента
def get_patient_profile(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = supabase.table("doc_patient_profiles").select("*").eq("user_id", user_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logging.error(f"Ошибка при получении профиля пациента: {e}")
        return None


# Функция для сохранения медицинских записей
def save_medical_record(user_id: str, record_type: str, content: str, source: str = "") -> bool:
    try:
        response = supabase.table("doc_medical_records").insert({
            "user_id": user_id,
            "record_type": record_type,
            "content": content,
            "source": source,
            "created_at": datetime.now().isoformat()
        }).execute()
        return len(response.data) > 0
    except Exception as e:
        logging.error(f"Ошибка при сохранении медицинской записи: {e}")
        return False


# Функция для получения медицинских записей
def get_medical_records(user_id: str, record_type: str = None) -> List[Dict[str, Any]]:
    try:
        query = supabase.table("doc_medical_records").select("*").eq("user_id", user_id)
        if record_type:
            query = query.eq("record_type", record_type)
        response = query.order("created_at", desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        logging.error(f"Ошибка при получении медицинских записей: {e}")
        return []


# Функция для сохранения в базу знаний
def save_to_knowledge_base(question: str, answer: str, source: str = ""):
    try:
        supabase.table("doc_knowledge_base").insert({
            "question": question,
            "answer": answer,
            "source": source,
            "created_at": datetime.now().isoformat()
        }).execute()
        save_to_vector_knowledge_base(question, answer, source)
    except Exception as e:
        logging.error(f"Ошибка при сохранении в базу знаний: {e}")


# Функция для сохранения обратной связи
def save_user_feedback(user_id: str, question: str, helped: bool):
    try:
        supabase.table("doc_user_feedback").insert({
            "user_id": user_id,
            "question": question,
            "helped": helped,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logging.error(f"Ошибка при сохранении обратной связи: {e}")


# Функция для генерации ответа с MOE подходом (старая версия, оставлена для совместимости)
async def generate_answer(question: str, context: str = "", history: List[Dict[str, str]] = None,
                          patient_data: Dict[str, Any] = None, user_id: int = None) -> str:
    answer, _, _ = await generate_answer_with_failover(question, context, history, patient_data, user_id)
    return answer


# Функция для поиска в интернете
async def search_web(query: str) -> str:
    try:
        response = tavily_client.search(query, max_results=3)
        return "\n".join([f"{result['content']}\nИсточник: {result['url']}" for result in response["results"]])
    except Exception as e:
        logging.error(f"Ошибка при поиске в интернете: {e}")
        return ""


# Функция для очистки состояния
async def clear_conversation_state(state: FSMContext, chat_id: int):
    try:
        scheduler.remove_job(f"reminder_{chat_id}")
    except:
        pass
    await state.clear()


# Обработчик команды /start
@dp.message(Command("start"))
async def start_command(message: types.Message, state: FSMContext):
    await clear_conversation_state(state, message.chat.id)
    profile = get_patient_profile(generate_user_uuid(message.from_user.id))
    if profile:
        await message.answer(
            f"👋 Здравствуйте, {profile['name']}! Я ваш ИИ-ассистент врача.\n\n"
            f"📊 Я могу помочь вам с анализом анализов, ответить на медицинские вопросы и хранить ваш анамнез.\n\n"
            f"💡 Просто задайте ваш вопрос, или загрузите анализы для анализа.\n\n"
            f"📊 Доступные команды:\n"
            f"/profile - ваш профиль\n"
            f"/stats - моя статистика помощи\n"
            f"/history - история обращений\n"
            f"/clear - очистить историю\n"
            f"/models - статус моделей",
            reply_markup=get_main_keyboard()
        )
    else:
        await message.answer(
            "👋 Здравствуйте! Я ваш ИИ-ассистент врача.\n\n"
            "📊 Я могу помочь вам с анализом анализов, ответить на медицинские вопросы и хранить ваш анамнез.\n\n"
            "💡 Для начала работы, пожалуйста, создайте профиль пациента.\n\n"
            "📊 Доступные команды:\n"
            "/profile - создать профиль\n"
            "/stats - моя статистика помощи\n"
            "/history - история обращений\n"
            "/clear - очистить историю\n"
            "/models - статус моделей",
            reply_markup=get_main_keyboard()
        )


# Обработчик команды /models для проверки статуса моделей
@dp.message(Command("models"))
async def models_command(message: types.Message):
    status_text = "🤖 <b>Статус моделей:</b>\n\n"

    for provider, config in MODEL_CONFIG.items():
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


# Обработчик загрузки файлов
@dp.message(F.document)
async def handle_document(message: types.Message, state: FSMContext):
    profile = get_patient_profile(generate_user_uuid(message.from_user.id))
    if message.document.mime_type == "application/pdf":
        file_id = message.document.file_id
        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/{file_path}"
        processing_msg = await message.answer("📊 Обрабатываю PDF файл...")

        pdf_text = await extract_text_from_pdf(file_url)
        if pdf_text:
            # Проверяем, есть ли профиль у пользователя перед сохранением
            if not profile:
                # Если профиля нет, сначала создаем анонимный профиль
                create_patient_profile(generate_user_uuid(message.from_user.id), "аноним", None, None, message.from_user.id)
                await message.answer("✅ Создан анонимный профиль для обработки PDF.")
            
            # Теперь сохраняем в медицинские записи (профиль уже существует)
            save_medical_record(
                user_id=generate_user_uuid(message.from_user.id),
                record_type="analysis",
                content=pdf_text[:2000],
                source=f"PDF file: {message.document.file_name}"
            )

            # Анализируем результаты анализов с помощью агента
            test_results = await test_agent.analyze_test_results(pdf_text)
            if test_results:
                # Сохраняем структурированные результаты
                await save_test_results(
                    user_id=generate_user_uuid(message.from_user.id),
                    test_results=test_results,
                    source=f"PDF file: {message.document.file_name}"
                )

                await processing_msg.edit_text("✅ PDF файл успешно обработан. Результаты анализов сохранены.")

                # Если профиля нет, пытаемся извлечь данные пациента
                if not profile:
                    patient_data = await extract_patient_data_from_text(pdf_text)
                    if patient_data and (
                            patient_data.get("name") or patient_data.get("age") or patient_data.get("gender")):
                        extracted_info = "📝 Я обнаружил(а) в вашем анализе следующие данные:\n\n"
                        if patient_data.get("name"):
                            extracted_info += f"👤 Имя: {patient_data['name']}\n"
                        if patient_data.get("age"):
                            extracted_info += f"🎂 Возраст: {patient_data['age']}\n"
                        if patient_data.get("gender"):
                            extracted_info += f"⚧️ Пол: {patient_data['gender']}\n"
                        extracted_info += "\nСоздать профиль с этими данными?"

                        await message.answer(
                            extracted_info,
                            reply_markup=InlineKeyboardBuilder().add(
                                types.InlineKeyboardButton(
                                    text="✅ Да, использовать",
                                    callback_data="use_extracted_data_pdf"
                                ),
                                types.InlineKeyboardButton(
                                    text="❌ Нет, создать анонимный профиль",
                                    callback_data="create_anonymous_profile_pdf"
                                )
                            ).as_markup()
                        )
                        await state.set_state(DoctorStates.confirming_profile)
                        await state.update_data(
                            extracted_patient_data=patient_data,
                            pdf_text=pdf_text
                        )
                        return

                # Предлагаем проанализировать результаты
                await message.answer(
                    "🔍 Хотите, чтобы я проанализировал(а) ваши анализы?",
                    reply_markup=InlineKeyboardBuilder().add(
                        types.InlineKeyboardButton(
                            text="✅ Да, проанализировать",
                            callback_data="analyze_pdf"
                        )
                    ).as_markup()
                )
                await state.set_state(DoctorStates.waiting_for_clarification)
                await state.update_data(pdf_text=pdf_text)
            else:
                await processing_msg.edit_text("😔 Не удалось извлечь результаты анализов из PDF файла.")
        else:
            await processing_msg.edit_text(
                "😔 Не удалось извлечь текст из PDF файла. Пожалуйста, попробуйте другой файл.")
    else:
        await message.answer("😔 Пожалуйста, загрузите PDF файл с анализами.")


# Обработчик изображений
@dp.message(F.photo)
async def handle_photo(message: types.Message, state: FSMContext):
    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    file_url = f"https://api.telegram.org/file/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/{file_path}"
    processing_msg = await message.answer("🔍 Анализирую изображение...")
    
    # Анализируем изображение для извлечения текста
    analysis_result = await analyze_image(file_url, "Извлеки все медицинские данные и информацию о пациенте с этого изображения. Верни текст с анализами и данными пациента.")
    await processing_msg.edit_text("✅ Изображение успешно проанализировано.")
    
    # Пытаемся извлечь данные пациента из анализа
    patient_data = await extract_patient_data_from_text(analysis_result)
    
    # Проверяем, есть ли профиль у пользователя
    profile = get_patient_profile(generate_user_uuid(message.from_user.id))
    
    if not profile:
        # Если профиля нет, предлагаем создать его на основе извлеченных данных
        if patient_data and (patient_data.get("name") or patient_data.get("age") or patient_data.get("gender")):
            extracted_info = "📝 Я обнаружил(а) в вашем анализе следующие данные:\n\n"
            if patient_data.get("name"):
                extracted_info += f"👤 Имя: {patient_data['name']}\n"
            if patient_data.get("age"):
                extracted_info += f"🎂 Возраст: {patient_data['age']}\n"
            if patient_data.get("gender"):
                extracted_info += f"⚧️ Пол: {patient_data['gender']}\n"
            
            extracted_info += "\nИспользовать эти данные для создания вашего профиля?"
            
            await message.answer(
                extracted_info,
                reply_markup=InlineKeyboardBuilder().add(
                    types.InlineKeyboardButton(
                        text="✅ Да, использовать",
                        callback_data="use_extracted_data"
                    ),
                    types.InlineKeyboardButton(
                        text="❌ Нет, создать анонимный профиль",
                        callback_data="create_anonymous_profile"
                    )
                ).as_markup()
            )
            
            # Сохраняем извлеченные данные и результат анализа в состоянии
            await state.set_state(DoctorStates.confirming_profile)
            await state.update_data(
                extracted_patient_data=patient_data,
                analysis_result=analysis_result
            )
        else:
            # Если не удалось извлечь данные, создаем анонимный профиль
            create_patient_profile(generate_user_uuid(message.from_user.id), "аноним", None, None, message.from_user.id)
            await message.answer(
                "✅ Создан анонимный профиль.\n\n"
                f"📊 <b>Результат анализа:</b>\n\n{escape_html(analysis_result)}\n\n"
                f"⚠️ Помните, что это автоматический анализ, и он не заменяет консультацию специалиста.",
                parse_mode="HTML"
            )
            
            # Сохраняем в медицинские записи ПОСЛЕ создания профиля
            save_medical_record(
                user_id=generate_user_uuid(message.from_user.id),
                record_type="image_analysis",
                content=analysis_result,
                source="Изображение из Telegram"
            )
    else:
        # Если профиль уже есть, сначала сохраняем в медицинские записи
        save_medical_record(
            user_id=generate_user_uuid(message.from_user.id),
            record_type="image_analysis",
            content=analysis_result,
            source="Изображение из Telegram"
        )
        
        # Затем показываем результат анализа
        await message.answer(
            f"📊 <b>Результат анализа:</b>\n\n{escape_html(analysis_result)}\n\n"
            f"⚠️ Помните, что это автоматический анализ, и он не заменяет консультацию специалиста.",
            parse_mode="HTML"
            )


# Основной обработчик сообщений
@dp.message(F.text)
async def handle_message(message: types.Message, state: FSMContext):
    question = message.text
    chat_id = message.chat.id
    user_id = message.from_user.id
    data = await state.get_data()
    
    # Получаем историю диалога и данные пациента
    history = data.get("history", [])
    profile = get_patient_profile(generate_user_uuid(user_id))
    
    # Добавляем текущее сообщение в историю
    history.append({"role": "user", "content": question})
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]
        await message.answer(
            "🔄 История диалога стала слишком длинной, я удалил самые старые сообщения для оптимизации.")
    
    # Проверяем, есть ли медицинские записи у пользователя
    has_medical_records = len(get_medical_records(generate_user_uuid(user_id))) > 0
    
    # Проверяем, нужно ли уточнять информацию
    clarification_count = data.get("clarification_count", 0)
    is_enough, clarification_question, ai_mode = await clarification_agent.analyze_and_ask(
        question, history, profile, clarification_count, has_medical_records
    )
    
    # Если нужно уточнить информацию
    if not is_enough and clarification_question:
        await message.answer(clarification_question)
        await state.set_state(DoctorStates.waiting_for_clarification)
        await state.update_data(
            original_question=question,
            clarification_count=clarification_count + 1,
            history=history,
            user_id=user_id
        )
        return
    
    # Если информации достаточно, продолжаем стандартную обработку
    processing_msg = await message.answer("🔍 Ищу информацию по вашему вопросу...")
    
    # Проверяем, есть ли в вопросе запрос на анализ анализов
    analysis_keywords = ['анализ', 'анализы', 'результат', 'показатель', 'кровь', 'моча', 'биохимия', 'общий анализ']
    test_context = ""
    if any(keyword in question.lower() for keyword in analysis_keywords):
        # Получаем сводку по анализам от агента
        test_summary = await test_agent.get_test_summary(generate_user_uuid(user_id))
        if test_summary:
            test_context = f"\n\n📊 {test_summary}"
    
    # Определяем системный промпт в зависимости от режима ИИ
    if ai_mode == "doctor":
        system_prompt = """Ты — ИИ-врач главный, опытный медицинский специалист с глубокими знаниями в медицине. 
        Твоя задача — давать профессиональные медицинские консультации, анализировать результаты анализов, 
        интерпретировать медицинские данные и предоставлять квалифицированные рекомендации. 
        Используй всю доступную информацию о пациенте, историю диалога и медицинские записи для точной диагностики и консультации."""
        mode_indicator = "👨‍⚕️ ИИ-врач главный"
    else:
        system_prompt = """Ты — ИИ-ассистент врача, который помогает собрать информацию и подготовить данные для консультации. 
        Твоя задача — помогать пользователям с медицинскими вопросами, анализировать их анализы и предоставлять информацию о здоровье. 
        Отвечай максимально точно и информативно, используя предоставленный контекст."""
        mode_indicator = "👩‍⚕️ ИИ-ассистент врача"
    
    # 1. Сначала ищем в авторитетных медицинских источниках
    medical_context = await search_medical_sources(question)
    if medical_context:
        await processing_msg.edit_text(f"📚 Найдено в медицинских источниках. Генерирую ответ... ({mode_indicator})")
        answer, provider, metadata = await generate_answer_with_failover(
            question, medical_context + test_context, history, profile, str(user_id), system_prompt
        )
        source = "авторитетных медицинских источников"
    else:
        # 2. Если не нашли в медицинских источниках, ищем в своей базе знаний
        await processing_msg.edit_text(f"🗂️ Ищу в накопленной базе знаний... ({mode_indicator})")
        kb_context = search_knowledge_base(question)
        if kb_context:
            await processing_msg.edit_text(f"💡 Найдено в базе знаний. Генерирую ответ... ({mode_indicator})")
            answer, provider, metadata = await generate_answer_with_failover(
                question, kb_context + test_context, history, profile, str(user_id), system_prompt
            )
            source = "накопленной базы знаний"
        else:
            # 3. Если нигде не нашли, ищем в интернете
            await processing_msg.edit_text(f"🌐 Ищу дополнительную информацию в интернете... ({mode_indicator})")
            web_context = await search_web(f"{question} медицина здоровье")
            answer, provider, metadata = await generate_answer_with_failover(
                question, web_context + test_context, history, profile, str(user_id), system_prompt
            )
            source = "интернета"
    
    await processing_msg.delete()
    history.append({"role": "assistant", "content": answer})
    
    # Добавляем индикатор режима ИИ в ответ
    mode_text = f"\n\n{mode_indicator}" if ai_mode else ""
    
    await message.answer(f"{escape_html(answer)}\n\n📖 <b>Источник:</b> {escape_html(source)}{mode_text}", parse_mode="HTML")
    await message.answer("❓ Помог ли вам мой ответ?", reply_markup=get_feedback_keyboard())
    await state.set_state(DoctorStates.waiting_for_feedback)
    await state.update_data(
        question=question,
        answer=answer,
        source=source,
        provider=provider,
        metadata=metadata,
        attempts=0,
        user_id=str(user_id),
        history=history
    )
    scheduler.add_job(
        send_reminder,
        "date",
        run_date=datetime.now() + timedelta(hours=1),
        args=[chat_id],
        id=f"reminder_{chat_id}"
    )


# Обработчик кнопок обратной связи
@dp.callback_query(F.data.in_(["feedback_yes", "feedback_no", "search_more"]))
async def handle_feedback_callback(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    question = data["question"]
    answer = data["answer"]
    source = data["source"]
    provider = data.get("provider", "")
    metadata = data.get("metadata", {})
    attempts = data.get("attempts", 0)
    user_id = data.get("user_id", str(callback.from_user.id))
    history = data.get("history", [])
    chat_id = callback.message.chat.id

    if callback.data == "feedback_yes":
        # Сохраняем успешный ответ с цепочкой размышлений
        await save_successful_response(user_id, question, answer, provider, metadata, history)

        if source != "авторитетных медицинских источников":
            save_to_knowledge_base(question, answer, source)

        save_user_feedback(user_id, question, True)
        await callback.message.edit_text(
            "✅ Отлично! Я рад(а), что смог(ла) помочь.\n"
            "Если у вас появятся еще вопросы — обращайтесь! 😊\n\n"
            "⚠️ Помните, что для точной диагностики и лечения необходима консультация врача.",
            reply_markup=get_main_keyboard()
        )
        await clear_conversation_state(state, chat_id)
    elif callback.data == "feedback_no":
        if attempts < 2:
            await callback.message.edit_text(
                "😔 Понимаю, что ответ не помог. Что вы хотите сделать дальше?",
                reply_markup=get_clarification_keyboard()
            )
            await state.update_data(attempts=attempts + 1)
        else:
            save_user_feedback(user_id, question, False)
            await callback.message.edit_text(
                "😔 К сожалению, я не смог(ла) найти достаточно информации по вашему вопросу.\n\n"
                "Рекомендую:\n"
                "• 🏥 Обратиться к врачу для очной консультации\n"
                "• 🔍 Попробовать поискать в надежных медицинских источниках\n"
                "• 📊 Загрузить анализы для более точного анализа",
                reply_markup=get_main_keyboard()
            )
            await clear_conversation_state(state, chat_id)
    elif callback.data == "search_more":
        await callback.message.edit_text("🔍 Ищу дополнительную информацию...")
        profile = get_patient_profile(generate_user_uuid(user_id))
        web_context = await search_web(f"{question} медицина диагноз лечение")
        new_answer, new_provider, new_metadata = await generate_answer_with_failover(question, web_context, history,
                                                                                      profile, generate_user_uuid(user_id))
        history.append({"role": "assistant", "content": new_answer})
        await state.update_data(history=history)
        await callback.message.edit_text(
            f"{escape_html(new_answer)}\n\n📖 <b>Источник:</b> дополнительный поиск в интернете",
            parse_mode="HTML"
        )
        await bot.send_message(
            chat_id,
            "❓ Помог ли вам этот новый ответ?",
            reply_markup=get_feedback_keyboard()
        )
        await state.update_data(
            answer=new_answer,
            provider=new_provider,
            metadata=new_metadata,
            source="интернета (дополнительный поиск)",
            attempts=attempts + 1
        )


# Обработчик создания профиля на основе извлеченных данных
@dp.callback_query(F.data.in_(["use_extracted_data", "create_anonymous_profile"]))
async def handle_profile_creation_callback(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    patient_data = data.get("extracted_patient_data", {})
    analysis_result = data.get("analysis_result", "")
    
    if callback.data == "use_extracted_data":
        # Используем извлеченные данные
        name = patient_data.get("name", "аноним")
        age = patient_data.get("age")
        gender = patient_data.get("gender")
        
        # Создаем профиль
        if create_patient_profile(generate_user_uuid(callback.from_user.id), name, age, gender, callback.from_user.id):
            await callback.message.edit_text(
                f"✅ Профиль успешно создан!\n\n"
                f"👤 <b>Ваш профиль:</b>\n"
                f"📝 Имя: {name}\n"
                f"🎂 Возраст: {age if age else 'не указан'}\n"
                f"⚧️ Пол: {gender if gender else 'не указан'}\n\n"
                f"📊 <b>Результат анализа:</b>\n\n{escape_html(analysis_result)}\n\n"
                f"⚠️ Помните, что это автоматический анализ, и он не заменяет консультацию специалиста.",
                parse_mode="HTML"
            )
        else:
            await callback.message.edit_text("😔 Не удалось создать профиль. Пожалуйста, попробуйте еще раз.")
    else:
        # Создаем анонимный профиль
        create_patient_profile(generate_user_uuid(callback.from_user.id), "аноним", None, None, callback.from_user.id)
        await callback.message.edit_text(
            "✅ Создан анонимный профиль.\n\n"
            f"📊 <b>Результат анализа:</b>\n\n{escape_html(analysis_result)}\n\n"
            f"⚠️ Помните, что это автоматический анализ, и он не заменяет консультацию специалиста.",
            parse_mode="HTML"
        )
    
    await state.clear()

# Обработчик создания профиля на основе извлеченных данных из PDF
@dp.callback_query(F.data.in_(["use_extracted_data_pdf", "create_anonymous_profile_pdf"]))
async def handle_pdf_profile_creation_callback(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    patient_data = data.get("extracted_patient_data", {})
    pdf_text = data.get("pdf_text", "")
    test_results = data.get("test_results", [])
    
    if callback.data == "use_extracted_data_pdf":
        # Используем извлеченные данные
        name = patient_data.get("name", "аноним")
        age = patient_data.get("age")
        gender = patient_data.get("gender")
        
        # Создаем профиль
        if create_patient_profile(generate_user_uuid(callback.from_user.id), name, age, gender, callback.from_user.id):
            await callback.message.edit_text(
                f"✅ Профиль успешно создан!\n\n"
                f"👤 <b>Ваш профиль:</b>\n"
                f"📝 Имя: {name}\n"
                f"🎂 Возраст: {age if age else 'не указан'}\n"
                f"⚧️ Пол: {gender if gender else 'не указан'}\n\n"
                "🔍 Хотите, чтобы я проанализировал(а) ваши анализы?",
                reply_markup=InlineKeyboardBuilder().add(
                    types.InlineKeyboardButton(
                        text="✅ Да, проанализировать",
                        callback_data="analyze_pdf"
                    )
                ).as_markup()
            )
            # Сохраняем данные для анализа
            await state.set_state(DoctorStates.waiting_for_clarification)
            await state.update_data(pdf_text=pdf_text)
        else:
            await callback.message.edit_text("😔 Не удалось создать профиль. Пожалуйста, попробуйте еще раз.")
    else:
        # Создаем анонимный профиль
        create_patient_profile(generate_user_uuid(callback.from_user.id), "аноним", None, None, callback.from_user.id)
        await callback.message.edit_text(
            "✅ Создан анонимный профиль.\n\n"
            "🔍 Хотите, чтобы я проанализировал(а) ваши анализы?",
            reply_markup=InlineKeyboardBuilder().add(
                types.InlineKeyboardButton(
                    text="✅ Да, проанализировать",
                    callback_data="analyze_pdf"
                )
            ).as_markup()
        )
        # Сохраняем данные для анализа
        await state.set_state(DoctorStates.waiting_for_clarification)
        await state.update_data(pdf_text=pdf_text)


# Обработчик кнопок уточнения
@dp.callback_query(F.data.in_(
    ["clarify_question", "upload_tests", "try_again", "analyze_pdf", "create_extracted_profile", "manual_profile"]))
async def handle_clarification_callback(callback: types.CallbackQuery, state: FSMContext):
    if callback.data == "clarify_question":
        await callback.message.edit_text(
            "🔄 Пожалуйста, уточните ваш вопрос или опишите симптомы более подробно."
        )
        await state.set_state(DoctorStates.waiting_for_clarification)
    elif callback.data == "upload_tests":
        await callback.message.edit_text(
            "📊 Пожалуйста, загрузите PDF файл с вашими анализами или отправьте фото медицинского документа."
        )
        await state.set_state(DoctorStates.waiting_for_file)
    elif callback.data == "try_again":
        data = await state.get_data()
        question = data["question"]
        history = data.get("history", [])
        profile = get_patient_profile(generate_user_uuid(callback.from_user.id))

        await callback.message.edit_text("🔄 Пробую найти другой ответ...")
        web_context = await search_web(f"{question} медицина здоровье лечение")
        new_answer, new_provider, new_metadata = await generate_answer_with_failover(question, web_context, history,
                                                                                      profile, generate_user_uuid(callback.from_user.id))
        history.append({"role": "assistant", "content": new_answer})
        await state.update_data(history=history)
        await callback.message.edit_text(
            f"{escape_html(new_answer)}\n\n📖 <b>Источник:</b> дополнительный поиск в интернете",
            parse_mode="HTML",
            reply_markup=get_feedback_keyboard()
        )
        await state.update_data(
            answer=new_answer,
            provider=new_provider,
            metadata=new_metadata,
            source="интернета"
        )
        await state.set_state(DoctorStates.waiting_for_feedback)
    elif callback.data == "analyze_pdf":
        data = await state.get_data()
        pdf_text = data.get("pdf_text", "")
        if pdf_text:
            await callback.message.edit_text("📊 Анализирую результаты анализов...")
            profile = get_patient_profile(generate_user_uuid(callback.from_user.id))

            # Используем агента для анализа
            analysis_result = await test_agent.get_test_summary(generate_user_uuid(callback.from_user.id))
            if analysis_result:
                save_medical_record(
                    user_id=generate_user_uuid(callback.from_user.id),
                    record_type="analysis_result",
                    content=analysis_result,
                    source="Анализ PDF файла"
                )
                await callback.message.edit_text(
                    f"📊 <b>Результат анализа:</b>\n\n{escape_html(analysis_result)}\n\n"
                    f"⚠️ Помните, что это автоматический анализ, и он не заменяет консультацию специалиста.",
                    parse_mode="HTML",
                    reply_markup=get_main_keyboard()
                )
                await state.clear()
            else:
                await callback.message.edit_text(
                    "😔 Не удалось проанализировать результаты анализов. Пожалуйста, попробуйте еще раз."
                )
        else:
            await callback.message.edit_text(
                "😔 Не удалось найти данные для анализа. Пожалуйста, загрузите PDF файл с анализами снова."
            )
            await state.set_state(DoctorStates.waiting_for_file)
    elif callback.data == "create_extracted_profile":
        data = await state.get_data()
        patient_data = data.get("extracted_patient_data", {})
        if patient_data and (patient_data.get("name") or patient_data.get("age") or patient_data.get("gender")):
            missing_data = []
            if not patient_data.get("name"):
                missing_data.append("имя")
            if not patient_data.get("age"):
                missing_data.append("возраст")
            if not patient_data.get("gender"):
                missing_data.append("пол")

            if missing_data:
                await callback.message.edit_text(
                    f"📝 Для создания профиля не хватает следующих данных: {', '.join(missing_data)}.\n\n"
                    f"Пожалуйста, отправьте недостающую информацию в формате:\n"
                    f"<b>Имя: [ваше имя]</b>\n"
                    f"<b>Возраст: [ваш возраст]</b>\n"
                    f"<b>Пол: [М/Ж]</b>",
                    parse_mode="HTML"
                )
                await state.set_state(DoctorStates.waiting_for_patient_id)
                await state.update_data(extracted_patient_data=patient_data)
            else:
                if create_patient_profile(generate_user_uuid(callback.from_user.id), patient_data['name'], patient_data['age'],
                                          patient_data['gender'], callback.from_user.id):
                    await callback.message.edit_text(
                        f"✅ Профиль успешно создан!\n\n"
                        f"👤 <b>Ваш профиль:</b>\n"
                        f"📝 Имя: {patient_data['name']}\n"
                        f"🎂 Возраст: {patient_data['age']}\n"
                        f"⚧️ Пол: {'Мужской' if patient_data['gender'] == 'М' else 'Женский'}\n\n"
                        f"Теперь вы можете задавать медицинские вопросы и загружать анализы.",
                        parse_mode="HTML",
                        reply_markup=get_main_keyboard()
                    )

                    pdf_text = data.get("pdf_text", "")
                    if pdf_text:
                        await bot.send_message(
                            callback.message.chat.id,
                            "🔍 Хотите, чтобы я проанализировал(а) ваши анализы?",
                            reply_markup=InlineKeyboardBuilder().add(
                                types.InlineKeyboardButton(
                                    text="✅ Да, проанализировать",
                                    callback_data="analyze_pdf"
                                )
                            ).as_markup()
                        )
                        await state.set_state(DoctorStates.waiting_for_clarification)
                        await state.update_data(pdf_text=pdf_text)

                    await state.clear()
                else:
                    await callback.message.edit_text(
                        "😔 Не удалось создать профиль. Пожалуйста, попробуйте ввести данные вручную с помощью команды /profile.")
        else:
            await callback.message.edit_text(
                "😔 Не удалось извлечь данные пациента. Пожалуйста, создайте профиль вручную с помощью команды /profile.")
    elif callback.data == "manual_profile":
        await callback.message.edit_text(
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


# Обработчик кнопок главного меню
@dp.callback_query(F.data.in_(["my_tests", "my_history", "create_profile"]))
async def handle_main_menu_callback(callback: types.CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)

    if callback.data == "my_tests":
        tests = get_patient_tests(user_id)
        if tests:
            tests_text = "📊 <b>Ваши анализы:</b>\n\n"
            for test in tests[:10]:
                status = "⚠️" if test.get('is_abnormal') else "✅"
                tests_text += f"{status} {test['test_name']}: {test['value']} {test['unit'] or ''} (норма: {test['reference_range'] or 'не указана'}) от {test['test_date'] or 'дата не указана'}\n"
                if test.get('notes'):
                    tests_text += f"   💬 {test['notes']}\n"
                tests_text += "\n"
            await callback.message.edit_text(
                tests_text,
                parse_mode="HTML",
                reply_markup=get_main_keyboard()
            )
        else:
            await callback.message.edit_text(
                "📊 У вас пока нет загруженных анализов.\n\n"
                "Вы можете загрузить PDF файл с анализами или отправить фото медицинского документа.",
                reply_markup=get_main_keyboard()
            )
    elif callback.data == "my_history":
        records = get_medical_records(user_id)
        if records:
            history_text = "📝 <b>Ваша медицинская история:</b>\n\n"
            for record in records[:5]:
                record_type = record.get("record_type", "запись")
                history_text += f"📅 {record['created_at'][:10]} ({record_type}): {record['content'][:100]}...\n\n"
            await callback.message.edit_text(
                history_text,
                parse_mode="HTML",
                reply_markup=get_main_keyboard()
            )
        else:
            await callback.message.edit_text(
                "📝 У вас пока нет медицинской истории.\n\n"
                "Она будет формироваться по мере ваших обращений и загрузки анализов.",
                reply_markup=get_main_keyboard()
            )
    elif callback.data == "create_profile":
        profile = get_patient_profile(user_id)
        if profile:
            await callback.message.edit_text(
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
            await callback.message.edit_text(
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


# Обработчик уточненного вопроса
@dp.message(DoctorStates.waiting_for_clarification)
async def handle_clarification(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    data = await state.get_data()
    
    # Получаем сохраненные данные
    original_question = data.get("original_question", "")
    history = data.get("history", [])
    clarification_count = data.get("clarification_count", 0)
    profile = get_patient_profile(generate_user_uuid(user_id))
    
    # Добавляем ответ пользователя в историю
    history.append({"role": "user", "content": message.text})
    
    # Проверяем, нужно ли еще уточнять информацию
    is_enough, clarification_question = await clarification_agent.analyzeAnd_ask(
        original_question, history, profile, clarification_count
    )
    
    # Если нужно еще уточнить
    if not is_enough and clarification_question:
        await message.answer(clarification_question)
        await state.update_data(
            clarification_count=clarification_count + 1,
            history=history
        )
        return
    
    # Если информации достаточно или достигнут лимит уточнений
    processing_msg = await message.answer("🔍 Ищу информацию по вашему вопросу...")
    
    # Проверяем, есть ли в вопросе запрос на анализ анализов
    analysis_keywords = ['анализ', 'анализы', 'результат', 'показатель', 'кровь', 'моча', 'биохимия', 'общий анализ']
    test_context = ""
    if any(keyword in original_question.lower() for keyword in analysis_keywords):
        # Получаем сводку по анализам от агента
        test_summary = await test_agent.get_test_summary(generate_user_uuid(user_id))
        if test_summary:
            test_context = f"\n\n📊 {test_summary}"
    
    # 1. Сначала ищем в авторитетных медицинских источниках
    medical_context = await search_medical_sources(original_question)
    if medical_context:
        await processing_msg.edit_text("📚 Найдено в медицинских источниках. Генерирую ответ...")
        answer, provider, metadata = await generate_answer_with_failover(
            original_question, medical_context + test_context, history, profile, str(user_id)
        )
        source = "авторитетных медицинских источников"
    else:
        # 2. Если не нашли в медицинских источниках, ищем в своей базе знаний
        await processing_msg.edit_text("🗂️ Ищу в накопленной базе знаний...")
        kb_context = search_knowledge_base(original_question)
        if kb_context:
            await processing_msg.edit_text("💡 Найдено в базе знаний. Генерирую ответ...")
            answer, provider, metadata = await generate_answer_with_failover(
                original_question, kb_context + test_context, history, profile, str(user_id)
            )
            source = "накопленной базы знаний"
        else:
            # 3. Если нигде не нашли, ищем в интернете
            await processing_msg.edit_text("🌐 Ищу дополнительную информацию в интернете...")
            web_context = await search_web(f"{original_question} медицина здоровье")
            answer, provider, metadata = await generate_answer_with_failover(
                original_question, web_context + test_context, history, profile, str(user_id)
            )
            source = "интернета"
    
    await processing_msg.delete()
    history.append({"role": "assistant", "content": answer})
    await message.answer(f"{escape_html(answer)}\n\n📖 <b>Источник:</b> {escape_html(source)}", parse_mode="HTML")
    await message.answer("❓ Помог ли вам мой ответ?", reply_markup=get_feedback_keyboard())
    await state.set_state(DoctorStates.waiting_for_feedback)
    await state.update_data(
        question=original_question,
        answer=answer,
        source=source,
        provider=provider,
        metadata=metadata,
        attempts=0,
        user_id=str(user_id),
        history=history
    )
    scheduler.add_job(
        send_reminder,
        "date",
        run_date=datetime.now() + timedelta(hours=1),
        args=[message.chat.id],
        id=f"reminder_{message.chat.id}"
    )


# Обработчик загрузки файла в состоянии ожидания
@dp.message(DoctorStates.waiting_for_file)
async def handle_file_upload(message: types.Message, state: FSMContext):
    if message.document:
        await handle_document(message, state)
    elif message.photo:
        await handle_photo(message, state)
    else:
        await message.answer("😔 Пожалуйста, загрузите PDF файл с анализами или отправьте фото медицинского документа.")


# Функция для отложенного напоминания
async def send_reminder(chat_id: int):
    try:
        await bot.send_message(
            chat_id,
            "🔔 Напоминаю: помог ли вам мой предыдущий ответ?",
            reply_markup=get_feedback_keyboard()
        )
    except Exception as e:
        logging.error(f"Ошибка при отправке напоминания: {e}")


# Планировщик для отложенных напоминаний и сброса токенов
@dp.startup()
async def on_startup():
    scheduler.start()

    # Добавляем задачу для ежедневного сброса счетчиков токенов в полночь
    scheduler.add_job(
        reset_token_usage,
        "cron",
        hour=0,
        minute=0,
        id="reset_token_usage"
    )


@dp.shutdown()
async def on_shutdown():
    scheduler.shutdown()


# Запуск бота
async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())