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
# Vision модели (qwen, gemini) используются только для анализа изображений
# Text модели (deepseek, gpt, glm, kimi) используются для текстовых задач
# 
# ВАЖНО: Разные провайдеры используют разные названия параметров:
# - OpenAI/OpenRouter: max_tokens
# - Cerebras: max_tokens (не max_completion_tokens)
# - Groq: max_tokens
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
            {"name": "qwen-3-235b-a22b-thinking-2507", "priority": 1, "type": "text"},
            {"name": "qwen-3-235b-a22b-instruct-2507", "priority": 2, "type": "text"}

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
    generated_uuid = str(uuid.uuid5(telegram_namespace, str(telegram_user_id)))
    
    logging.info(f"Сгенерирован UUID для Telegram user ID {telegram_user_id}: {generated_uuid}")
    
    return generated_uuid

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
    updating_profile = State()


# Функция для экранирования HTML
def escape_html(text: str) -> str:
    logging.debug(f"Экранирование HTML для текста длиной {len(text)} символов")
    
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    
    logging.debug(f"HTML экранирован, результат длиной {len(escaped)} символов")
    return escaped


# Функция для проверки доступности модели
async def check_model_availability(provider: str, model_name: str) -> bool:
    """Проверяет доступность модели и наличие токенов"""
    try:
        logging.info(f"Проверка доступности модели {model_name} у провайдера {provider}")
        
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
                    is_available = model_name in available_models
                    logging.info(f"Модель {model_name} доступна в OpenRouter: {is_available}")
                    return is_available
                else:
                    logging.warning(f"Ошибка при проверке доступности модели OpenRouter: {response.status_code}")
            except Exception as e:
                logging.error(f"Ошибка при проверке доступности модели OpenRouter: {e}")

        # Для других провайдеров просто проверяем, что API ключ существует
        logging.info(f"Модель {model_name} считается доступной для провайдера {provider}")
        return True
    except Exception as e:
        logging.error(f"Ошибка при проверке доступности модели {model_name} у провайдера {provider}: {e}")
        return False


# Функция для обновления счетчика использованных токенов
def update_token_usage(provider: str, tokens_used: int):
    """Обновляет счетчик использованных токенов для провайдера"""
    if provider in TOKEN_LIMITS:
        TOKEN_LIMITS[provider]["used_today"] += tokens_used
        logging.info(f"Обновлен счетчик токенов для {provider}: +{tokens_used}, всего сегодня: {TOKEN_LIMITS[provider]['used_today']}")
    else:
        logging.warning(f"Провайдер {provider} не найден в TOKEN_LIMITS")


# Функция для сброса счетчиков токенов (можно вызывать раз в день)
def reset_token_usage():
    """Сбрасывает ежедневные счетчики токенов"""
    logging.info("Сброс счетчиков токенов")
    
    for provider in TOKEN_LIMITS:
        old_value = TOKEN_LIMITS[provider]["used_today"]
        TOKEN_LIMITS[provider]["used_today"] = 0
        logging.info(f"Сброшен счетчик токенов для {provider}: {old_value} -> 0")
    
    logging.info("Все счетчики токенов сброшены")


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
    logging.info(f"call_model_with_failover: тип модели: {model_type}, предпочтение: {model_preference}")
    logging.info(f"Количество сообщений: {len(messages)}")
    
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
    
    logging.info(f"Всего доступных моделей: {len(all_models)}")
    
    # Фильтруем модели по типу, если указан
    if model_type:
        original_count = len(all_models)
        all_models = [m for m in all_models if m.get("type") == model_type]
        logging.info(f"Отфильтровано по типу '{model_type}': {len(all_models)} из {original_count}")
        
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
            logging.info(f"Восстановлено общее количество моделей: {len(all_models)}")
    
    # Сортируем по приоритету
    all_models.sort(key=lambda x: x["priority"])
    logging.info(f"Модели отсортированы по приоритету")
    
    # Если указана предпочтительная модель, перемещаем её в начало
    if model_preference:
        preferred_models = [m for m in all_models if m["name"] == model_preference]
        other_models = [m for m in all_models if m["name"] != model_preference]
        all_models = preferred_models + other_models
        logging.info(f"Предпочтительная модель '{model_preference}' перемещена в начало списка")
    
    last_error = None
    logging.info(f"Начинаю попытки вызова моделей, всего моделей: {len(all_models)}")
    
    # Пробуем модели в порядке приоритета
    for i, model_info in enumerate(all_models):
        provider = model_info["provider"]
        model_name = model_info["name"]
        client = model_info["client"]
        
        logging.info(f"Попытка {i+1}/{len(all_models)}: модель {model_name} от провайдера {provider}")
        
        # Проверяем доступность модели
        if not await check_model_availability(provider, model_name):
            logging.info(f"Модель {model_name} провайдера {provider} недоступна, пробуем следующую")
            continue
        
        try:
            logging.info(f"Пробую модель {model_name} от провайдера {provider}")
            
            # Дополнительная диагностика для Cerebras
            if provider == "cerebras":
                config = MODEL_CONFIG[provider]
                logging.info(f"Cerebras API Key: {config.get('api_key', '')[:10]}...")
                logging.info(f"Cerebras Base URL: {config.get('base_url', '')}")
                logging.info(f"Client initialized: {config.get('client') is not None}")
            
            # Добавляем системный промпт, если он указан
            if system_prompt:
                # Проверяем, есть ли уже системный промпт в сообщениях
                has_system = any(msg.get("role") == "system" for msg in messages)
                if not has_system:
                    messages = [{"role": "system", "content": system_prompt}] + messages
                    logging.info("Добавлен системный промпт в сообщения")
            
            # Добавляем заголовки для OpenRouter
            extra_headers = {}
            if provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://github.com/vokforever/ai-doctor",
                    "X-Title": "AI Doctor Bot"
                }
                logging.info("Добавлены специальные заголовки для OpenRouter")
            
            # Выполняем запрос
            # Для Qwen 3 235B Thinking модели добавляем специальные параметры
            extra_params = {}
            if provider == "cerebras" and "qwen-3-235b" in model_name:
                extra_params = {
                    "max_tokens": 64000,  # Cerebras API использует max_tokens, не max_completion_tokens
                    "temperature": 0.7,
                    "top_p": 0.9
                }
                logging.info(f"Добавлены специальные параметры для Qwen 3 235B: {extra_params}")
            
            logging.info(f"Вызываю модель {model_name} с {len(messages)} сообщениями")
            
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **extra_params,
                **({"extra_headers": extra_headers} if extra_headers else {})
            )
            
            logging.info(f"Модель {model_name} успешно ответила")
            
            # Получаем ответ
            current_answer = completion.choices[0].message.content
            logging.info(f"Получен ответ длиной {len(current_answer)} символов")
            
            # Для некоторых моделей (например, Cerebras) может быть цепочка размышлений
            thinking_process = ""
            if provider == "cerebras" and hasattr(completion.choices[0], 'thinking'):
                thinking_process = completion.choices[0].thinking
                logging.info("Найдена цепочка размышлений в choices[0].thinking")
            elif provider == "cerebras" and hasattr(completion.choices[0].message, 'thinking'):
                thinking_process = completion.choices[0].message.thinking
                logging.info("Найдена цепочка размышлений в choices[0].message.thinking")
            
            # Обновляем счетчик токенов (если есть информация)
            if hasattr(completion, 'usage') and completion.usage:
                tokens_used = completion.usage.total_tokens
                update_token_usage(provider, tokens_used)
                logging.info(f"Использовано токенов {provider}: {tokens_used}")
            
            # Сохраняем информацию о модели
            metadata = {
                "provider": provider,
                "model": model_name,
                "type": model_info.get("type", "text"),
                "thinking": thinking_process,
                "usage": getattr(completion, 'usage', None)
            }
            
            logging.info(f"Успешно завершена работа с моделью {model_name} от провайдера {provider}")
            
            # Возвращаем первый успешный ответ
            return current_answer, provider, metadata
        except Exception as e:
            last_error = e
            error_msg = f"Ошибка при использовании модели {model_name} от провайдера {provider}: {e}"
            
            logging.error(f"Ошибка при вызове модели {model_name}: {e}")
            
            # Дополнительная диагностика для Cerebras
            if provider == "cerebras":
                error_msg += f"\nПроверьте:\n"
                error_msg += f"- Правильность API ключа\n"
                error_msg += f"- Доступность модели {model_name}\n"
                error_msg += f"- Лимиты токенов\n"
                error_msg += f"- Статус API Cerebras"
                
                # Проверяем конкретные ошибки Cerebras
                if "model_not_found" in str(e):
                    error_msg += f"\n❌ Модель {model_name} не найдена. Проверьте правильность названия."
                    logging.error(f"Модель {model_name} не найдена в Cerebras")
                elif "authentication" in str(e).lower():
                    error_msg += f"\n❌ Ошибка аутентификации. Проверьте API ключ."
                    logging.error("Ошибка аутентификации в Cerebras")
                elif "rate_limit" in str(e).lower():
                    error_msg += f"\n❌ Превышен лимит запросов."
                    logging.error("Превышен лимит запросов в Cerebras")
            
            logging.warning(error_msg)
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


# Инициализация агента уточнения
clarification_agent = ClarificationAgent()


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
        logging.info(f"Сохранение успешного ответа для пользователя {user_id}")
        logging.info(f"Вопрос: {question[:100]}...")
        logging.info(f"Провайдер: {provider}")
        
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
            logging.info(f"История диалога: {len(conversation_history)} сообщений")

        # Сохраняем в базу данных
        response = supabase.table("doc_successful_responses").insert(save_data).execute()

        if response.data:
            logging.info("Успешный ответ сохранен в базу данных")
        else:
            logging.warning("Успешный ответ не был сохранен в базу данных")
            
    except Exception as e:
        logging.error(f"Ошибка при сохранении успешного ответа: {e}")


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
                system_prompt="Ты — медицинский эксперт по анализам. Извлеки из текста все результаты анализов в структурированном формате.",
                model_type="text"
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
    logging.debug("Создание клавиатуры обратной связи")
    
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
    
    logging.debug("Клавиатура обратной связи создана")
    return builder.as_markup()


# Функция для создания клавиатуры уточнения
def get_clarification_keyboard():
    logging.debug("Создание клавиатуры уточнения")
    
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
    
    logging.debug("Клавиатура уточнения создана")
    return builder.as_markup()


# Функция для создания главной клавиатуры
def get_main_keyboard():
    logging.debug("Создание главной клавиатуры")
    
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
    
    logging.debug("Главная клавиатура создана")
    return builder.as_markup()


# Функция для получения эмбеддинга
def get_embedding(text: str) -> List[float]:
    """Получение эмбеддинга текста с помощью Mistral AI"""
    try:
        logging.info(f"Получение эмбеддинга для текста длиной {len(text)} символов")
        
        headers = {
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-embed",
            "input": text
        }
        
        logging.info("Отправляю запрос к Mistral AI API")
        response = requests.post("https://api.mistral.ai/v1/embeddings", headers=headers, json=data)
        response.raise_for_status()
        
        embedding = response.json()["data"][0]["embedding"]
        logging.info(f"Получен эмбеддинг длиной {len(embedding)} векторов")
        
        return embedding
    except Exception as e:
        logging.error(f"Ошибка при получении эмбеддинга: {e}")
        return []


# Функция для косинусного сходства
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Вычисление косинусного сходства между двумя векторами"""
    try:
        logging.debug(f"Вычисление косинусного сходства для векторов длиной {len(a)} и {len(b)}")
        
        a = np.array(a)
        b = np.array(b)
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        logging.debug(f"Косинусное сходство: {similarity:.6f}")
        return similarity
    except Exception as e:
        logging.error(f"Ошибка при вычислении косинусного сходства: {e}")
        return 0.0


# Функция для векторного поиска
def vector_search(query: str, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """Поиск похожих вопросов в векторной базе знаний"""
    try:
        logging.info(f"Векторный поиск для запроса: {query} с порогом: {threshold}")
        
        query_embedding = get_embedding(query)
        if not query_embedding:
            logging.warning("Не удалось получить эмбеддинг для запроса")
            return []

        # Получаем все записи с эмбеддингами
        response = supabase.table("doc_knowledge_base_vector").select("*").execute()
        logging.info(f"Найдено {len(response.data)} записей в векторной базе")
        
        results = []
        for item in response.data:
            if item.get("embedding"):
                # Конвертируем строку JSON обратно в список
                try:
                    item_embedding = json.loads(item["embedding"])
                    similarity = cosine_similarity(query_embedding, item_embedding)
                    if similarity >= threshold:
                        results.append((item["question"], item["answer"], similarity))
                        logging.info(f"Найдена релевантная запись с схожестью: {similarity:.3f}")
                except (json.JSONDecodeError, TypeError):
                    logging.warning(f"Ошибка при обработке эмбеддинга записи: {item.get('question', 'N/A')}")
                    continue

        # Сортируем по схожести
        results.sort(key=lambda x: x[2], reverse=True)
        logging.info(f"Всего найдено {len(results)} релевантных записей")
        return results[:3]  # Возвращаем топ-3 результата
    except Exception as e:
        logging.error(f"Ошибка при векторном поиске: {e}")
        return []


# Функция для сохранения в векторную базу знаний
def save_to_vector_knowledge_base(question: str, answer: str, source: str = ""):
    """Сохранение вопроса и ответа с эмбеддингом"""
    try:
        logging.info(f"Сохранение в векторную базу знаний: вопрос длиной {len(question)} символов")
        
        embedding = get_embedding(question)
        if embedding:
            # Конвертируем эмбеддинг в строку JSON для сохранения
            embedding_json = json.dumps(embedding)
            logging.info(f"Эмбеддинг получен, длина JSON: {len(embedding_json)} символов")
            
            response = supabase.table("doc_knowledge_base_vector").insert({
                "question": question,
                "answer": answer,
                "source": source,
                "embedding": embedding_json,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            if response.data:
                logging.info("Данные успешно сохранены в векторную базу знаний")
            else:
                logging.warning("Данные не были сохранены в векторную базу знаний")
        else:
            logging.warning("Не удалось получить эмбеддинг для вопроса")
            
    except Exception as e:
        logging.error(f"Ошибка при сохранении в векторную базу знаний: {e}")


# Функция для поиска в медицинских источниках
async def search_medical_sources(query: str) -> str:
    try:
        search_query = f"{query} медицина здоровье"
        logging.info(f"Поиск в медицинских источниках: {search_query}")
        
        response = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=3
        )
        
        logging.info(f"Получено {len(response.get('results', []))} результатов от Tavily")
        
        results = []
        for result in response["results"]:
            if any(source in result["url"] for source in MEDICAL_SOURCES):
                results.append(f"Источник: {result['url']}\n{result['content']}")
                logging.info(f"Добавлен результат из авторитетного источника: {result['url']}")
        
        if results:
            logging.info(f"Найдено {len(results)} результатов из авторитетных медицинских источников")
        else:
            logging.info("Результаты из авторитетных медицинских источников не найдены")
            
        return "\n\n".join(results) if results else ""
    except Exception as e:
        logging.error(f"Ошибка при поиске в медицинских источниках: {e}")
        return ""


# Функция для извлечения текста из PDF
async def extract_text_from_pdf(file_path: str) -> str:
    try:
        logging.info(f"Извлечение текста из PDF: {file_path}")
        
        async with aiohttp.ClientSession() as session:
            logging.info("Отправляю запрос к PDF файлу")
            async with session.get(file_path) as response:
                if response.status == 200:
                    logging.info("PDF файл успешно загружен")
                    pdf_data = await response.read()
                    logging.info(f"Размер PDF данных: {len(pdf_data)} байт")
                    
                    pdf_file = io.BytesIO(pdf_data)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    logging.info(f"PDF содержит {len(pdf_reader.pages)} страниц")
                    
                    text = ""
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        text += page_text + "\n"
                        logging.info(f"Страница {i+1}: {len(page_text)} символов")
                    
                    logging.info(f"Общий объем извлеченного текста: {len(text)} символов")
                    return text
                else:
                    logging.error(f"Ошибка при загрузке PDF: HTTP {response.status}")
                    return ""
                    
    except Exception as e:
        logging.error(f"Ошибка при извлечении текста из PDF: {e}")
        return ""


# Функция для извлечения данных пациента из текста
async def extract_patient_data_from_text(text: str) -> Dict[str, Any]:
    try:
        logging.info(f"Извлечение данных пациента из текста длиной {len(text)} символов")
        
        messages = [
            {
                "role": "system",
                "content": f"""Ты — помощник, который извлекает данные пациента из медицинских документов. 
                
                ТЕКУЩАЯ ДАТА: {datetime.now().strftime('%d.%m.%Y')} (год: {datetime.now().year})
                
                Извлеки имя, возраст, пол и дату рождения, если они есть. 
                
                ВАЖНО: 
                - При извлечении возраста учитывай текущую дату. Если в документе указан возраст 
                "33 года", а сейчас {datetime.now().year} год, то возраст пациента сейчас больше 33 лет.
                - Дату рождения ищи в форматах: ДД.ММ.ГГГГ, ДД/ММ/ГГГГ, ДД-ММ-ГГГГ, или текстом "родился 15.03.1990"
                - Если указан только год рождения, используй его для вычисления возраста
                
                Верни ответ в формате JSON: 
                {{"name": "имя", "age": число, "gender": "М" или "Ж", "birth_date": "ГГГГ-ММ-ДД"}}. 
                Если каких-то данных нет, поставь null.
                
                Примеры дат: "1990-03-15", "1985-12-01" """
            },
            {
                "role": "user",
                "content": text[:2000]
            }
        ]
        
        logging.info("Отправляю запрос к ИИ для извлечения данных пациента")
        response_text, _, _ = await call_model_with_failover(
            messages=messages,
            system_prompt="Ты — помощник, который извлекает данные пациента из медицинских документов.",
            model_type="text"
        )
        
        logging.info(f"Получен ответ от ИИ: {len(response_text)} символов")
        
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logging.info(f"Найден JSON в ответе: {json_str}")
                data = json.loads(json_str)
                
                # Обрабатываем дату рождения
                birth_date = data.get("birth_date")
                if birth_date:
                    logging.info(f"Извлечена дата рождения: {birth_date}")
                    # Пытаемся распарсить дату в различных форматах
                    parsed_date = parse_birth_date(birth_date)
                    if parsed_date:
                        birth_date = parsed_date
                        logging.info(f"Дата рождения распарсена: {birth_date}")
                    else:
                        birth_date = None
                        logging.warning("Не удалось распарсить дату рождения")
                
                # Вычисляем текущий возраст на основе извлеченного возраста или даты рождения
                extracted_age = data.get("age")
                current_age = None
                
                if birth_date:
                    # Если есть дата рождения, вычисляем точный возраст
                    current_age = calculate_age_from_birth_date(birth_date)
                    logging.info(f"Возраст вычислен по дате рождения: {current_age}")
                elif extracted_age and isinstance(extracted_age, int):
                    # Если есть только возраст, вычисляем примерный
                    current_age = calculate_current_age(extracted_age)
                    logging.info(f"Возраст вычислен по извлеченному возрасту: {current_age}")
                
                result = {
                    "name": data.get("name"),
                    "age": current_age,
                    "gender": data.get("gender"),
                    "birth_date": birth_date
                }
                
                logging.info(f"Данные пациента извлечены через ИИ: {result}")
                return result
                
        except json.JSONDecodeError as e:
            logging.warning(f"Ошибка парсинга JSON: {e}")
            pass

        # Если не удалось извлечь JSON, пробуем простой парсинг
        logging.info("Использую простой парсинг для извлечения данных")
        name_match = re.search(r'(?:Пациент|ФИО|Имя):\s*([А-Яа-я\s]+)', text)
        age_match = re.search(r'(?:Возраст|Лет):\s*(\d+)', text)
        gender_match = re.search(r'(?:Пол):\s*([МЖ])', text)
        
        # Ищем дату рождения в различных форматах
        birth_date = extract_birth_date_from_text(text)

        extracted_age = int(age_match.group(1)) if age_match else None
        current_age = None
        
        if birth_date:
            current_age = calculate_age_from_birth_date(birth_date)
            logging.info(f"Возраст вычислен по дате рождения (простой парсинг): {current_age}")
        elif extracted_age:
            current_age = calculate_current_age(extracted_age)
            logging.info(f"Возраст вычислен по извлеченному возрасту (простой парсинг): {current_age}")

        result = {
            "name": name_match.group(1).strip() if name_match else None,
            "age": current_age,
            "gender": gender_match.group(1) if gender_match else None,
            "birth_date": birth_date
        }
        
        logging.info(f"Данные пациента извлечены простым парсингом: {result}")
        return result
        
    except Exception as e:
        logging.error(f"Ошибка при извлечении данных пациента: {e}")
        return {}


# Функция для вычисления текущего возраста на основе указанного возраста
def calculate_current_age(extracted_age: int) -> int:
    """Вычисляет текущий возраст на основе указанного в документе"""
    try:
        current_year = datetime.now().year
        # Предполагаем, что возраст указан на момент создания документа
        # Для простоты считаем, что документ создан в текущем году
        current_age = extracted_age
        logging.info(f"Вычисление текущего возраста: извлеченный возраст {extracted_age}, текущий год {current_year}, результат {current_age}")
        return current_age
    except Exception as e:
        logging.error(f"Ошибка при вычислении текущего возраста: {e}")
        return extracted_age


def parse_birth_date(date_str: str) -> Optional[str]:
    """
    Парсит дату рождения в различных форматах и возвращает в формате YYYY-MM-DD
    """
    try:
        logging.info(f"Парсинг даты рождения: {date_str}")
        
        if not date_str:
            logging.info("Строка даты пуста")
            return None
            
        # Убираем лишние пробелы
        date_str = date_str.strip()
        logging.info(f"Очищенная строка даты: {date_str}")
        
        # Пытаемся распарсить различные форматы
        try:
            # Пробуем стандартный парсер
            logging.info("Пробую стандартный парсер dateutil")
            parsed_date = parse(date_str, dayfirst=True, yearfirst=False)
            result = parsed_date.strftime('%Y-%m-%d')
            logging.info(f"Дата успешно распарсена стандартным парсером: {result}")
            return result
        except Exception as e:
            logging.info(f"Стандартный парсер не сработал: {e}")
            pass
        
        # Пытаемся распарсить вручную различные форматы
        logging.info("Пробую ручной парсинг с регулярными выражениями")
        patterns = [
            r'(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-](\d{4})',  # ДД.ММ.ГГГГ
            r'(\d{4})[\.\/\-](\d{1,2})[\.\/\-](\d{1,2})',  # ГГГГ.ММ.ДД
            r'(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-](\d{2})',  # ДД.ММ.ГГ
            r'(\d{4})',  # Только год
        ]
        
        for i, pattern in enumerate(patterns):
            logging.info(f"Пробую паттерн {i+1}: {pattern}")
            match = re.search(pattern, date_str)
            if match:
                logging.info(f"Найдено совпадение с паттерном {i+1}: {match.groups()}")
                if len(match.groups()) == 3:
                    if len(match.group(3)) == 4:  # ДД.ММ.ГГГГ
                        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                        logging.info(f"Интерпретирую как ДД.ММ.ГГГГ: день={day}, месяц={month}, год={year}")
                    else:  # ГГГГ.ММ.ДД
                        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                        logging.info(f"Интерпретирую как ГГГГ.ММ.ДД: год={year}, месяц={month}, день={day}")
                    
                    # Проверяем валидность даты
                    if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= datetime.now().year:
                        result = f"{year:04d}-{month:02d}-{day:02d}"
                        logging.info(f"Дата валидна: {result}")
                        return result
                    else:
                        logging.warning(f"Дата невалидна: день={day}, месяц={month}, год={year}")
                elif len(match.groups()) == 1:  # Только год
                    year = int(match.group(1))
                    logging.info(f"Найден только год: {year}")
                    if 1900 <= year <= datetime.now().year:
                        result = f"{year:04d}-01-01"  # Используем 1 января как примерную дату
                        logging.info(f"Год валиден, используем примерную дату: {result}")
                        return result
                    else:
                        logging.warning(f"Год невалиден: {year}")
        
        logging.info("Не удалось распарсить дату ни одним из способов")
        return None
        
    except Exception as e:
        logging.error(f"Ошибка при парсинге даты рождения: {e}")
        return None


def extract_birth_date_from_text(text: str) -> Optional[str]:
    """
    Извлекает дату рождения из текста с помощью регулярных выражений
    """
    try:
        logging.info(f"Извлечение даты рождения из текста длиной {len(text)} символов")
        
        # Ищем различные форматы дат
        patterns = [
            r'(?:родился|дата рождения|д\.р\.|Д\.Р\.):\s*(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{4})',
            r'(?:родился|дата рождения|д\.р\.|Д\.Р\.):\s*(\d{4})',
            r'(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{4})',
            r'(\d{4})'
        ]
        
        for i, pattern in enumerate(patterns):
            logging.info(f"Пробую паттерн {i+1}: {pattern}")
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                logging.info(f"Найдено совпадение с паттерном {i+1}: {date_str}")
                
                parsed_date = parse_birth_date(date_str)
                if parsed_date:
                    logging.info(f"Дата успешно извлечена: {parsed_date}")
                    return parsed_date
                else:
                    logging.warning(f"Не удалось распарсить найденную дату: {date_str}")
        
        logging.info("Дата рождения не найдена в тексте")
        return None
        
    except Exception as e:
        logging.error(f"Ошибка при извлечении даты рождения из текста: {e}")
        return None


def calculate_age_from_birth_date(birth_date: str) -> Optional[int]:
    """
    Вычисляет точный возраст на основе даты рождения
    """
    try:
        logging.info(f"Вычисление возраста из даты рождения: {birth_date}")
        
        if not birth_date:
            logging.info("Дата рождения не указана")
            return None
            
        # Парсим дату рождения
        birth_dt = datetime.strptime(birth_date, '%Y-%m-%d')
        current_dt = datetime.now()
        
        logging.info(f"Дата рождения: {birth_dt}, текущая дата: {current_dt}")
        
        # Вычисляем возраст
        age = current_dt.year - birth_dt.year
        
        # Корректируем, если день рождения еще не наступил в этом году
        if (current_dt.month, current_dt.day) < (birth_dt.month, birth_dt.day):
            age -= 1
            logging.info("День рождения еще не наступил, возраст уменьшен на 1")
        
        # Проверяем разумность результата
        if age < 0 or age > 120:
            logging.warning(f"Вычисленный возраст {age} выходит за разумные пределы")
            return None
        
        logging.info(f"Вычисленный возраст: {age}")
        return age
        
    except Exception as e:
        logging.error(f"Ошибка при вычислении возраста из даты рождения: {e}")
        return None


# Функция для анализа изображения
async def analyze_image(image_url: str, query: str = "Что показано на этом медицинском изображении?") -> str:
    try:
        logging.info(f"Анализ изображения: {image_url}")
        logging.info(f"Запрос для анализа: {query}")
        
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
        
        logging.info("Созданы сообщения для анализа изображения")
        
        # Для анализа изображений используем модели с поддержкой изображений
        logging.info("Вызываю модель для анализа изображения")
        response, _, _ = await call_model_with_failover(
            messages=messages,
            model_type="vision",  # Указываем тип модели для анализа изображений
            system_prompt="Ты — медицинский эксперт по анализу изображений."
        )
        
        logging.info(f"Анализ изображения завершен, результат: {len(response)} символов")
        return response
        
    except Exception as e:
        logging.error(f"Ошибка при анализе изображения: {e}")
        return "Не удалось проанализировать изображение. Попробуйте еще раз."


# Функция для поиска в базе знаний
def search_knowledge_base(query: str) -> str:
    try:
        logging.info(f"Поиск в базе знаний для запроса: {query}")
        
        vector_results = vector_search(query)
        if vector_results:
            logging.info(f"Найдено {len(vector_results)} результатов в векторной базе")
            return "\n\n".join([f"Вопрос: {q}\nОтвет: {a}" for q, a, _ in vector_results])

        logging.info("Ищу в обычной базе знаний")
        response = supabase.table("doc_knowledge_base").select("*").execute()
        results = [item["answer"] for item in response.data if query.lower() in item["question"].lower()]
        
        if results:
            logging.info(f"Найдено {len(results)} результатов в обычной базе знаний")
        else:
            logging.info("В базе знаний ничего не найдено")
            
        return "\n".join(results) if results else ""
    except Exception as e:
        logging.error(f"Ошибка при поиске в базе знаний: {e}")
        return ""


# Функция для создания профиля пациента
def create_patient_profile(user_id: str, name: str, age: int, gender: str, telegram_id: int = None, birth_date: str = None) -> bool:
    try:
        logging.info(f"Создание профиля пациента для пользователя {user_id}: {name}, возраст {age}, пол {gender}")
        
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
            logging.info(f"Добавлен Telegram ID: {telegram_id}")
            
        # Добавляем дату рождения если она передана
        if birth_date:
            profile_data["birth_date"] = birth_date
            logging.info(f"Добавлена дата рождения: {birth_date}")
            
        response = supabase.table("doc_patient_profiles").insert(profile_data).execute()
        
        success = len(response.data) > 0
        if success:
            logging.info("Профиль пациента успешно создан")
        else:
            logging.warning("Профиль пациента не был создан")
            
        return success
        
    except Exception as e:
        logging.error(f"Ошибка при создании профиля пациента: {e}")
        return False


def update_patient_profile(user_id: str, **updates) -> bool:
    """
    Обновляет профиль пациента новыми данными.
    Поддерживает обновление: name, age, gender, birth_date, phone, email, address, medical_history, allergies
    """
    try:
        logging.info(f"Обновление профиля пациента {user_id} с данными: {updates}")
        
        # Подготавливаем данные для обновления
        update_data = {}
        
        # Добавляем только те поля, которые переданы и не None
        for key, value in updates.items():
            if value is not None:
                update_data[key] = value
        
        # Добавляем время обновления
        update_data["updated_at"] = datetime.now().isoformat()
        
        if not update_data:
            logging.info("Нечего обновлять в профиле пациента")
            return True  # Нечего обновлять
            
        logging.info(f"Данные для обновления: {update_data}")
        
        response = supabase.table("doc_patient_profiles").update(update_data).eq("user_id", user_id).execute()
        
        success = len(response.data) > 0
        if success:
            logging.info("Профиль пациента успешно обновлен")
        else:
            logging.warning("Профиль пациента не был обновлен")
            
        return success
        
    except Exception as e:
        logging.error(f"Ошибка при обновлении профиля пациента: {e}")
        return False


def merge_patient_data(existing_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Объединяет существующие данные пациента с новыми данными.
    Новые данные имеют приоритет, но существующие данные сохраняются, если новых нет.
    """
    logging.info(f"Объединение данных пациента: существующие ключи: {list(existing_data.keys())}, новые ключи: {list(new_data.keys())}")
    
    merged = existing_data.copy()
    
    for key, value in new_data.items():
        if value is not None:
            # Для даты рождения проверяем, что новая дата более точная
            if key == "birth_date" and merged.get("birth_date"):
                existing_date = merged["birth_date"]
                if isinstance(existing_date, str) and len(existing_date) == 10:  # YYYY-MM-DD
                    if isinstance(value, str) and len(value) == 10:  # YYYY-MM-DD
                        # Новая дата более точная, обновляем
                        logging.info(f"Обновляю дату рождения с {existing_date} на {value}")
                        merged[key] = value
                elif isinstance(existing_date, str) and len(existing_date) == 4:  # YYYY
                    if isinstance(value, str) and len(value) == 10:  # YYYY-MM-DD
                        # Новая дата более точная, обновляем
                        logging.info(f"Обновляю дату рождения с {existing_date} на {value}")
                        merged[key] = value
            else:
                if key in merged and merged[key] != value:
                    logging.info(f"Обновляю поле {key} с {merged[key]} на {value}")
                merged[key] = value
    
    logging.info(f"Результат объединения: {list(merged.keys())}")
    return merged


# Функция для получения профиля пациента
def get_patient_profile(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        logging.info(f"Получение профиля пациента для пользователя: {user_id}")
        
        response = supabase.table("doc_patient_profiles").select("*").eq("user_id", user_id).execute()
        
        if response.data:
            profile = response.data[0]
            logging.info(f"Профиль найден: {profile.get('name', 'N/A')}, возраст: {profile.get('age', 'N/A')}")
            return profile
        else:
            logging.info("Профиль пациента не найден")
            return None
            
    except Exception as e:
        logging.error(f"Ошибка при получении профиля пациента: {e}")
        return None


# Функция для интеллектуальной проверки дублирования медицинских записей с помощью ИИ
async def check_duplicate_medical_record_ai(user_id: str, content: str, record_type: str = "image_analysis") -> bool:
    """
    Интеллектуально проверяет, есть ли уже запись с аналогичными данными у пользователя.
    Использует ИИ для анализа сути данных, а не точного текстового совпадения.
    Возвращает True, если дубликат найден.
    """
    try:
        logging.info(f"ИИ-проверка дублирования для пользователя: {user_id}")
        
        # Получаем последние записи пользователя
        query = supabase.table("doc_medical_records").select("*").eq("user_id", user_id).eq("record_type", record_type)
        response = query.order("created_at", desc=True).limit(10).execute()
        
        if not response.data:
            logging.info("Записей для сравнения не найдено")
            return False
        
        # Используем ИИ для анализа дублирования
        for record in response.data:
            if await is_duplicate_by_ai(content, record.get("content", "")):
                logging.info(f"ИИ обнаружил дубликат записи с ID: {record.get('id')}")
                return True
        
        logging.info("ИИ не обнаружил дубликатов")
        return False
        
    except Exception as e:
        logging.error(f"Ошибка при ИИ-проверке дублирования: {e}")
        # В случае ошибки ИИ, возвращаем False чтобы не блокировать сохранение
        return False

async def is_duplicate_by_ai(new_content: str, existing_content: str) -> bool:
    """
    Использует ИИ для определения, являются ли два содержимых дубликатами.
    Анализирует суть данных, а не точное текстовое совпадение.
    """
    try:
        # Формируем промпт для ИИ
        prompt = f"""
        Проанализируй два медицинских анализа и определи, являются ли они дубликатами (повтором одного и того же анализа).

        АНАЛИЗ 1 (новый):
        {new_content[:2000]}

        АНАЛИЗ 2 (существующий):
        {existing_content[:2000]}

        Критерии для определения дубликата:
        1. Одинаковые типы анализов (например, anti-HEV IgG, anti-HCV, IgE и т.д.)
        2. Одинаковые результаты (положительные/отрицательные, числовые значения)
        3. Одинавый пациент (имя, дата рождения)
        4. Анализы сданы в один день или очень близко по времени

        Ответь только "ДА" если это дубликат, или "НЕТ" если это разные анализы.
        """

        # Используем доступную модель для анализа
        analysis_result = await call_model_with_failover(
            messages=[{"role": "user", "content": prompt}],
            model_type="text"
        )
        
        if analysis_result:
            is_duplicate = "ДА" in analysis_result.upper()
            logging.info(f"ИИ определил дубликат: {is_duplicate} (ответ: {analysis_result})")
            return is_duplicate
        
        return False
        
    except Exception as e:
        logging.error(f"Ошибка при ИИ-анализе дублирования: {e}")
        return False

# Функция для проверки дублирования медицинских записей
def check_duplicate_medical_record(user_id: str, content: str, record_type: str = "image_analysis") -> bool:
    """
    Проверяет, есть ли уже запись с таким же содержимым у пользователя.
    Возвращает True, если дубликат найден.
    """
    try:
        logging.info(f"Проверка дублирования для пользователя: {user_id}")
        
        # Получаем последние записи пользователя
        query = supabase.table("doc_medical_records").select("*").eq("user_id", user_id).eq("record_type", record_type)
        response = query.order("created_at", desc=True).limit(10).execute()
        
        if not response.data:
            logging.info("Записей для сравнения не найдено")
            return False
        
        # Проверяем на дублирование по содержимому
        for record in response.data:
            if record.get("content") == content:
                logging.info(f"Найден дубликат записи с ID: {record.get('id')}")
                return True
        
        logging.info("Дубликаты не найдены")
        return False
        
    except Exception as e:
        logging.error(f"Ошибка при проверке дублирования: {e}")
        return False


# Функция для сохранения медицинских записей
async def save_medical_record(user_id: str, record_type: str, content: str, source: str = "") -> bool:
    try:
        logging.info(f"Сохранение медицинской записи для пользователя: {user_id}")
        logging.info(f"Тип записи: {record_type}, источник: {source}")
        logging.info(f"Длина содержимого: {len(content)} символов")
        
        # Проверяем на дублирование с помощью улучшенной ИИ-проверки перед сохранением
        if await check_duplicate_medical_record_ai_enhanced(user_id, content, record_type):
            logging.info("Улучшенная ИИ-проверка обнаружила дубликат записи, пропускаем сохранение")
            return True  # Возвращаем True, так как запись уже существует
        
        response = supabase.table("doc_medical_records").insert({
            "user_id": user_id,
            "record_type": record_type,
            "content": content,
            "source": source,
            "created_at": datetime.now().isoformat()
        }).execute()
        
        success = len(response.data) > 0
        logging.info(f"Медицинская запись сохранена: {success}")
        
        return success
    except Exception as e:
        logging.error(f"Ошибка при сохранении медицинской записи: {e}")
        return False


# Функция для получения медицинских записей
def get_medical_records(user_id: str, record_type: str = None) -> List[Dict[str, Any]]:
    try:
        logging.info(f"Получение медицинских записей для пользователя: {user_id}")
        if record_type:
            logging.info(f"Фильтр по типу записи: {record_type}")
            
        query = supabase.table("doc_medical_records").select("*").eq("user_id", user_id)
        if record_type:
            query = query.eq("record_type", record_type)
        response = query.order("created_at", desc=True).execute()
        
        records = response.data if response.data else []
        logging.info(f"Найдено {len(records)} медицинских записей")
        
        # Логируем типы найденных записей
        if records:
            record_types = [record.get('record_type', 'unknown') for record in records]
            logging.info(f"Типы записей: {record_types}")
        
        return records
    except Exception as e:
        logging.error(f"Ошибка при получении медицинских записей: {e}")
        return []


# Функция для сохранения в базу знаний
def save_to_knowledge_base(question: str, answer: str, source: str = ""):
    try:
        logging.info(f"Сохранение в базу знаний: вопрос длиной {len(question)} символов, ответ длиной {len(answer)} символов")
        
        response = supabase.table("doc_knowledge_base").insert({
            "question": question,
            "answer": answer,
            "source": source,
            "created_at": datetime.now().isoformat()
        }).execute()
        
        if response.data:
            logging.info("Данные успешно сохранены в базу знаний")
        else:
            logging.warning("Данные не были сохранены в базу знаний")
            
        # Также сохраняем в векторную базу знаний
        save_to_vector_knowledge_base(question, answer, source)
        
    except Exception as e:
        logging.error(f"Ошибка при сохранении в базу знаний: {e}")


# Функция для сохранения обратной связи
def save_user_feedback(user_id: str, question: str, helped: bool):
    try:
        logging.info(f"Сохранение обратной связи от пользователя {user_id}: помогло ли: {helped}")
        
        response = supabase.table("doc_user_feedback").insert({
            "user_id": user_id,
            "question": question,
            "helped": helped,
            "created_at": datetime.now().isoformat()
        }).execute()
        
        if response.data:
            logging.info("Обратная связь успешно сохранена")
        else:
            logging.warning("Обратная связь не была сохранена")
            
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
        logging.info(f"Поиск в интернете для запроса: {query}")
        
        response = tavily_client.search(query, max_results=3)
        logging.info(f"Получено {len(response.get('results', []))} результатов от Tavily")
        
        results = []
        for result in response["results"]:
            results.append(f"{result['content']}\nИсточник: {result['url']}")
            logging.info(f"Добавлен результат: {result['url']}")
        
        if results:
            logging.info(f"Создан контекст из {len(results)} результатов")
        else:
            logging.info("Результаты поиска не найдены")
            
        return "\n".join(results)
    except Exception as e:
        logging.error(f"Ошибка при поиске в интернете: {e}")
        return ""


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
            f"• /cleanup_duplicates - очистка дублирующихся записей\n\n"
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
            f"• /cleanup_duplicates - очистка дублирующихся записей\n\n"
            f"🔍 Что вас интересует?",
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


# Обработчик команды /cleanup_duplicates
@dp.message(Command("cleanup_duplicates"))
async def cleanup_duplicates_command(message: types.Message):
    try:
        user_id = generate_user_uuid(message.from_user.id)
        deleted_count = cleanup_duplicate_medical_records(user_id)
        
        if deleted_count > 0:
            await message.answer(f"🧹 Очистка завершена! Удалено {deleted_count} дублирующихся записей.")
        else:
            await message.answer("✅ Дублирующихся записей не найдено.")
            
    except Exception as e:
        logging.error(f"Ошибка при очистке дубликатов: {e}")
        await message.answer("😔 Не удалось очистить дубликаты")


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
            await save_medical_record(
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
                            patient_data.get("name") or patient_data.get("age") or patient_data.get("gender") or patient_data.get("birth_date")):
                        extracted_info = "📝 Я обнаружил(а) в вашем анализе следующие данные:\n\n"
                        if patient_data.get("name"):
                            extracted_info += f"👤 Имя: {patient_data['name']}\n"
                        if patient_data.get("age"):
                            extracted_info += f"🎂 Возраст: {patient_data['age']}\n"
                        if patient_data.get("gender"):
                            extracted_info += f"⚧️ Пол: {patient_data['gender']}\n"
                        if patient_data.get("birth_date"):
                            extracted_info += f"📅 Дата рождения: {patient_data['birth_date']}\n"
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
                else:
                    # Если профиль уже есть, проверяем, есть ли новые данные для обновления
                    patient_data = await extract_patient_data_from_text(pdf_text)
                    updates_needed = {}
                    update_info = ""
                    
                    # Проверяем, какие данные можно обновить
                    if patient_data.get("name") and patient_data["name"] != profile.get("name"):
                        updates_needed["name"] = patient_data["name"]
                        update_info += f"👤 Имя: {profile.get('name')} → {patient_data['name']}\n"
                        
                    if patient_data.get("birth_date") and patient_data["birth_date"] != profile.get("birth_date"):
                        updates_needed["birth_date"] = patient_data["birth_date"]
                        update_info += f"📅 Дата рождения: {profile.get('birth_date', 'не указана')} → {patient_data['birth_date']}\n"
                        
                    if patient_data.get("age") and patient_data["age"] != profile.get("age"):
                        updates_needed["age"] = patient_data["age"]
                        update_info += f"🎂 Возраст: {profile.get('age', 'не указан')} → {patient_data['age']}\n"
                        
                    if patient_data.get("gender") and patient_data["gender"] != profile.get("gender"):
                        updates_needed["gender"] = patient_data["gender"]
                        update_info += f"⚧️ Пол: {profile.get('gender', 'не указан')} → {patient_data['gender']}\n"
                    
                    # Если есть обновления, предлагаем их применить
                    if updates_needed:
                        update_message = f"🔄 Обнаружены новые данные пациента:\n\n{update_info}\n\nОбновить профиль?"
                        
                        await message.answer(
                            update_message,
                            reply_markup=InlineKeyboardBuilder().add(
                                types.InlineKeyboardButton(
                                    text="✅ Да, обновить",
                                    callback_data="update_profile_data"
                                ),
                                types.InlineKeyboardButton(
                                    text="❌ Нет, оставить как есть",
                                    callback_data="keep_existing_data"
                                )
                            ).as_markup()
                        )
                        
                        # Сохраняем данные для обновления в состоянии
                        await state.set_state(DoctorStates.updating_profile)
                        await state.update_data(
                            profile_updates=updates_needed,
                            pdf_text=pdf_text
                        )
                        return
                    
                    # Если обновлений нет, предлагаем проанализировать результаты
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
    logging.info(f"Получено изображение от пользователя {message.from_user.id}")
    
    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    file_url = f"https://api.telegram.org/file/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/{file_path}"
    
    logging.info(f"URL изображения: {file_url}")
    processing_msg = await message.answer("🔍 Анализирую изображение...")
    
    # Анализируем изображение для извлечения текста
    logging.info("Начинаю анализ изображения")
    analysis_result = await analyze_image(file_url, "Извлеки все медицинские данные и информацию о пациенте с этого изображения. Верни текст с анализами и данными пациента.")
    await processing_msg.edit_text("✅ Изображение успешно проанализировано.")
    
    logging.info(f"Результат анализа изображения: {len(analysis_result)} символов")
    
    # Пытаемся извлечь данные пациента из анализа
    logging.info("Извлечение данных пациента из результата анализа")
    patient_data = await extract_patient_data_from_text(analysis_result)
    
    if patient_data:
        logging.info(f"Извлечены данные пациента: {patient_data}")
    else:
        logging.info("Данные пациента не извлечены")
    
    # Проверяем, есть ли профиль у пользователя
    profile = get_patient_profile(generate_user_uuid(message.from_user.id))
    
    if not profile:
        logging.info("Профиль пациента не найден")
        # Если профиля нет, предлагаем создать его на основе извлеченных данных
        if patient_data and (patient_data.get("name") or patient_data.get("age") or patient_data.get("gender") or patient_data.get("birth_date")):
            logging.info("Предлагаю создать профиль на основе извлеченных данных")
            extracted_info = "📝 Я обнаружил(а) в вашем анализе следующие данные:\n\n"
            if patient_data.get("name"):
                extracted_info += f"👤 Имя: {patient_data['name']}\n"
            if patient_data.get("age"):
                extracted_info += f"🎂 Возраст: {patient_data['age']}\n"
            if patient_data.get("gender"):
                extracted_info += f"⚧️ Пол: {patient_data['gender']}\n"
            if patient_data.get("birth_date"):
                extracted_info += f"📅 Дата рождения: {patient_data['birth_date']}\n"
            
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
            logging.info("Состояние установлено на подтверждение профиля")
        else:
            logging.info("Создаю анонимный профиль")
            # Если не удалось извлечь данные, создаем анонимный профиль
            create_patient_profile(generate_user_uuid(message.from_user.id), "аноним", None, None, message.from_user.id)
            await message.answer(
                "✅ Создан анонимный профиль.\n\n"
                f"📊 <b>Результат анализа:</b>\n\n{escape_html(analysis_result)}\n\n"
                f"⚠️ Помните, что это автоматический анализ, и он не заменяет консультацию специалиста.",
                parse_mode="HTML"
            )
            
            # Сохраняем в медицинские записи ПОСЛЕ создания профиля
            await save_medical_record(
                user_id=generate_user_uuid(message.from_user.id),
                record_type="image_analysis",
                content=analysis_result,
                source="Изображение из Telegram"
            )
            logging.info("Медицинская запись сохранена для анонимного профиля")
    else:
        logging.info(f"Профиль пациента найден: {profile.get('name', 'N/A')}")
        # Если профиль уже есть, проверяем, есть ли новые данные для обновления
        updates_needed = {}
        update_info = ""
        
        # Проверяем, какие данные можно обновить
        if patient_data.get("name") and patient_data["name"] != profile.get("name"):
            updates_needed["name"] = patient_data["name"]
            update_info += f"👤 Имя: {profile.get('name')} → {patient_data['name']}\n"
            
        if patient_data.get("birth_date") and patient_data["birth_date"] != profile.get("birth_date"):
            updates_needed["birth_date"] = patient_data["birth_date"]
            update_info += f"📅 Дата рождения: {profile.get('birth_date', 'не указана')} → {patient_data['birth_date']}\n"
            
        if patient_data.get("age") and patient_data["age"] != profile.get("age"):
            updates_needed["age"] = patient_data["age"]
            update_info += f"🎂 Возраст: {profile.get('age', 'не указан')} → {patient_data['age']}\n"
            
        if patient_data.get("gender") and patient_data["gender"] != profile.get("gender"):
            updates_needed["gender"] = patient_data["gender"]
            update_info += f"⚧️ Пол: {profile.get('gender', 'не указан')} → {patient_data['gender']}\n"
        
        # Если есть обновления, предлагаем их применить
        if updates_needed:
            logging.info(f"Предлагаю обновить профиль: {updates_needed}")
            update_message = f"🔄 Обнаружены новые данные пациента:\n\n{update_info}\n\nОбновить профиль?"
            
            await message.answer(
                update_message,
                reply_markup=InlineKeyboardBuilder().add(
                    types.InlineKeyboardButton(
                        text="✅ Да, обновить",
                        callback_data="update_profile_data"
                    ),
                    types.InlineKeyboardButton(
                        text="❌ Нет, оставить как есть",
                        callback_data="keep_existing_data"
                    )
                ).as_markup()
            )
            
            # Сохраняем данные для обновления в состоянии
            await state.set_state(DoctorStates.updating_profile)
            await state.update_data(
                profile_updates=updates_needed,
                analysis_result=analysis_result
            )
        else:
            # Если обновлений нет, просто показываем результат анализа
            await message.answer(
                f"📊 <b>Результат анализа:</b>\n\n{escape_html(analysis_result)}\n\n"
                f"ℹ️ Новых данных пациента не обнаружено.\n"
                f"⚠️ Помните, что это автоматический анализ, и он не заменяет консультацию специалиста.",
                parse_mode="HTML"
            )
        
        # В любом случае сохраняем в медицинские записи
        await save_medical_record(
            user_id=generate_user_uuid(message.from_user.id),
            record_type="image_analysis",
            content=analysis_result,
            source="Изображение из Telegram"
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
    
    logging.info(f"Обработка сообщения от пользователя {user_id}")
    logging.info(f"История диалога: {len(history)} сообщений")
    logging.info(f"Профиль пациента: {'найден' if profile else 'не найден'}")
    
    # Добавляем текущее сообщение в историю
    history.append({"role": "user", "content": question})
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]
        await message.answer(
            "🔄 История диалога стала слишком длинной, я удалил самые старые сообщения для оптимизации.")
    
    # Проверяем, есть ли медицинские записи у пользователя
    has_medical_records = len(get_medical_records(generate_user_uuid(user_id))) > 0
    logging.info(f"Медицинские записи у пользователя: {'есть' if has_medical_records else 'нет'}")
    
    # Проверяем, нужно ли уточнять информацию
    clarification_count = data.get("clarification_count", 0)
    is_enough, clarification_question, ai_mode = await clarification_agent.analyze_and_ask(
        question, history, profile, clarification_count, has_medical_records
    )
    
    logging.info(f"Анализ уточнения: достаточно информации: {is_enough}, нужна ли уточнение: {clarification_question is not None}")
    
    # Если нужно уточнить информацию
    if not is_enough and clarification_question:
        logging.info("Отправляю уточняющий вопрос пользователю")
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
    
    # Интеллектуальный анализ типа запроса
    query_analysis = await intelligent_analyzer.analyze_query_type(question, generate_user_uuid(user_id))
    
    logging.info(f"Результат анализа запроса: {query_analysis}")
    
    # Улучшенная логика: если спрашивают о конкретных показателях, но нет медицинских данных
    if query_analysis["is_specific_indicator_question"] and not query_analysis["has_medical_data"]:
        logging.info("Пользователь спрашивает о конкретных показателях, но нет медицинских данных")
        await message.answer(
            "❌ У вас пока нет загруженных результатов анализов. "
            "Пожалуйста, сначала загрузите изображение или PDF файл с результатами анализов, "
            "а затем я смогу ответить на ваш вопрос о конкретных показателях."
        )
        return
    
    processing_msg = await message.answer("🔍 Анализирую ваш вопрос...")
    
    # Определяем режим работы ИИ на основе интеллектуального анализа
    if query_analysis["needs_doctor_mode"]:
        ai_mode = "doctor"
        mode_indicator = "👨‍⚕️ ИИ-врач главный"
        logging.info("Выбран режим ИИ-врач главный")
    else:
        mode_indicator = "👩‍⚕️ ИИ-ассистент врача"
        logging.info("Выбран режим ИИ-ассистент врача")
    
    # Извлекаем релевантный медицинский контекст
    medical_context = ""
    if query_analysis["has_medical_data"]:
        logging.info(f"Найдены медицинские записи для пользователя {user_id}: {len(query_analysis['medical_records'])} записей")
        
        medical_context = await intelligent_analyzer.get_relevant_medical_context(
            question, query_analysis["medical_records"]
        )
        
        logging.info(f"Релевантный медицинский контекст найден: {len(medical_context) > 0}")
        
        # Если не нашли релевантный контекст, но есть медицинские записи, 
        # показываем все доступные записи для анализа
        if not medical_context and query_analysis["medical_records"]:
            logging.info(f"Не найден релевантный контекст для вопроса: {question}")
            logging.info(f"Доступные медицинские записи: {len(query_analysis['medical_records'])}")
            
            # Создаем базовый контекст из всех медицинских записей
            all_records_context = "\n\n📊 ВСЕ ДОСТУПНЫЕ МЕДИЦИНСКИЕ ДАННЫЕ:\n"
            for i, record in enumerate(query_analysis["medical_records"][:3], 1):
                content = record.get('content', '')[:1000]  # Ограничиваем длину
                all_records_context += f"\n--- Запись {i} ---\n{content}\n"
            
            medical_context = all_records_context
            logging.info(f"Создан базовый контекст из всех медицинских записей: {len(medical_context)} символов")
    else:
        logging.info(f"Медицинские записи для пользователя {user_id} не найдены")
    
    # Формируем системный промпт в зависимости от режима
    base_system_prompt = f"""Ты — ИИ-ассистент врача. Твоя задача — помогать пользователям с медицинскими вопросами, 
    анализировать их анализы и предоставлять информацию о здоровье. Отвечай максимально точно и информативно, 
    используя предоставленный контекст. Учитывай историю диалога и данные пациента, если они доступны.
    
    ВАЖНО: Если в контексте есть результаты анализов пациента, всегда используй их для ответа на вопросы о показателях.
    Если пользователь спрашивает о конкретном анализе или показателе, найди его в результатах анализов и приведи точные данные.
    Не ищи эту информацию в других источниках, если она есть в предоставленных результатах анализов.
    
    ТЕКУЩАЯ ДАТА: {datetime.now().strftime('%d.%m.%Y')} (год: {datetime.now().year})
    
    ВАЖНО: Ты не ставишь диагноз и не заменяешь консультацию врача. Всегда рекомендуй консультацию 
    со специалистом для точной диагностики и лечения.
    Если в контексте есть точный ответ из авторитетных медицинских источников — используй его.
    Всегда указывай источник информации, если он известен.
    Отвечай на русском языке.
    Структурируй ответ с использованием эмодзи для лучшего восприятия.
    
    При работе с возрастом пациента учитывай текущую дату и корректируй возраст соответствующим образом."""
    
    if ai_mode == "doctor":
        system_prompt = base_system_prompt + """
        
        Ты — ИИ-врач главный, опытный медицинский специалист с глубокими знаниями в медицине. 
        Твоя задача — давать профессиональные медицинские консультации, анализировать результаты анализов, 
        интерпретировать медицинские данные и предоставлять квалифицированные рекомендации.
        
        ИНСТРУКЦИЯ: Используй все свои знания для глубокой интерпретации медицинских данных. 
        Предоставляй подробный анализ, возможные причины отклонений и рекомендации по дальнейшим действиям."""
        logging.info("Создан расширенный системный промпт для режима врача")
    else:
        system_prompt = base_system_prompt
        logging.info("Используется базовый системный промпт для режима ассистента")
    
    # Улучшенная логика обработки запроса
    if query_analysis["is_specific_indicator_question"] and medical_context:
        # Вопрос об анализах с доступными данными - используем только локальные данные
        logging.info("Обрабатываю вопрос об анализах с доступными медицинскими данными")
        await processing_msg.edit_text(f"📊 Анализирую ваши медицинские данные... ({mode_indicator})")
        
        # Добавляем дополнительную инструкцию для поиска конкретных показателей
        enhanced_system_prompt = system_prompt + f"""
        
        ДОПОЛНИТЕЛЬНАЯ ИНСТРУКЦИЯ: Пользователь спрашивает о конкретном показателе: "{question}"
        
        ВНИМАТЕЛЬНО изучи предоставленные медицинские данные и найди информацию об этом показателе.
        Если нашел - приведи точные значения и результаты.
        Если не нашел - честно скажи, что такой информации нет в предоставленных данных.
        НЕ придумывай данные, которых нет в контексте."""
        
        answer, provider, metadata = await generate_answer_with_failover(
            question, medical_context, history, profile, str(user_id), enhanced_system_prompt, model_type="text"
        )
        source = "ваших медицинских данных"
        
    else:
        # Стандартная обработка для остальных запросов
        logging.info("Обрабатываю стандартный запрос")
        # 1. Сначала ищем в авторитетных медицинских источниках
        medical_context_external = await search_medical_sources(question)
        if medical_context_external:
            logging.info("Найдена информация в медицинских источниках")
            await processing_msg.edit_text(f"📚 Найдено в медицинских источниках. Генерирую ответ... ({mode_indicator})")
            answer, provider, metadata = await generate_answer_with_failover(
                question, medical_context_external + medical_context, history, profile, str(user_id), system_prompt, model_type="text"
            )
            source = "авторитетных медицинских источников"
        else:
            # 2. Если не нашли в медицинских источниках, ищем в своей базе знаний
            logging.info("Ищу в базе знаний")
            await processing_msg.edit_text(f"🗂️ Ищу в накопленной базе знаний... ({mode_indicator})")
            kb_context = search_knowledge_base(question)
            if kb_context:
                logging.info("Найдена информация в базе знаний")
                await processing_msg.edit_text(f"💡 Найдено в базе знаний. Генерирую ответ... ({mode_indicator})")
                answer, provider, metadata = await generate_answer_with_failover(
                    question, kb_context + medical_context, history, profile, str(user_id), system_prompt, model_type="text"
                )
                source = "накопленной базы знаний"
            else:
                # 3. Если нигде не нашли, ищем в интернете
                logging.info("Ищу в интернете")
                await processing_msg.edit_text(f"🌐 Ищу дополнительную информацию в интернете... ({mode_indicator})")
                web_context = await search_web(f"{question} медицина здоровье")
                answer, provider, metadata = await generate_answer_with_failover(
                    question, web_context + medical_context, history, profile, str(user_id), system_prompt, model_type="text"
                )
                source = "интернета"
    
    await processing_msg.delete()
    history.append({"role": "assistant", "content": answer})
    
    logging.info(f"Ответ сгенерирован, длина: {len(answer)} символов")
    logging.info(f"Источник информации: {source}")
    logging.info(f"Провайдер ИИ: {provider}")
    
    # Добавляем индикатор режима ИИ в ответ
    mode_text = f"\n\n{mode_indicator}"
    
    await message.answer(f"{escape_html(answer)}\n\n📖 <b>Источник:</b> {escape_html(source)}{mode_text}", parse_mode="HTML")
    await message.answer("❓ Помог ли вам мой ответ?", reply_markup=get_feedback_keyboard())
    
    logging.info("Ответ отправлен пользователю, запрашиваю обратную связь")
    
    # Сохраняем состояние
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
    
    logging.info("Состояние сохранено, добавлено напоминание")
    
    # Добавляем напоминание
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
    logging.info(f"Обратная связь от пользователя {callback.from_user.id}: {callback.data}")
    
    data = await state.get_data()
    user_id = data.get("user_id")
    question = data.get("question", "")
    answer = data.get("answer", "")
    source = data.get("source", "")
    provider = data.get("provider", "")
    
    logging.info(f"Данные обратной связи: вопрос: {question[:50]}..., источник: {source}, провайдер: {provider}")
    
    if callback.data == "feedback_yes":
        logging.info("Пользователь подтвердил, что ответ помог")
        await callback.message.edit_text("✅ Спасибо за обратную связь! Рад, что смог помочь.")
        
        # Сохраняем положительную обратную связь
        if user_id:
            save_user_feedback(user_id, question, True)
            logging.info("Положительная обратная связь сохранена")
        
    elif callback.data == "feedback_no":
        logging.info("Пользователь указал, что ответ не помог")
        await callback.message.edit_text("😔 Извините, что не смог помочь. Попробуйте переформулировать вопрос или загрузить дополнительные анализы.")
        
        # Сохраняем отрицательную обратную связь
        if user_id:
            save_user_feedback(user_id, question, False)
            logging.info("Отрицательная обратная связь сохранена")
        
    elif callback.data == "search_more":
        logging.info("Пользователь запросил дополнительный поиск")
        await callback.message.edit_text("🔍 Ищу дополнительную информацию...")
        
        # Ищем дополнительную информацию
        try:
            # Ищем в медицинских источниках
            medical_context = await search_medical_sources(question)
            if medical_context:
                logging.info("Найдена дополнительная информация в медицинских источниках")
                additional_answer, additional_provider, additional_metadata = await generate_answer_with_failover(
                    question, medical_context, [], None, user_id, None, model_type="text"
                )
                await callback.message.edit_text(f"🔍 Дополнительная информация:\n\n{additional_answer}")
            else:
                logging.info("Дополнительная информация не найдена")
                await callback.message.edit_text("😔 К сожалению, не удалось найти дополнительную информацию. Попробуйте переформулировать вопрос.")
        except Exception as e:
            logging.error(f"Ошибка при поиске дополнительной информации: {e}")
            await callback.message.edit_text("😔 Произошла ошибка при поиске дополнительной информации. Попробуйте позже.")
    
    # Очищаем состояние
    await state.clear()
    logging.info("Состояние очищено после обработки обратной связи")


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
        birth_date = patient_data.get("birth_date")
        
        # Создаем профиль
        if create_patient_profile(generate_user_uuid(callback.from_user.id), name, age, gender, callback.from_user.id, birth_date):
            await callback.message.edit_text(
                f"✅ Профиль успешно создан!\n\n"
                f"👤 <b>Ваш профиль:</b>\n"
                f"📝 Имя: {name}\n"
                f"🎂 Возраст: {age if age else 'не указан'}\n"
                f"⚧️ Пол: {gender if gender else 'не указан'}\n"
                f"📅 Дата рождения: {birth_date if birth_date else 'не указана'}\n\n"
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


# Обработчик обновления профиля пациента
@dp.callback_query(F.data.in_(["update_profile_data", "keep_existing_data"]))
async def handle_profile_update_callback(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    profile_updates = data.get("profile_updates", {})
    analysis_result = data.get("analysis_result", "")
    pdf_text = data.get("pdf_text", "")
    
    # Определяем тип контента для отображения
    content_text = analysis_result if analysis_result else pdf_text[:500] + "..." if pdf_text else "обработанный документ"
    
    if callback.data == "update_profile_data":
        # Обновляем профиль
        user_id = generate_user_uuid(callback.from_user.id)
        if update_patient_profile(user_id, **profile_updates):
            # Получаем обновленный профиль для отображения
            updated_profile = get_patient_profile(user_id)
            
            update_summary = "✅ Профиль успешно обновлен!\n\n"
            update_summary += "👤 <b>Обновленные данные:</b>\n"
            
            for field, value in profile_updates.items():
                if field == "name":
                    update_summary += f"📝 Имя: {value}\n"
                elif field == "age":
                    update_summary += f"🎂 Возраст: {value}\n"
                elif field == "gender":
                    update_summary += f"⚧️ Пол: {value}\n"
                elif field == "birth_date":
                    update_summary += f"📅 Дата рождения: {value}\n"
            
            update_summary += f"\n📊 <b>Результат обработки:</b>\n\n{escape_html(content_text)}\n\n"
            update_summary += "⚠️ Помните, что это автоматический анализ, и он не заменяет консультацию специалиста."
            
            await callback.message.edit_text(update_summary, parse_mode="HTML")
        else:
            await callback.message.edit_text("😔 Не удалось обновить профиль. Пожалуйста, попробуйте еще раз.")
    else:
        # Оставляем существующие данные
        await callback.message.edit_text(
            f"ℹ️ Профиль оставлен без изменений.\n\n"
            f"📊 <b>Результат обработки:</b>\n\n{escape_html(content_text)}\n\n"
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
        birth_date = patient_data.get("birth_date")
        
        # Создаем профиль
        if create_patient_profile(generate_user_uuid(callback.from_user.id), name, age, gender, callback.from_user.id, birth_date):
            await callback.message.edit_text(
                f"✅ Профиль успешно создан!\n\n"
                f"👤 <b>Ваш профиль:</b>\n"
                f"📝 Имя: {name}\n"
                f"🎂 Возраст: {age if age else 'не указан'}\n"
                f"⚧️ Пол: {gender if gender else 'не указан'}\n"
                f"📅 Дата рождения: {birth_date if birth_date else 'не указана'}\n\n"
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
        logging.info(f"Отправка напоминания в чат {chat_id}")
        
        await bot.send_message(
            chat_id,
            "🔔 Напоминаю: помог ли вам мой предыдущий ответ?",
            reply_markup=get_feedback_keyboard()
        )
        
        logging.info(f"Напоминание успешно отправлено в чат {chat_id}")
        
    except Exception as e:
        logging.error(f"Ошибка при отправке напоминания в чат {chat_id}: {e}")


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
    
    # Добавляем задачу для еженедельной очистки дубликатов (каждое воскресенье в 2:00)
    scheduler.add_job(
        cleanup_all_duplicates,
        "cron",
        day_of_week="sun",
        hour=2,
        minute=0,
        id="cleanup_duplicates"
    )
    logging.info("Добавлена задача еженедельной очистки дубликатов")


@dp.shutdown()
async def on_shutdown():
    logging.info("Остановка планировщика задач")
    scheduler.shutdown()
    logging.info("Планировщик задач остановлен")


# Запуск бота
async def main():
    logging.info("Запуск бота")
    try:
        await dp.start_polling(bot)
        logging.info("Бот успешно запущен")
    except Exception as e:
        logging.error(f"Ошибка при запуске бота: {e}")
        raise


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
                full_context = "\n\n�� ВАШИ МЕДИЦИНСКИЕ ДАННЫЕ:\n"
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

# Создаем экземпляр интеллектуального анализатора
intelligent_analyzer = IntelligentQueryAnalyzer()


# Функция для очистки существующих дубликатов медицинских записей
def cleanup_duplicate_medical_records(user_id: str, record_type: str = "image_analysis") -> int:
    """
    Удаляет дублирующиеся записи для пользователя, оставляя только самую новую.
    Использует улучшенную проверку дубликатов по точным критериям.
    Возвращает количество удаленных записей.
    """
    try:
        logging.info(f"Очистка дубликатов для пользователя: {user_id}")
        
        # Получаем все записи пользователя
        query = supabase.table("doc_medical_records").select("*").eq("user_id", user_id).eq("record_type", record_type)
        response = query.order("created_at", desc=True).execute()
        
        if not response.data or len(response.data) <= 1:
            logging.info("Дубликатов для очистки не найдено")
            return 0
        
        records = response.data
        duplicates_to_delete = []
        processed_records = []
        
        # Находим дубликаты по точным критериям
        for i, record in enumerate(records):
            if record["id"] in duplicates_to_delete:
                continue
                
            current_content = record.get("content", "")
            is_duplicate = False
            
            # Сравниваем с уже обработанными записями
            for processed_record in processed_records:
                processed_content = processed_record.get("content", "")
                
                # Сначала проверяем по точным критериям
                if is_exact_duplicate_by_criteria(current_content, processed_content):
                    duplicates_to_delete.append(record["id"])
                    logging.info(f"Найден дубликат по точным критериям: ID {record['id']}")
                    logging.info(f"  Содержимое: {current_content[:100]}...")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                processed_records.append(record)
        
        if not duplicates_to_delete:
            logging.info("Дубликатов для удаления не найдено")
            return 0
        
        logging.info(f"Найдено {len(duplicates_to_delete)} дубликатов для удаления")
        
        # Удаляем дубликаты
        deleted_count = 0
        for record_id in duplicates_to_delete:
            try:
                delete_response = supabase.table("doc_medical_records").delete().eq("id", record_id).execute()
                if delete_response.data:
                    deleted_count += 1
                    logging.info(f"Удален дубликат с ID: {record_id}")
            except Exception as e:
                logging.error(f"Ошибка при удалении записи {record_id}: {e}")
        
        logging.info(f"Удалено {deleted_count} дублирующихся записей")
        return deleted_count
        
    except Exception as e:
        logging.error(f"Ошибка при очистке дубликатов: {e}")
        return 0


# Функция для очистки дубликатов у всех пользователей
def cleanup_all_duplicates():
    """
    Очищает дубликаты у всех пользователей в системе.
    Используется для автоматической очистки по расписанию.
    """
    try:
        logging.info("Начинаю автоматическую очистку дубликатов для всех пользователей")
        
        # Получаем всех пользователей
        response = supabase.table("doc_patient_profiles").select("user_id").execute()
        
        if not response.data:
            logging.info("Пользователей для очистки не найдено")
            return
        
        total_deleted = 0
        for profile in response.data:
            user_id = profile.get("user_id")
            if user_id:
                deleted_count = cleanup_duplicate_medical_records(user_id)
                total_deleted += deleted_count
        
        logging.info(f"Автоматическая очистка завершена. Всего удалено {total_deleted} дублирующихся записей")
        
    except Exception as e:
        logging.error(f"Ошибка при автоматической очистке дубликатов: {e}")


# Создаем экземпляр интеллектуального анализатора
intelligent_analyzer = IntelligentQueryAnalyzer()


# Функция для извлечения результатов анализов из текста
def extract_analysis_results(content: str) -> dict:
    """
    Извлекает результаты анализов из текста с фокусом на anti-HEV IgG и другие тесты
    """
    results = {}
    try:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            # Ищем строки с результатами анализов
            if '**' in line and ('anti-' in line.lower() or 'ige' in line.lower() or 'igg' in line.lower()):
                # Ищем строки типа "8. **Anti-HEV IgG:** ОТРИЦАТЕЛЬНО"
                if ':' in line:
                    # Разделяем по первому двоеточию
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        # Очищаем название теста от номеров и звездочек
                        test_name = parts[0].strip()
                        test_name = test_name.replace('**', '').replace('*', '').strip()
                        
                        # Убираем номера в начале строки
                        import re
                        test_name = re.sub(r'^\d+\.\s*', '', test_name).strip()
                        
                        # Приводим к нижнему регистру для сравнения
                        test_name_lower = test_name.lower()
                        
                        # Очищаем результат
                        result = parts[1].strip()
                        
                        # Если результат содержит только **, ищем следующую строку с результатом
                        if result.strip() == '**' and i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if 'результат:' in next_line.lower() and ':' in next_line:
                                result_parts = next_line.split(':', 1)
                                if len(result_parts) == 2:
                                    result = result_parts[1].strip()
                        
                        # Ищем конкретные результаты
                        if 'отрицательно' in result.lower():
                            result = 'отрицательно'
                        elif 'положительно' in result.lower():
                            result = 'положительно'
                        elif 'мл' in result.lower() or 'ме/мл' in result.lower():
                            # Числовое значение
                            numbers = re.findall(r'\d+', result)
                            if numbers:
                                result = f"{numbers[0]} единиц"
                        
                        # Сохраняем результат
                        results[test_name_lower] = result
                        
    except Exception as e:
        logging.error(f"Ошибка при извлечении результатов: {e}")
    
    return results


# Функция для извлечения даты анализа из текста
def extract_analysis_date(content: str) -> str:
    """
    Извлекает дату анализа из текста
    """
    try:
        lines = content.split('\n')
        for line in lines:
            line = line.strip().lower()
            # Ищем различные форматы дат
            if any(keyword in line for keyword in ['дата анализа:', 'дата сдачи:', 'дата:', 'сдано:']):
                if ':' in line:
                    date_part = line.split(':', 1)[1].strip()
                    # Очищаем дату от лишних символов
                    import re
                    date_match = re.search(r'\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4}', date_part)
                    if date_match:
                        return date_match.group()
        return ""
    except Exception as e:
        logging.error(f"Ошибка при извлечении даты анализа: {e}")
        return ""


# Функция для извлечения информации о пациенте из текста
def extract_patient_info(content: str) -> dict:
    """
    Извлекает информацию о пациенте из текста
    """
    patient = {}
    try:
        lines = content.split('\n')
        for line in lines:
            line = line.strip().lower()
            if 'имя пациента:' in line or 'фио пациента:' in line:
                name = line.split(':', 1)[1].strip() if ':' in line else ''
                patient['name'] = name
            elif 'дата рождения:' in line:
                birth_date = line.split(':', 1)[1].strip() if ':' in line else ''
                patient['birth_date'] = birth_date
            elif 'возраст:' in line:
                age = line.split(':', 1)[1].strip() if ':' in line else ''
                patient['age'] = age
                
    except Exception as e:
        logging.error(f"Ошибка при извлечении информации о пациенте: {e}")
    
    return patient


# Функция для точной проверки дубликатов по критериям пользователя
def is_exact_duplicate_by_criteria(content1: str, content2: str) -> bool:
    """
    Проверяет дубликаты по точным критериям: тест, результат и дата анализа
    """
    try:
        # Извлекаем результаты анализов
        results1 = extract_analysis_results(content1)
        results2 = extract_analysis_results(content2)
        
        # Извлекаем даты анализов
        date1 = extract_analysis_date(content1)
        date2 = extract_analysis_date(content2)
        
        # Если даты не совпадают, это не дубликат
        if date1 and date2 and date1 != date2:
            return False
        
        # Проверяем совпадение по anti-HEV IgG (приоритетный тест)
        if 'anti-hev igg' in results1 and 'anti-hev igg' in results2:
            if results1['anti-hev igg'] == results2['anti-hev igg']:
                logging.info(f"Найден точный дубликат anti-HEV IgG: {results1['anti-hev igg']}, дата: {date1}")
                return True
        
        # Проверяем другие тесты
        for test_name in results1:
            if test_name in results2 and results1[test_name] == results2[test_name]:
                if date1 and date2:  # Если даты совпадают
                    logging.info(f"Найден точный дубликат теста {test_name}: {results1[test_name]}, дата: {date1}")
                    return True
        
        return False
        
    except Exception as e:
        logging.error(f"Ошибка при точной проверке дубликатов: {e}")
        return False


# Функция для интеллектуальной проверки дублирования медицинских записей с приоритетом точных критериев
async def check_duplicate_medical_record_ai_enhanced(user_id: str, content: str, record_type: str = "image_analysis") -> bool:
    """
    Улучшенная проверка дублирования с приоритетом точных критериев
    """
    try:
        logging.info(f"Улучшенная ИИ-проверка дублирования для пользователя: {user_id}")
        
        # Получаем последние записи пользователя
        query = supabase.table("doc_medical_records").select("*").eq("user_id", user_id).eq("record_type", record_type)
        response = query.order("created_at", desc=True).limit(10).execute()
        
        if not response.data:
            logging.info("Записей для сравнения не найдено")
            return False
        
        # Сначала проверяем по точным критериям
        for record in response.data:
            if is_exact_duplicate_by_criteria(content, record.get("content", "")):
                logging.info(f"Точные критерии обнаружили дубликат записи с ID: {record.get('id')}")
                return True
        
        # Если точные критерии не сработали, используем ИИ как fallback
        for record in response.data:
            if await is_duplicate_by_ai(content, record.get("content", "")):
                logging.info(f"ИИ обнаружил дубликат записи с ID: {record.get('id')}")
                return True
        
        logging.info("Дубликаты не обнаружены")
        return False
        
    except Exception as e:
        logging.error(f"Ошибка при улучшенной ИИ-проверке дублирования: {e}")
        # В случае ошибки, возвращаем False чтобы не блокировать сохранение
        return False


if __name__ == "__main__":
    asyncio.run(main())