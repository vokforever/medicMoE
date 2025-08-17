import os
import asyncio
import requests
import json
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
openai_client = OpenAI(
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


# Агент для анализа анализов на основе horizon-beta
class TestAnalysisAgent:
    def __init__(self):
        self.model = "openrouter/horizon-beta"

    async def analyze_test_results(self, text: str) -> List[Dict[str, Any]]:
        """Анализ текста анализов и извлечение структурированных данных"""
        try:
            completion = openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Ты — медицинский эксперт по анализам. Извлеки из текста все результаты анализов в структурированном формате.

                        Для каждого анализа укажи:
                        1. Название анализа (на русском)
                        2. Значение
                        3. Референсные значения (норма)
                        4. Единицы измерения
                        5. Дату анализа (если есть)
                        6. Отклонение от нормы (если есть)

                        Верни ответ в формате JSON массива объектов:
                        [
                            {
                                "test_name": "Название анализа",
                                "value": "Значение",
                                "reference_range": "Референсные значения",
                                "unit": "Единицы измерения",
                                "test_date": "ГГГГ-ММ-ДД",
                                "is_abnormal": true/false,
                                "notes": "Примечания"
                            }
                        ]

                        Если даты нет, укажи null. Если референсные значения не указаны, укажи null."""
                    },
                    {
                        "role": "user",
                        "content": text[:4000]  # Ограничиваем длину текста
                    }
                ]
            )

            response_text = completion.choices[0].message.content

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

    async def get_test_summary(self, user_id: int, test_names: List[str] = None) -> str:
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
            completion = openai_client.chat.completions.create(
                model=self.model,
                messages=[
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
            )

            summary = completion.choices[0].message.content

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
async def save_test_results(user_id: int, test_results: List[Dict[str, Any]], source: str = ""):
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
def get_patient_tests(user_id: int, test_names: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
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
        completion = openai_client.chat.completions.create(
            model="z-ai/glm-4.5-air:free",
            messages=[
                {
                    "role": "system",
                    "content": """Ты — помощник, который извлекает данные пациента из медицинских документов. 
                    Извлеки имя, возраст и пол, если они есть. Верни ответ в формате JSON: 
                    {"name": "имя", "age": число, "gender": "М" или "Ж"}. 
                    Если каких-то данных нет, поставь null."""
                },
                {
                    "role": "user",
                    "content": text[:2000]
                }
            ]
        )
        response_text = completion.choices[0].message.content

        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return {
                    "name": data.get("name"),
                    "age": data.get("age"),
                    "gender": data.get("gender")
                }
        except json.JSONDecodeError:
            pass

        # Если не удалось извлечь JSON, пробуем простой парсинг
        name_match = re.search(r'(?:Пациент|ФИО|Имя):\s*([А-Яа-я\s]+)', text)
        age_match = re.search(r'(?:Возраст|Лет):\s*(\d+)', text)
        gender_match = re.search(r'(?:Пол):\s*([МЖ])', text)

        return {
            "name": name_match.group(1).strip() if name_match else None,
            "age": int(age_match.group(1)) if age_match else None,
            "gender": gender_match.group(1) if gender_match else None
        }
    except Exception as e:
        logging.error(f"Ошибка при извлечении данных пациента: {e}")
        return {}


# Функция для анализа изображения
async def analyze_image(image_url: str, query: str = "Что показано на этом медицинском изображении?") -> str:
    try:
        completion = openai_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/vokforever/ai-doctor",
                "X-Title": "AI Doctor Bot"
            },
            model="qwen/qwen2.5-vl-72b-instruct:free",
            messages=[
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
        )
        return completion.choices[0].message.content
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
def create_patient_profile(user_id: int, name: str, age: int, gender: str) -> bool:
    try:
        response = supabase.table("doc_patient_profiles").insert({
            "user_id": user_id,
            "name": name,
            "age": age,
            "gender": gender,
            "created_at": datetime.now().isoformat()
        }).execute()

        return len(response.data) > 0
    except Exception as e:
        logging.error(f"Ошибка при создании профиля пациента: {e}")
        return False


# Функция для получения профиля пациента
def get_patient_profile(user_id: int) -> Optional[Dict[str, Any]]:
    try:
        response = supabase.table("doc_patient_profiles").select("*").eq("user_id", user_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logging.error(f"Ошибка при получении профиля пациента: {e}")
        return None


# Функция для сохранения медицинских записей
def save_medical_record(user_id: int, record_type: str, content: str, source: str = "") -> bool:
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
def get_medical_records(user_id: int, record_type: str = None) -> List[Dict[str, Any]]:
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
def save_user_feedback(user_id: int, question: str, helped: bool):
    try:
        supabase.table("doc_user_feedback").insert({
            "user_id": user_id,
            "question": question,
            "helped": helped,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logging.error(f"Ошибка при сохранении обратной связи: {e}")


# Функция для генерации ответа с MOE подходом
async def generate_answer(question: str, context: str = "", history: List[Dict[str, str]] = None,
                          patient_data: Dict[str, Any] = None, user_id: int = None) -> str:
    models_to_try = [
        "openrouter/horizon-beta",
        "moonshotai/kimi-k2:free",
        "z-ai/glm-4.5-air:free",
        "openai/gpt-oss-20b:free",
        "mistralai/mistral-large-2407",
        "mistralai/mistral-7b-instruct"
    ]

    last_error = None
    best_answer = ""
    best_model = ""

    # MOE: пробуем несколько моделей и выбираем лучший ответ
    for model in models_to_try[:3]:
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Ты — ИИ-ассистент врача. Твоя задача — помогать пользователям с медицинскими вопросами, 
                    анализировать их анализы и предоставлять информацию о здоровье. Отвечай максимально точно и информативно, 
                    используя предоставленный контекст. Учитывай историю диалога и данные пациента, если они доступны.

                    ВАЖНО: Ты не ставишь диагноз и не заменяешь консультацию врача. Всегда рекомендуй консультацию 
                    со специалистом для точной диагностики и лечения.

                    Если в контексте есть точный ответ из авторитетных медицинских источников — используй его.
                    Всегда указывай источник информации, если он известен.
                    Отвечай на русском языке.
                    Структурируй ответ с использованием эмодзи для лучшего восприятия."""
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

            # Добавляем заголовки для OpenRouter
            extra_headers = {
                "HTTP-Referer": "https://github.com/vokforever/ai-doctor",
                "X-Title": "AI Doctor Bot"
            }

            completion = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                extra_headers=extra_headers
            )

            current_answer = completion.choices[0].message.content

            # Для MOE: оцениваем качество ответа
            if not best_answer or len(current_answer) > len(best_answer):
                best_answer = current_answer
                best_model = model

        except Exception as e:
            last_error = e
            logging.warning(f"Ошибка при использовании модели {model}: {e}")
            continue

    # Если получили хотя бы один ответ от MOE, возвращаем лучший
    if best_answer:
        logging.info(f"Использована модель: {best_model}")
        return best_answer

    # Если все модели MOE не сработали, пробуем оставшиеся модели
    for model in models_to_try[3:]:
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Ты — ИИ-ассистент врача. Твоя задача — помогать пользователям с медицинскими вопросами, 
                    анализировать их анализы и предоставлять информацию о здоровье. Отвечай максимально точно и информативно, 
                    используя предоставленный контекст. Учитывай историю диалога и данные пациента, если они доступны.

                    ВАЖНО: Ты не ставишь диагноз и не заменяешь консультацию врача. Всегда рекомендуй консультацию 
                    со специалистом для точной диагностики и лечения.

                    Если в контексте есть точный ответ из авторитетных медицинских источников — используй его.
                    Всегда указывай источник информации, если он известен.
                    Отвечай на русском языке.
                    Структурируй ответ с использованием эмодзи для лучшего восприятия."""
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

            # Добавляем заголовки для OpenRouter
            extra_headers = {
                "HTTP-Referer": "https://github.com/vokforever/ai-doctor",
                "X-Title": "AI Doctor Bot"
            }

            completion = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                extra_headers=extra_headers
            )
            return completion.choices[0].message.content

        except Exception as e:
            last_error = e
            logging.warning(f"Ошибка при использовании модели {model}: {e}")
            continue

    # Если все модели не сработали
    logging.error(f"Все модели недоступны. Последняя ошибка: {last_error}")
    return "😔 К сожалению, произошла ошибка при генерации ответа. Все модели временно недоступны. Попробуйте повторить запрос позже."


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

    profile = get_patient_profile(message.from_user.id)

    if profile:
        await message.answer(
            f"👋 Здравствуйте, {profile['name']}! Я ваш ИИ-ассистент врача.\n\n"
            f"📊 Я могу помочь вам с анализом анализов, ответить на медицинские вопросы и хранить ваш анамнез.\n\n"
            f"💡 Просто задайте ваш вопрос, или загрузите анализы для анализа.\n\n"
            f"📊 Доступные команды:\n"
            f"/profile - ваш профиль\n"
            f"/stats - моя статистика помощи\n"
            f"/history - история обращений\n"
            f"/clear - очистить историю",
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
            "/clear - очистить историю",
            reply_markup=get_main_keyboard()
        )


# Обработчик команды /profile
@dp.message(Command("profile"))
async def profile_command(message: types.Message, state: FSMContext):
    profile = get_patient_profile(message.from_user.id)

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
        response = supabase.table("doc_user_feedback").select("*").eq("user_id", message.from_user.id).execute()
        total = len(response.data)
        helped = sum(1 for item in response.data if item["helped"])

        await message.answer(
            f"📊 Ваша статистика:\n"
            f"Всего вопросов: {total}\n"
            f"Помогло ответов: {helped}\n"
            f"Успешность: {helped / total * 100:.1f}%" if total > 0 else "📊 У вас пока нет статистики"
        )
    except Exception as e:
        logging.error(f"Ошибка при получении статистики: {e}")
        await message.answer("😔 Не удалось загрузить статистику")


# Обработчик команды /history
@dp.message(Command("history"))
async def history_command(message: types.Message):
    try:
        response = supabase.table("doc_user_feedback").select("*").eq("user_id", message.from_user.id).order(
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

        supabase.table("doc_user_feedback").delete().eq("user_id", message.from_user.id).execute()
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
            if create_patient_profile(message.from_user.id, name, age, gender):
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
    profile = get_patient_profile(message.from_user.id)

    if message.document.mime_type == "application/pdf":
        file_id = message.document.file_id
        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/{file_path}"

        processing_msg = await message.answer("📊 Обрабатываю PDF файл...")

        pdf_text = await extract_text_from_pdf(file_url)

        if pdf_text:
            # Сохраняем в медицинские записи
            save_medical_record(
                user_id=message.from_user.id,
                record_type="analysis",
                content=pdf_text[:2000],
                source=f"PDF файл: {message.document.file_name}"
            )

            # Анализируем результаты анализов с помощью агента
            test_results = await test_agent.analyze_test_results(pdf_text)

            if test_results:
                # Сохраняем структурированные результаты
                await save_test_results(
                    user_id=message.from_user.id,
                    test_results=test_results,
                    source=f"PDF файл: {message.document.file_name}"
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
                                    text="✅ Да, создать",
                                    callback_data="create_extracted_profile"
                                ),
                                types.InlineKeyboardButton(
                                    text="❌ Нет, ввести вручную",
                                    callback_data="manual_profile"
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
    profile = get_patient_profile(message.from_user.id)

    if not profile:
        await message.answer(
            "😔 Для загрузки изображений необходимо создать профиль пациента.\n"
            "Используйте команду /profile для создания профиля."
        )
        return

    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    file_url = f"https://api.telegram.org/file/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/{file_path}"

    processing_msg = await message.answer("🔍 Анализирую изображение...")

    analysis_result = await analyze_image(file_url, "Что показано на этом медицинском изображении? Опиши подробно.")

    await processing_msg.edit_text("✅ Изображение успешно проанализировано.")

    save_medical_record(
        user_id=message.from_user.id,
        record_type="image_analysis",
        content=analysis_result,
        source="Изображение из Telegram"
    )

    await message.answer(
        f"📊 <b>Результат анализа:</b>\n\n{analysis_result}\n\n"
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
    history = data.get("history", [])

    history.append({"role": "user", "content": question})

    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]
        await message.answer(
            "🔄 История диалога стала слишком длинной, я удалил самые старые сообщения для оптимизации.")

    profile = get_patient_profile(user_id)

    processing_msg = await message.answer("🔍 Ищу информацию по вашему вопросу...")

    # Проверяем, есть ли в вопросе запрос на анализ анализов
    analysis_keywords = ['анализ', 'анализы', 'результат', 'показатель', 'кровь', 'моча', 'биохимия', 'общий анализ']
    test_context = ""

    if any(keyword in question.lower() for keyword in analysis_keywords):
        # Получаем сводку по анализам от агента
        test_summary = await test_agent.get_test_summary(user_id)
        if test_summary:
            test_context = f"\n\n📊 {test_summary}"

    # 1. Сначала ищем в авторитетных медицинских источниках
    medical_context = await search_medical_sources(question)

    if medical_context:
        await processing_msg.edit_text("📚 Найдено в медицинских источниках. Генерирую ответ...")
        answer = await generate_answer(question, medical_context + test_context, history, profile, user_id)
        source = "авторитетных медицинских источников"
    else:
        # 2. Если не нашли в медицинских источниках, ищем в своей базе знаний
        await processing_msg.edit_text("🗂️ Ищу в накопленной базе знаний...")
        kb_context = search_knowledge_base(question)

        if kb_context:
            await processing_msg.edit_text("💡 Найдено в базе знаний. Генерирую ответ...")
            answer = await generate_answer(question, kb_context + test_context, history, profile, user_id)
            source = "накопленной базы знаний"
        else:
            # 3. Если нигде не нашли, ищем в интернете
            await processing_msg.edit_text("🌐 Ищу дополнительную информацию в интернете...")
            web_context = await search_web(f"{question} медицина здоровье")
            answer = await generate_answer(question, web_context + test_context, history, profile, user_id)
            source = "интернета"

    await processing_msg.delete()

    history.append({"role": "assistant", "content": answer})

    await message.answer(f"{escape_html(answer)}\n\n📖 <b>Источник:</b> {escape_html(source)}", parse_mode="HTML")
    await message.answer("❓ Помог ли вам мой ответ?", reply_markup=get_feedback_keyboard())

    await state.set_state(DoctorStates.waiting_for_feedback)
    await state.update_data(
        question=question,
        answer=answer,
        source=source,
        attempts=0,
        user_id=user_id,
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
    attempts = data.get("attempts", 0)
    user_id = data.get("user_id", callback.from_user.id)
    chat_id = callback.message.chat.id

    if callback.data == "feedback_yes":
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

        history = data.get("history", [])
        profile = get_patient_profile(user_id)

        web_context = await search_web(f"{question} медицина диагноз лечение")
        new_answer = await generate_answer(question, web_context, history, profile, user_id)

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
            source="интернета (дополнительный поиск)",
            attempts=attempts + 1
        )


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

        profile = get_patient_profile(callback.from_user.id)

        await callback.message.edit_text("🔄 Пробую найти другой ответ...")

        web_context = await search_web(f"{question} медицина здоровье лечение")
        new_answer = await generate_answer(question, web_context, history, profile, callback.from_user.id)

        history.append({"role": "assistant", "content": new_answer})
        await state.update_data(history=history)

        await callback.message.edit_text(
            f"{escape_html(new_answer)}\n\n📖 <b>Источник:</b> дополнительный поиск в интернете",
            parse_mode="HTML",
            reply_markup=get_feedback_keyboard()
        )

        await state.update_data(answer=new_answer, source="интернета")
        await state.set_state(DoctorStates.waiting_for_feedback)

    elif callback.data == "analyze_pdf":
        data = await state.get_data()
        pdf_text = data.get("pdf_text", "")

        if pdf_text:
            await callback.message.edit_text("📊 Анализирую результаты анализов...")

            profile = get_patient_profile(callback.from_user.id)

            # Используем агента для анализа
            analysis_result = await test_agent.get_test_summary(callback.from_user.id)

            if analysis_result:
                save_medical_record(
                    user_id=callback.from_user.id,
                    record_type="analysis_result",
                    content=analysis_result,
                    source="Анализ PDF файла"
                )

                await callback.message.edit_text(
                    f"📊 <b>Результат анализа:</b>\n\n{analysis_result}\n\n"
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
                if create_patient_profile(callback.from_user.id, patient_data['name'], patient_data['age'],
                                          patient_data['gender']):
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
    user_id = callback.from_user.id

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
    data = await state.get_data()
    history = data.get("history", [])

    history.append({"role": "user", "content": message.text})
    await state.update_data(history=history)

    await state.clear()
    await handle_message(message, state)


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


# Планировщик для отложенных напоминаний
@dp.startup()
async def on_startup():
    scheduler.start()


@dp.shutdown()
async def on_shutdown():
    scheduler.shutdown()


# Запуск бота
async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())