import os
from openai import OpenAI
from tavily import TavilyClient
from supabase import create_client, Client
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
import logging
logging.basicConfig(level=logging.INFO)

# Инициализация клиентов
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# OpenRouter
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# Tavily API
tavily_client = TavilyClient(api_key=tavily_api_key)

# Supabase
supabase: Client = create_client(
    supabase_url=supabase_url,
    supabase_key=supabase_key
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
        "api_key": openrouter_api_key,
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
