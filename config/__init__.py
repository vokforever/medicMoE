"""
Конфигурационные файлы системы
"""

# Загрузка переменных окружения
from dotenv import load_dotenv
load_dotenv()

from .medical_config import medical_config, MedicalConfig

# Для обратной совместимости со старым кодом
try:
    # Пытаемся импортировать старые переменные из корневого config.py
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import MODEL_CONFIG, TOKEN_LIMITS, AGENT_CACHE_EXPIRE_HOURS
except ImportError:
    # Если старые переменные не найдены, используем значения по умолчанию
    # OpenRouter
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_client = __import__('openai').OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    
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
                {"name": "meta-llama/llama-4-scout-17b-16e-instruct", "priority": 1, "type": "vision"},
                {"name": "meta-llama/llama-4-maverick-17b-128e-instruct", "priority": 2, "type": "vision"},
                {"name": "llama-3.1-8b-instant", "priority": 3, "type": "text"},
                {"name": "llama-3.3-70b-versatile", "priority": 4, "type": "text"},
                {"name": "meta-llama/llama-guard-4-12b", "priority": 5, "type": "text"}
            ],
            "client": __import__('openai').OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY")
            )
        },
        "cerebras": {
            "api_key": os.getenv("CEREBRAS_API_KEY"),
            "base_url": "https://api.cerebras.ai/v1",
            "models": [
                {"name": "qwen-3-235b-a22b-thinking-2507", "priority": 1, "type": "text"},
                {"name": "qwen-3-235b-a22b-instruct-2507", "priority": 2, "type": "text"}
            ],
            "client": __import__('openai').OpenAI(
                base_url="https://api.cerebras.ai/v1",
                api_key=os.getenv("CEREBRAS_API_KEY")
            )
        }
    }
    
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
    AGENT_CACHE_EXPIRE_HOURS = 24

# Дополнительные переменные для обратной совместимости
MEDICAL_SOURCES = [
    "medlineplus.gov",
    "mayoclinic.org",
    "webmd.com",
    "nih.gov",
    "who.int"
]

# Для обратной совместимости со старым кодом
try:
    # Пытаемся импортировать supabase из корневого config.py
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import supabase
except ImportError:
    # Если старые переменные не найдены, инициализируем из окружения
    from supabase import create_client, Client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if supabase_url and supabase_key:
        supabase: Client = create_client(supabase_url, supabase_key)
    else:
        supabase = None

# Для обратной совместимости со старым кодом
try:
    # Пытаемся импортировать bot_token из корневого config.py
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import bot_token, tavily_client
except ImportError:
    # Если старые переменные не найдены, загружаем из окружения
    import os
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tavily_client = os.getenv("TAVILY_API_KEY")

__all__ = ['medical_config', 'MedicalConfig', 'MODEL_CONFIG', 'TOKEN_LIMITS', 'AGENT_CACHE_EXPIRE_HOURS', 'MEDICAL_SOURCES', 'supabase', 'bot_token', 'tavily_client']
