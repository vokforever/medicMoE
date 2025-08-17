# AI Doctor Bot - Модульная структура

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка переменных окружения
Создайте файл `.env` в корне проекта:
```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token_here

# OpenRouter API
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_DAILY_LIMIT=100000

# Supabase
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# Tavily API
TAVILY_API_KEY=your_tavily_key_here

# Groq API (опционально)
GROQ_API_KEY=your_groq_key_here
GROQ_DAILY_LIMIT=50000

# Cerebras API (опционально)
CEREBRAS_API_KEY=your_cerebras_key_here
CEREBRAS_DAILY_LIMIT=50000

# Mistral AI (для эмбеддингов)
MISTRAL_API_KEY=your_mistral_key_here
```

### 3. Запуск бота
```bash
python main_new.py
```

## 📁 Структура проекта

```
doc_rag_supa_moe/
├── config.py                 # Конфигурация и клиенты
├── models.py                 # Работа с ИИ-моделями
├── agents.py                 # Интеллектуальные агенты
├── database.py               # Операции с базой данных
├── utils.py                  # Утилиты и вспомогательные функции
├── keyboards.py              # Клавиатуры Telegram
├── main_new.py               # Основной файл бота
├── structured_tests_agent.py # Агент структурированных анализов
├── PROJECT_STRUCTURE.md      # Детальное описание структуры
├── .cursorrules              # Правила для Cursor IDE
├── README_MODULAR.md         # Этот файл
└── requirements.txt          # Зависимости
```

## 🔧 Основные модули

### `config.py` - Конфигурация
Центральный модуль с настройками:
- API ключи и клиенты
- Конфигурация ИИ-моделей
- Константы и лимиты

### `models.py` - ИИ-модели
Управляет вызовом ИИ с failover:
- Автоматическое переключение между провайдерами
- Управление лимитами токенов
- Обработка ошибок

### `agents.py` - Интеллектуальные агенты
Агенты для анализа и принятия решений:
- `ClarificationAgent` - определяет достаточность информации
- `TestAnalysisAgent` - анализирует медицинские анализы
- `IntelligentQueryAnalyzer` - определяет тип запроса

### `database.py` - База данных
Операции с Supabase:
- Профили пациентов
- Медицинские записи
- Результаты анализов

### `utils.py` - Утилиты
Вспомогательные функции:
- Поиск в медицинских источниках
- Векторный поиск
- Обработка изображений и PDF

### `keyboards.py` - Клавиатуры
Инлайн-клавиатуры для Telegram:
- Главное меню
- Обратная связь
- Управление профилями

## 💡 Как использовать модули

### Добавление новой функции

1. **Определите модуль**: В какой модуль добавить функцию?
   - Конфигурация → `config.py`
   - ИИ-модели → `models.py`
   - Агенты → `agents.py`
   - База данных → `database.py`
   - Утилиты → `utils.py`
   - Клавиатуры → `keyboards.py`

2. **Следуйте паттернам**:
   ```python
   import logging
   from typing import Optional, Dict, Any
   
   def new_function(param: str) -> Optional[Dict[str, Any]]:
       """
       Описание функции на русском языке.
       
       Args:
           param: Описание параметра
           
       Returns:
           Описание возвращаемого значения
           
       Raises:
           Exception: Описание исключения
       """
       try:
           logging.info(f"Выполнение функции с параметром: {param}")
           
           # Ваша логика здесь
           result = {"status": "success", "data": param}
           
           logging.info(f"Функция выполнена успешно: {result}")
           return result
           
       except Exception as e:
           logging.error(f"Ошибка в функции: {e}")
           return None
   ```

3. **Импортируйте в main_new.py**:
   ```python
   from utils import new_function
   
   # Используйте в обработчике
   result = new_function("test")
   ```

### Создание нового агента

1. **Создайте класс в `agents.py`**:
   ```python
   class NewAgent:
       def __init__(self):
           self.name = "New Agent"
       
       async def process(self, data: str) -> str:
           """Обработка данных"""
           # Ваша логика здесь
           return f"Обработано: {data}"
   ```

2. **Инициализируйте в `main_new.py`**:
   ```python
   from agents import NewAgent
   
   new_agent = NewAgent()
   ```

### Добавление новой клавиатуры

1. **Создайте функцию в `keyboards.py`**:
   ```python
   def get_new_keyboard() -> types.InlineKeyboardMarkup:
       builder = InlineKeyboardBuilder()
       builder.add(types.InlineKeyboardButton(
           text="Новая кнопка",
           callback_data="new_action"
       ))
       return builder.as_markup()
   ```

2. **Импортируйте и используйте**:
   ```python
   from keyboards import get_new_keyboard
   
   await message.answer("Выберите действие:", reply_markup=get_new_keyboard())
   ```

## 🔍 Отладка и диагностика

### Проверка статуса моделей
```bash
# В Telegram боте
/models
```

### Логирование
Все модули имеют детальное логирование:
```python
import logging

# Уровни логирования
logging.debug("Детальная информация")
logging.info("Общая информация")
logging.warning("Предупреждения")
logging.error("Ошибки")
```

### Проверка подключений
```python
# Проверка Supabase
from config import supabase
try:
    response = supabase.table("doc_patient_profiles").select("*").limit(1).execute()
    print("Supabase подключен")
except Exception as e:
    print(f"Ошибка Supabase: {e}")
```

## 📊 Мониторинг производительности

### Использование токенов
- Отслеживается в `TOKEN_LIMITS`
- Сбрасывается ежедневно в полночь
- Логируется каждое использование

### Размер базы данных
```python
# Проверка размера таблиц
from config import supabase

def check_table_sizes():
    tables = ["doc_patient_profiles", "doc_medical_records", "doc_test_results"]
    for table in tables:
        response = supabase.table(table).select("*", count="exact").execute()
        print(f"{table}: {response.count} записей")
```

### Время выполнения
```python
import time

def measure_performance(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Функция {func.__name__} выполнена за {end - start:.2f} секунд")
        return result
    return wrapper
```

## 🚀 Развертывание

### 1. Подготовка
```bash
# Установка зависимостей
pip install -r requirements.txt

# Проверка переменных окружения
python -c "from config import *; print('Конфигурация загружена')"
```

### 2. Запуск
```bash
# Обычный запуск
python main_new.py

# С логированием в файл
python main_new.py > bot.log 2>&1

# В фоне
nohup python main_new.py > bot.log 2>&1 &
```

### 3. Мониторинг
```bash
# Проверка процесса
ps aux | grep main_new.py

# Просмотр логов
tail -f bot.log

# Проверка статуса
curl -s http://localhost:8000/health || echo "Бот не отвечает"
```

## 🧪 Тестирование

### Модульные тесты
```python
# test_models.py
import pytest
from models import check_model_availability

async def test_model_availability():
    result = await check_model_availability("openrouter", "test-model")
    assert isinstance(result, bool)
```

### Интеграционные тесты
```python
# test_integration.py
import pytest
from database import create_patient_profile, get_patient_profile

async def test_patient_profile_flow():
    # Создание профиля
    success = await create_patient_profile("test-user", "Test", 30, "М")
    assert success is True
    
    # Получение профиля
    profile = get_patient_profile("test-user")
    assert profile["name"] == "Test"
```

## 📚 Полезные команды

### Git
```bash
# Создание новой ветки для функции
git checkout -b feature/new-function

# Проверка изменений
git diff

# Коммит изменений
git add .
git commit -m "Добавлена новая функция для анализа анализов"
```

### Python
```bash
# Проверка синтаксиса
python -m py_compile config.py

# Импорт модуля для проверки
python -c "import models; print('Модуль models загружен')"

# Запуск с отладкой
python -v main_new.py
```

## 🆘 Решение проблем

### Частые ошибки

1. **ImportError: No module named 'config'**
   - Убедитесь, что находитесь в корневой папке проекта
   - Проверьте, что все файлы созданы

2. **Supabase connection error**
   - Проверьте переменные окружения
   - Убедитесь в правильности URL и ключа

3. **AI model errors**
   - Проверьте API ключи
   - Используйте команду `/models` для диагностики

4. **Memory errors**
   - Ограничьте размер контекста в `config.py`
   - Уменьшите `MAX_HISTORY_LENGTH`

### Получение помощи

1. **Проверьте логи** - детальная информация об ошибках
2. **Используйте команду `/models`** - проверка статуса ИИ
3. **Проверьте конфигурацию** - переменные окружения
4. **Изучите документацию** - `PROJECT_STRUCTURE.md`

## 🔄 Обновления и миграции

### Обновление модуля
1. Создайте резервную копию
2. Внесите изменения
3. Протестируйте функциональность
4. Обновите документацию

### Миграция базы данных
1. Создайте SQL скрипт миграции
2. Протестируйте на тестовой базе
3. Выполните на продакшене
4. Обновите код при необходимости

## 📈 Масштабирование

### Горизонтальное масштабирование
- Запустите несколько экземпляров бота
- Используйте Redis для общего состояния
- Балансируйте нагрузку через nginx

### Вертикальное масштабирование
- Увеличьте лимиты токенов
- Добавьте больше ИИ-моделей
- Оптимизируйте запросы к базе данных

## 🎯 Лучшие практики

1. **Всегда логируйте** - для отладки и мониторинга
2. **Обрабатывайте ошибки** - используйте try-catch блоки
3. **Документируйте код** - добавляйте docstrings
4. **Тестируйте изменения** - перед развертыванием
5. **Следуйте паттернам** - используйте существующие структуры
6. **Мониторьте производительность** - отслеживайте использование ресурсов

## 📞 Поддержка

Для получения помощи:
1. Изучите документацию проекта
2. Проверьте логи и ошибки
3. Используйте команды диагностики
4. Обратитесь к разработчикам

---

**Удачной разработки! 🚀**
