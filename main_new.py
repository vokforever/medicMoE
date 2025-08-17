import os
import asyncio
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Импорты из наших модулей
from config import bot_token, supabase, reset_token_usage
from models import call_model_with_failover
from agents import ClarificationAgent, TestAnalysisAgent, IntelligentQueryAnalyzer
from database import (
    generate_user_uuid, create_patient_profile, update_patient_profile,
    get_patient_profile, save_test_results, get_patient_tests,
    get_medical_records, save_medical_record, save_to_knowledge_base,
    save_user_feedback, get_user_successful_responses, save_successful_response
)
from utils import (
    escape_html, search_medical_sources, search_web, search_knowledge_base,
    extract_patient_data_from_text, analyze_image, extract_text_from_pdf,
    check_duplicate_medical_record_ai_enhanced
)
from keyboards import (
    get_feedback_keyboard, get_clarification_keyboard, get_main_keyboard,
    get_profile_confirmation_keyboard, get_profile_update_keyboard,
    get_pdf_analysis_keyboard, get_complete_data_keyboard, get_add_date_keyboard
)

# Импорт и инициализация агента для структурированных данных
from structured_tests_agent import StructuredTestAgent
structured_test_agent = StructuredTestAgent(supabase)

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
    from config import MODEL_CONFIG, TOKEN_LIMITS
    from models import check_model_availability
    
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
        from utils import cleanup_duplicate_medical_records
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

if __name__ == "__main__":
    asyncio.run(main())
