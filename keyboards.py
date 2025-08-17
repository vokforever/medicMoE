import logging
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram import types

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
        text="📋 Структурированные анализы",
        callback_data="structured_tests"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📝 Мой анамнез",
        callback_data="my_history"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🆔 Создать профиль пациента",
        callback_data="create_profile"
    ))
    builder.adjust(2, 2)
    
    logging.debug("Главная клавиатура создана")
    return builder.as_markup()

# Функция для создания клавиатуры подтверждения профиля
def get_profile_confirmation_keyboard():
    logging.debug("Создание клавиатуры подтверждения профиля")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, использовать",
        callback_data="use_extracted_data"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Нет, создать анонимный профиль",
        callback_data="create_anonymous_profile"
    ))
    builder.adjust(1)
    
    logging.debug("Клавиатура подтверждения профиля создана")
    return builder.as_markup()

# Функция для создания клавиатуры обновления профиля
def get_profile_update_keyboard():
    logging.debug("Создание клавиатуры обновления профиля")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, обновить",
        callback_data="update_profile_data"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Нет, оставить как есть",
        callback_data="keep_existing_data"
    ))
    builder.adjust(1)
    
    logging.debug("Клавиатура обновления профиля создана")
    return builder.as_markup()

# Функция для создания клавиатуры анализа PDF
def get_pdf_analysis_keyboard():
    logging.debug("Создание клавиатуры анализа PDF")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, проанализировать",
        callback_data="analyze_pdf"
    ))
    builder.adjust(1)
    
    logging.debug("Клавиатура анализа PDF создана")
    return builder.as_markup()

# Функция для создания клавиатуры дополнения данных
def get_complete_data_keyboard():
    logging.debug("Создание клавиатуры дополнения данных")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Дополнить данные",
        callback_data="complete_test_data"
    ))
    builder.adjust(1)
    
    logging.debug("Клавиатура дополнения данных создана")
    return builder.as_markup()

# Функция для создания клавиатуры добавления даты
def get_add_date_keyboard(test_id: int):
    logging.debug(f"Создание клавиатуры добавления даты для теста {test_id}")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Добавить дату",
        callback_data=f"add_test_date_{test_id}"
    ))
    builder.adjust(1)
    
    logging.debug("Клавиатура добавления даты создана")
    return builder.as_markup()
