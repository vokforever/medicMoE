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

# Функция для создания клавиатуры управления анализами
def get_manage_tests_keyboard():
    logging.debug("Создание клавиатуры управления анализами")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="🗑️ Удалить анализы",
        callback_data="delete_tests"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🗑️ Удалить медицинские записи",
        callback_data="delete_medical_records"
    ))
    builder.add(types.InlineKeyboardButton(
        text="🗑️ Удалить все анализы",
        callback_data="delete_all_tests"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📅 Удалить по дате",
        callback_data="delete_by_date"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📊 Посмотреть все анализы",
        callback_data="view_all_tests"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Отмена",
        callback_data="cancel_manage"
    ))
    builder.adjust(2, 2, 2)
    
    logging.debug("Клавиатура управления анализами создана")
    return builder.as_markup()

# Функция для создания клавиатуры удаления анализов
def get_delete_test_keyboard(tests_data):
    logging.debug("Создание клавиатуры удаления анализов")
    
    builder = InlineKeyboardBuilder()
    
    # Добавляем кнопки для каждого анализа
    for i, test in enumerate(tests_data):
        test_id = test.get('id')
        test_name = test.get('test_name', 'Неизвестный анализ')
        test_date = test.get('test_date', '')
        
        # Формируем краткое название для кнопки
        short_name = test_name[:30] + "..." if len(test_name) > 30 else test_name
        button_text = f"🗑️ {short_name}"
        if test_date:
            button_text += f" ({test_date})"
        
        builder.add(types.InlineKeyboardButton(
            text=button_text,
            callback_data=f"delete_test_{test_id}"
        ))
    
    # Добавляем кнопку отмены
    builder.add(types.InlineKeyboardButton(
        text="❌ Отмена",
        callback_data="cancel_delete"
    ))
    
    # Располагаем кнопки по одной в строке
    builder.adjust(1)
    
    logging.debug("Клавиатура удаления анализов создана")
    return builder.as_markup()

# Функция для создания клавиатуры удаления медицинских записей
def get_delete_medical_record_keyboard(medical_records):
    logging.debug("Создание клавиатуры удаления медицинских записей")
    
    builder = InlineKeyboardBuilder()
    
    # Добавляем кнопки для каждой медицинской записи
    for i, record in enumerate(medical_records):
        record_id = record.get('id')
        content = record.get('content', '')
        created_at = record.get('created_at', '')[:10] if record.get('created_at') else 'Не указана'
        
        # Определяем тип записи
        if "не удалось извлечь" in content.lower() or len(content.strip()) < 100:
            record_type = "❌ Неудачный"
        else:
            record_type = "✅ Успешный"
        
        # Формируем краткое название для кнопки
        short_content = content[:25] + "..." if len(content) > 25 else content
        button_text = f"{record_type} {short_content}"
        button_text += f" ({created_at})"
        
        builder.add(types.InlineKeyboardButton(
            text=button_text,
            callback_data=f"delete_medical_record_{record_id}"
        ))
    
    # Добавляем кнопку отмены
    builder.add(types.InlineKeyboardButton(
        text="❌ Отмена",
        callback_data="cancel_delete"
    ))
    
    # Располагаем кнопки по одной в строке
    builder.adjust(1)
    
    logging.debug("Клавиатура удаления медицинских записей создана")
    return builder.as_markup()

# Функция для создания клавиатуры подтверждения удаления
def get_confirm_delete_keyboard(test_id: int, test_name: str):
    logging.debug(f"Создание клавиатуры подтверждения удаления для теста {test_id}")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, удалить",
        callback_data=f"confirm_delete_{test_id}"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Нет, отменить",
        callback_data="cancel_delete"
    ))
    builder.adjust(2)
    
    logging.debug("Клавиатура подтверждения удаления создана")
    return builder.as_markup()

# Функция для создания клавиатуры подтверждения удаления всех анализов
def get_confirm_delete_all_keyboard():
    logging.debug("Создание клавиатуры подтверждения удаления всех анализов")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, удалить все",
        callback_data="confirm_delete_all"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Нет, отменить",
        callback_data="cancel_delete"
    ))
    builder.adjust(2)
    
    logging.debug("Клавиатура подтверждения удаления всех анализов создана")
    return builder.as_markup()

# Функция для создания клавиатуры выбора периода удаления
def get_date_range_keyboard():
    logging.debug("Создание клавиатуры выбора периода удаления")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="📅 Удалить за сегодня",
        callback_data="delete_today"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📅 Удалить за неделю",
        callback_data="delete_week"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📅 Удалить за месяц",
        callback_data="delete_month"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📅 Удалить за год",
        callback_data="delete_year"
    ))
    builder.add(types.InlineKeyboardButton(
        text="📅 Удалить до определенной даты",
        callback_data="delete_before_date"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Отмена",
        callback_data="cancel_delete"
    ))
    builder.adjust(2, 2, 2)
    
    logging.debug("Клавиатура выбора периода удаления создана")
    return builder.as_markup()

# Функция для создания клавиатуры подтверждения удаления по периоду
def get_confirm_delete_period_keyboard(period: str):
    logging.debug(f"Создание клавиатуры подтверждения удаления за период: {period}")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, удалить",
        callback_data=f"confirm_period_{period}"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Нет, отменить",
        callback_data="cancel_delete"
    ))
    builder.adjust(2)
    
    logging.debug("Клавиатура подтверждения удаления по периоду создана")
    return builder.as_markup()

# Функция для создания клавиатуры подтверждения удаления медицинской записи
def get_confirm_delete_medical_record_keyboard(record_id: int, record_type: str):
    logging.debug(f"Создание клавиатуры подтверждения удаления медицинской записи {record_id}")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, удалить",
        callback_data=f"confirm_delete_medical_record_{record_id}"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Нет, отменить",
        callback_data="cancel_delete"
    ))
    builder.adjust(2)
    
    logging.debug("Клавиатура подтверждения удаления медицинской записи создана")
    return builder.as_markup()

# Функция для создания клавиатуры подтверждения удаления всех медицинских записей
def get_confirm_delete_all_medical_records_keyboard():
    logging.debug("Создание клавиатуры подтверждения удаления всех медицинских записей")
    
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="✅ Да, удалить все",
        callback_data="confirm_delete_all_medical_records"
    ))
    builder.add(types.InlineKeyboardButton(
        text="❌ Нет, отменить",
        callback_data="cancel_delete"
    ))
    builder.adjust(2)
    
    logging.debug("Клавиатура подтверждения удаления всех медицинских записей создана")
    return builder.as_markup()
