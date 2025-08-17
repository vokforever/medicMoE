import logging
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import supabase

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

# Функция для сохранения медицинских записей
async def save_medical_record(user_id: str, record_type: str, content: str, source: str = "") -> bool:
    try:
        logging.info(f"Сохранение медицинской записи для пользователя: {user_id}")
        logging.info(f"Тип записи: {record_type}, источник: {source}")
        logging.info(f"Длина содержимого: {len(content)} символов")
        
        # Проверяем на дублирование с помощью улучшенной ИИ-проверки перед сохранением
        from utils import check_duplicate_medical_record_ai_enhanced
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
        from utils import save_to_vector_knowledge_base
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
