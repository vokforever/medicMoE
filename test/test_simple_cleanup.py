#!/usr/bin/env python3
"""
Упрощенный тестовый скрипт для очистки дубликатов медицинских записей
"""

import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация Supabase
supabase: Client = create_client(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)

def normalize_content_for_comparison(content: str) -> str:
    """
    Нормализует содержимое для более точного сравнения.
    Убирает лишние пробелы, приводит к нижнему регистру, убирает незначимые различия.
    """
    if not content:
        return ""
    
    # Приводим к нижнему регистру
    normalized = content.lower()
    
    # Убираем лишние пробелы и переносы строк
    normalized = ' '.join(normalized.split())
    
    # Убираем пунктуацию и специальные символы
    import re
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Убираем множественные пробелы
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def cleanup_duplicate_medical_records(user_id: str, record_type: str = "image_analysis") -> int:
    """
    Удаляет дублирующиеся записи для пользователя, оставляя только самую новую.
    Использует интеллектуальный анализ схожести для определения дубликатов.
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
                
                # Если точные критерии не сработали, используем анализ схожести как fallback
                similarity = analyze_content_similarity(current_content, processed_content)
                if similarity > 0.8:
                    duplicates_to_delete.append(record["id"])
                    logging.info(f"Найден дубликат по схожести: ID {record['id']} (схожесть: {similarity:.2%})")
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

def show_user_records(user_id: str):
    """
    Показывает все записи пользователя для анализа
    """
    try:
        logging.info(f"Показываю записи пользователя: {user_id}")
        
        response = supabase.table("doc_medical_records").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not response.data:
            logging.info("Записей не найдено")
            return
        
        logging.info(f"Найдено {len(response.data)} записей:")
        for i, record in enumerate(response.data):
            content_preview = record.get("content", "")[:100] + "..." if len(record.get("content", "")) > 100 else record.get("content", "")
            logging.info(f"  {i+1}. ID: {record.get('id')}, Тип: {record.get('record_type')}, Создано: {record.get('created_at')}")
            logging.info(f"     Содержимое: {content_preview}")
        
    except Exception as e:
        logging.error(f"Ошибка при получении записей пользователя: {e}")


def show_full_record_content(user_id: str, record_id: int):
    """
    Показывает полное содержимое конкретной записи для отладки
    """
    try:
        logging.info(f"Показываю полное содержимое записи {record_id} для пользователя {user_id}")
        
        response = supabase.table("doc_medical_records").select("*").eq("id", record_id).eq("user_id", user_id).execute()
        
        if not response.data:
            logging.info("Запись не найдена")
            return
        
        record = response.data[0]
        content = record.get("content", "")
        
        logging.info(f"=== ПОЛНОЕ СОДЕРЖИМОЕ ЗАПИСИ {record_id} ===")
        logging.info(f"Длина: {len(content)} символов")
        logging.info("Содержимое:")
        logging.info(content)
        logging.info("=== КОНЕЦ СОДЕРЖИМОГО ===")
        
        # Пробуем извлечь результаты анализов
        results = extract_analysis_results(content)
        logging.info(f"Извлеченные результаты анализов: {results}")
        
        # Пробуем извлечь дату анализа
        date = extract_analysis_date(content)
        logging.info(f"Извлеченная дата анализа: {date}")
        
    except Exception as e:
        logging.error(f"Ошибка при получении полного содержимого записи: {e}")

def analyze_content_similarity(content1: str, content2: str) -> float:
    """
    Анализирует схожесть содержимого двух записей с фокусом на медицинские данные
    """
    try:
        # Нормализуем содержимое
        norm1 = normalize_content_for_comparison(content1)
        norm2 = normalize_content_for_comparison(content2)
        
        # Простой анализ схожести по ключевым словам
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Вычисляем коэффициент Жаккара
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Анализируем ключевые медицинские термины и результаты
        medical_terms = ['anti-hev', 'anti-hcv', 'anti-hb', 'igg', 'igm', 'ige', 'отрицательно', 'положительно']
        medical_similarity = 0
        
        for term in medical_terms:
            if term in norm1.lower() and term in norm2.lower():
                medical_similarity += 1
        
        medical_similarity = medical_similarity / len(medical_terms)
        
        # Анализируем конкретные результаты анализов
        analysis_results_similarity = analyze_analysis_results_similarity(content1, content2)
        
        # Анализируем информацию о пациенте
        patient_similarity = analyze_patient_info_similarity(content1, content2)
        
        # Общая схожесть с весами
        total_similarity = (
            similarity * 0.2 +           # Общая текстовая схожесть
            medical_similarity * 0.3 +   # Медицинские термины
            analysis_results_similarity * 0.4 +  # Результаты анализов
            patient_similarity * 0.1     # Информация о пациенте
        )
        
        return total_similarity
        
    except Exception as e:
        logging.error(f"Ошибка при анализе схожести: {e}")
        return 0.0

def analyze_analysis_results_similarity(content1: str, content2: str) -> float:
    """
    Анализирует схожесть результатов анализов
    """
    try:
        # Ищем результаты анализов в тексте
        results1 = extract_analysis_results(content1)
        results2 = extract_analysis_results(content2)
        
        if not results1 or not results2:
            return 0.0
        
        # Сравниваем результаты
        matching_results = 0
        total_results = max(len(results1), len(results2))
        
        for test_name, result1 in results1.items():
            if test_name in results2:
                result2 = results2[test_name]
                if result1 == result2:
                    matching_results += 1
        
        return matching_results / total_results if total_results > 0 else 0.0
        
    except Exception as e:
        logging.error(f"Ошибка при анализе результатов анализов: {e}")
        return 0.0

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
                        
                        # Также сохраняем с оригинальным названием для отладки
                        results[f"debug_{test_name_lower}"] = f"original: {test_name}, result: {result}"
                        
    except Exception as e:
        logging.error(f"Ошибка при извлечении результатов: {e}")
    
    return results

def analyze_patient_info_similarity(content1: str, content2: str) -> float:
    """
    Анализирует схожесть информации о пациенте
    """
    try:
        # Извлекаем информацию о пациенте
        patient1 = extract_patient_info(content1)
        patient2 = extract_patient_info(content2)
        
        if not patient1 or not patient2:
            return 0.0
        
        # Сравниваем ключевые поля
        matches = 0
        total_fields = 0
        
        for field in ['name', 'birth_date', 'age']:
            if field in patient1 and field in patient2:
                total_fields += 1
                if patient1[field] == patient2[field]:
                    matches += 1
        
        return matches / total_fields if total_fields > 0 else 0.0
        
    except Exception as e:
        logging.error(f"Ошибка при анализе информации о пациенте: {e}")
        return 0.0

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


def main():
    """
    Основная функция для тестирования
    """
    logging.info("=== Тестирование функций очистки дубликатов ===")
    
    # Получаем первого пользователя для тестирования
    try:
        response = supabase.table("doc_patient_profiles").select("user_id").limit(1).execute()
        if not response.data:
            logging.error("Пользователи не найдены")
            return
        
        test_user_id = response.data[0]["user_id"]
        logging.info(f"Тестируем на пользователе: {test_user_id}")
        
        # Получаем все записи пользователя для анализа
        records_response = supabase.table("doc_medical_records").select("*").eq("user_id", test_user_id).order("created_at", desc=True).execute()
        
        # Показываем записи до очистки
        logging.info("\n--- Записи ДО очистки ---")
        show_user_records(test_user_id)
        
        # Показываем полное содержимое первой записи для отладки
        logging.info("\n--- Отладка: полное содержимое первой записи ---")
        if records_response.data:
            first_record_id = records_response.data[0].get('id')
            show_full_record_content(test_user_id, first_record_id)
        
        # Тестируем точные критерии дубликатов
        logging.info("\n--- Тестирование точных критериев дубликатов ---")
        test_exact_criteria_matching(test_user_id)
        
        # Анализируем схожесть между записями
        logging.info("\n--- Анализ схожести содержимого ---")
        if records_response.data and len(records_response.data) >= 2:
            for i in range(len(records_response.data) - 1):
                record1 = records_response.data[i]
                record2 = records_response.data[i + 1]
                
                similarity = analyze_content_similarity(
                    record1.get("content", ""),
                    record2.get("content", "")
                )
                
                logging.info(f"Схожесть между записями {record1.get('id')} и {record2.get('id')}: {similarity:.2%}")
                
                if similarity > 0.8:
                    logging.info("  ⚠️  Высокая схожесть - возможный дубликат!")
                elif similarity > 0.5:
                    logging.info("  ⚠️  Средняя схожесть - частичный дубликат")
                else:
                    logging.info("  ✅  Низкая схожесть - разные записи")
        
        # Очищаем дубликаты
        logging.info("\n--- Очистка дубликатов ---")
        deleted_count = cleanup_duplicate_medical_records(test_user_id)
        
        # Показываем записи после очистки
        logging.info("\n--- Записи ПОСЛЕ очистки ---")
        show_user_records(test_user_id)
        
        logging.info(f"\nРезультат: удалено {deleted_count} дублирующихся записей")
        
    except Exception as e:
        logging.error(f"Ошибка в основной функции: {e}")


def test_exact_criteria_matching(user_id: str):
    """
    Тестирует точные критерии для поиска дубликатов anti-HEV IgG
    """
    try:
        logging.info("Тестирование точных критериев дубликатов...")
        
        records_response = supabase.table("doc_medical_records").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not records_response.data or len(records_response.data) < 2:
            logging.info("Недостаточно записей для тестирования точных критериев")
            return
        
        records = records_response.data
        
        # Тестируем попарное сравнение
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                record1 = records[i]
                record2 = records[j]
                
                content1 = record1.get("content", "")
                content2 = record2.get("content", "")
                
                # Извлекаем результаты анализов
                results1 = extract_analysis_results(content1)
                results2 = extract_analysis_results(content2)
                
                # Извлекаем даты анализов
                date1 = extract_analysis_date(content1)
                date2 = extract_analysis_date(content2)
                
                logging.info(f"\nСравнение записей {record1.get('id')} и {record2.get('id')}:")
                logging.info(f"  Запись 1 - anti-HEV IgG: {results1.get('anti-hev igg', 'не найден')}, дата: {date1}")
                logging.info(f"  Запись 2 - anti-HEV IgG: {results2.get('anti-hev igg', 'не найден')}, дата: {date2}")
                
                # Проверяем по точным критериям
                is_duplicate = is_exact_duplicate_by_criteria(content1, content2)
                
                if is_duplicate:
                    logging.info("  ✅ ОБНАРУЖЕН ДУБЛИКАТ по точным критериям!")
                else:
                    logging.info("  ❌ Дубликат не обнаружен")
                    
                    # Показываем детали анализа
                    if 'anti-hev igg' in results1 and 'anti-hev igg' in results2:
                        if results1['anti-hev igg'] == results2['anti-hev igg']:
                            if date1 and date2 and date1 == date2:
                                logging.info("    ⚠️  Тест и результат совпадают, даты совпадают - должен быть дубликат!")
                            elif date1 and date2 and date1 != date2:
                                logging.info("    ℹ️  Тест и результат совпадают, но даты разные - не дубликат")
                            else:
                                logging.info("    ℹ️  Тест и результат совпадают, но дата не извлечена")
                        else:
                            logging.info("    ℹ️  Тест найден, но результаты разные")
                    else:
                        logging.info("    ℹ️  Тест anti-HEV IgG не найден в одной или обеих записях")
        
    except Exception as e:
        logging.error(f"Ошибка при тестировании точных критериев: {e}")


if __name__ == "__main__":
    main()
