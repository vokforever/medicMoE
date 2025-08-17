import logging
import json
import re
import requests
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from dateutil.parser import parse
from config import MEDICAL_SOURCES, supabase
from models import call_model_with_failover

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

# Функция для экранирования Markdown
def escape_markdown(text: str) -> str:
    """Экранирует специальные символы для Markdown"""
    logging.debug(f"Экранирование Markdown для текста длиной {len(text)} символов")
    
    # Символы, которые нужно экранировать в Markdown
    escape_chars = r'\_*[]()~`>#+-=|{}.!'
    
    escaped = ''
    for char in text:
        if char in escape_chars:
            escaped += f'\\{char}'
        else:
            escaped += char
    
    logging.debug(f"Markdown экранирован, результат длиной {len(escaped)} символов")
    return escaped

# Функция для очистки результата анализа от звездочек и лишних символов
def clean_test_result(result: str) -> str:
    """Очищает результат анализа от звездочек и лишних символов"""
    logging.debug(f"Очистка результата анализа: {result}")
    
    if not result:
        return result
    
    # Убираем звездочки
    cleaned = result.replace('**', '').replace('*', '').strip()
    
    # Убираем лишние пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Если результат пустой после очистки, возвращаем "Не указан"
    if not cleaned:
        cleaned = "Не указан"
    
    logging.debug(f"Результат очищен: {cleaned}")
    return cleaned

# Функция для получения эмбеддинга
def get_embedding(text: str) -> List[float]:
    """Получение эмбеддинга текста с помощью Mistral AI"""
    try:
        logging.info(f"Получение эмбеддинга для текста длиной {len(text)} символов")
        
        import os
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
        from config import tavily_client
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

# Функция для поиска в интернете
async def search_web(query: str) -> str:
    try:
        from config import tavily_client
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

# Функция для извлечения текста из PDF
async def extract_text_from_pdf(file_path: str) -> str:
    try:
        logging.info(f"Извлечение текста из PDF: {file_path}")
        
        import aiohttp
        import PyPDF2
        import io
        
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
