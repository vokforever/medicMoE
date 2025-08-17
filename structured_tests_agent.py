"""
Модуль для работы со структурированными данными анализов
Обеспечивает извлечение, структурирование и управление данными анализов
"""

import logging
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from supabase import Client

class TestExtractionAgent:
    """Агент для извлечения и структурирования данных анализов"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.max_retries = 3
        
    async def extract_and_structure_tests(self, user_id: str) -> Dict[str, Any]:
        """
        Извлекает и структурирует данные анализов из текстовых записей
        """
        try:
            logging.info(f"Извлечение структурированных данных для пользователя: {user_id}")
            
            # 1. Получаем все медицинские записи пользователя
            medical_records = self._get_medical_records(user_id)
            
            if not medical_records:
                logging.info("Нет медицинских записей для обработки")
                return {"success": False, "message": "Нет медицинских записей для обработки"}
            
            logging.info(f"Найдено {len(medical_records)} медицинских записей")
            
            # 2. Извлекаем структурированные данные из каждой записи
            all_tests = []
            for record in medical_records:
                tests = await self._extract_tests_from_text(record['content'], record['id'])
                all_tests.extend(tests)
            
            logging.info(f"Извлечено {len(all_tests)} анализов")
            
            # 3. Сохраняем структурированные данные
            saved_count = await self._save_structured_tests(user_id, all_tests)
            
            # 4. Определяем, какие данные отсутствуют
            missing_data = await self._identify_missing_data(user_id)
            
            result = {
                "success": True,
                "tests_count": len(all_tests),
                "saved_count": saved_count,
                "missing_data": missing_data,
                "tests": all_tests
            }
            
            logging.info(f"Извлечение завершено: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении структурированных данных: {e}")
            return {"success": False, "message": str(e)}
    
    def _get_medical_records(self, user_id: str) -> List[Dict[str, Any]]:
        """Получение медицинских записей пользователя"""
        try:
            response = self.supabase.table("doc_medical_records").select("*").eq("user_id", user_id).execute()
            return response.data if response.data else []
        except Exception as e:
            logging.error(f"Ошибка при получении медицинских записей: {e}")
            return []
    
    async def _extract_tests_from_text(self, text: str, record_id: int) -> List[Dict[str, Any]]:
        """
        Извлекает данные анализов из текста с помощью ИИ
        """
        try:
            logging.info(f"Извлечение анализов из текста длиной {len(text)} символов")
            
            # Используем улучшенный парсинг для извлечения данных
            tests_data = self._parse_tests_improved(text)
            
            # Если улучшенный парсинг не дал результатов, используем ИИ
            if not tests_data:
                logging.info("Улучшенный парсинг не дал результатов, использую ИИ")
                tests_data = await self._parse_tests_with_ai(text)
            
            # Добавляем ID исходной записи
            for test in tests_data:
                test['source_record_id'] = record_id
            
            logging.info(f"Извлечено {len(tests_data)} анализов")
            return tests_data
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении анализов из текста: {e}")
            return []
    
    def _parse_tests_improved(self, text: str) -> List[Dict[str, Any]]:
        """Улучшенный парсинг анализов с извлечением всех данных"""
        try:
            tests = []
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Ищем строки с результатами анализов
                if any(keyword in line.lower() for keyword in ['anti-', 'igg', 'igm', 'ige', 'гепатит', 'аллергия', 'opisthorchis', 'toxocara', 'lamblia', 'ascaris']):
                    test_data = self._extract_test_from_line_improved(line, lines, i)
                    if test_data:
                        tests.append(test_data)
            
            return tests
            
        except Exception as e:
            logging.error(f"Ошибка при улучшенном парсинге: {e}")
            return []
    
    def _parse_tests_simple(self, text: str) -> List[Dict[str, Any]]:
        """Простой парсинг анализов по ключевым словам (для обратной совместимости)"""
        try:
            tests = []
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Ищем строки с результатами анализов
                if any(keyword in line.lower() for keyword in ['anti-', 'igg', 'igm', 'ige', 'гепатит', 'аллергия']):
                    test_data = self._extract_test_from_line(line)
                    if test_data:
                        tests.append(test_data)
            
            return tests
            
        except Exception as e:
            logging.error(f"Ошибка при простом парсинге: {e}")
            return []
    
    def _extract_test_from_line_improved(self, line: str, all_lines: List[str], line_index: int) -> Optional[Dict[str, Any]]:
        """Улучшенное извлечение данных анализа из строки с контекстом"""
        try:
            # Ищем паттерны типа "Anti-HCV total (анти-HCV): ОТРИЦАТЕЛЬНО"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    test_name = parts[0].strip()
                    result = parts[1].strip()
                    
                    # Очищаем название теста
                    test_name = re.sub(r'^\d+\.\s*', '', test_name)  # Убираем номера
                    test_name = re.sub(r'[**]', '', test_name).strip()  # Убираем звездочки
                    
                    # Определяем единицы измерения
                    units = self._extract_units(result)
                    
                    # Определяем референсные значения
                    reference_values = self._extract_reference_values(result)
                    
                    # Очищаем результат
                    clean_result = self._clean_result(result)
                    
                    # Ищем дополнительную информацию в соседних строках
                    test_system = self._find_test_system(all_lines, line_index)
                    equipment = self._find_equipment(all_lines, line_index)
                    
                    if test_name and clean_result:
                        return {
                            "test_name": test_name,
                            "result": clean_result,
                            "reference_values": reference_values,
                            "units": units,
                            "test_system": test_system,
                            "equipment": equipment,
                            "notes": None
                        }
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка при улучшенном извлечении теста из строки: {e}")
            return None
    
    def _extract_test_from_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Извлекает данные анализа из одной строки (для обратной совместимости)"""
        try:
            # Ищем паттерны типа "Anti-HCV total (анти-HCV): ОТРИЦАТЕЛЬНО"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    test_name = parts[0].strip()
                    result = parts[1].strip()
                    
                    # Очищаем название теста
                    test_name = re.sub(r'^\d+\.\s*', '', test_name)  # Убираем номера
                    test_name = re.sub(r'[**]', '', test_name).strip()  # Убираем звездочки
                    
                    # Определяем единицы измерения
                    units = self._extract_units(result)
                    
                    # Определяем референсные значения
                    reference_values = self._extract_reference_values(result)
                    
                    # Очищаем результат
                    clean_result = self._clean_result(result)
                    
                    if test_name and clean_result:
                        return {
                            "test_name": test_name,
                            "result": clean_result,
                            "reference_values": reference_values,
                            "units": units,
                            "test_system": None,
                            "equipment": None,
                            "notes": None
                        }
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении теста из строки: {e}")
            return None
    
    def _extract_units(self, result: str) -> Optional[str]:
        """Извлекает единицы измерения из результата"""
        units_patterns = [
            r'(\d+)\s*(МЕ/мл|мл|мг/л|ммоль/л|г/л|%)',
            r'(\d+)\s*(МЕ|мл|мг|ммоль|г)',
        ]
        
        for pattern in units_patterns:
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                return match.group(2) if len(match.groups()) > 1 else match.group(1)
        
        return None
    
    def _extract_reference_values(self, result: str) -> Optional[str]:
        """Извлекает референсные значения"""
        ref_patterns = [
            r'норма[:\s]*([^,\n]+)',
            r'референс[:\s]*([^,\n]+)',
            r'<([^,\n]+)',
            r'>([^,\n]+)',
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _clean_result(self, result: str) -> str:
        """Очищает результат от лишней информации"""
        # Убираем единицы измерения и референсные значения
        clean_result = re.sub(r'\s*\d+\s*(МЕ/мл|мл|мг/л|ммоль/л|г/л|%)', '', result)
        clean_result = re.sub(r'\s*норма[:\s]*[^,\n]+', '', clean_result)
        clean_result = re.sub(r'\s*референс[:\s]*[^,\n]+', '', clean_result)
        
        # Очищаем от лишних пробелов
        clean_result = re.sub(r'\s+', ' ', clean_result).strip()
        
        return clean_result
    
    def _find_test_system(self, all_lines: List[str], current_line_index: int) -> Optional[str]:
        """Ищет тест-систему в соседних строках"""
        try:
            # Ищем в текущей и соседних строках
            for i in range(max(0, current_line_index - 2), min(len(all_lines), current_line_index + 3)):
                line = all_lines[i].strip().lower()
                if 'тест-система' in line or 'test-system' in line:
                    # Извлекаем название тест-системы
                    if ':' in all_lines[i]:
                        parts = all_lines[i].split(':', 1)
                        if len(parts) == 2:
                            return parts[1].strip()
            return None
        except Exception as e:
            logging.error(f"Ошибка при поиске тест-системы: {e}")
            return None
    
    def _find_equipment(self, all_lines: List[str], current_line_index: int) -> Optional[str]:
        """Ищет оборудование в соседних строках"""
        try:
            # Ищем в текущей и соседних строках
            for i in range(max(0, current_line_index - 2), min(len(all_lines), current_line_index + 3)):
                line = all_lines[i].strip().lower()
                if 'оборудование' in line or 'equipment' in line:
                    # Извлекаем название оборудования
                    if ':' in all_lines[i]:
                        parts = all_lines[i].split(':', 1)
                        if len(parts) == 2:
                            return parts[1].strip()
            return None
        except Exception as e:
            logging.error(f"Ошибка при поиске оборудования: {e}")
            return None
    
    async def _parse_tests_with_ai(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение анализов с помощью ИИ"""
        try:
            # Здесь будет вызов ИИ для извлечения данных
            # Пока возвращаем пустой список
            return []
            
        except Exception as e:
            logging.error(f"Ошибка при ИИ-извлечении: {e}")
            return []
    
    async def _save_structured_tests(self, user_id: str, tests: List[Dict[str, Any]]) -> int:
        """
        Сохраняет структурированные данные анализов в базу
        """
        try:
            saved_count = 0
            
            for test in tests:
                try:
                    # Проверяем, есть ли уже такой анализ
                    existing = self.supabase.table("doc_structured_test_results").select("*").eq(
                        "user_id", user_id).eq("test_name", test.get("test_name")).execute()
                    
                    if existing.data:
                        # Обновляем существующую запись
                        self.supabase.table("doc_structured_test_results").update({
                            "result": test.get("result"),
                            "reference_values": test.get("reference_values"),
                            "units": test.get("units"),
                            "test_system": test.get("test_system"),
                            "equipment": test.get("equipment"),
                            "notes": test.get("notes"),
                            "source_record_id": test.get("source_record_id"),
                            "updated_at": datetime.now().isoformat()
                        }).eq("id", existing.data[0]["id"]).execute()
                        
                        logging.info(f"Обновлен анализ: {test.get('test_name')}")
                    else:
                        # Создаем новую запись
                        self.supabase.table("doc_structured_test_results").insert({
                            "user_id": user_id,
                            "test_name": test.get("test_name"),
                            "result": test.get("result"),
                            "reference_values": test.get("reference_values"),
                            "units": test.get("units"),
                            "test_system": test.get("test_system"),
                            "equipment": test.get("equipment"),
                            "notes": test.get("notes"),
                            "source_record_id": test.get("source_record_id")
                        }).execute()
                        
                        logging.info(f"Создан новый анализ: {test.get('test_name')}")
                    
                    saved_count += 1
                    
                except Exception as e:
                    logging.error(f"Ошибка при сохранении анализа {test.get('test_name')}: {e}")
                    continue
            
            logging.info(f"Сохранено {saved_count} анализов")
            return saved_count
            
        except Exception as e:
            logging.error(f"Ошибка при сохранении структурированных данных: {e}")
            return 0
    
    async def _identify_missing_data(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Определяет, какие данные отсутствуют в структурированных записях
        """
        try:
            missing_data = []
            
            # Получаем все структурированные тесты
            tests = self.supabase.table("doc_structured_test_results").select("*").eq("user_id", user_id).execute()
            
            for test in tests.data:
                missing_fields = []
                
                if not test.get("test_date"):
                    missing_fields.append("дата сдачи анализа")
                
                if not test.get("reference_values"):
                    missing_fields.append("референсные значения")
                
                if missing_fields:
                    missing_data.append({
                        "test_id": test["id"],
                        "test_name": test["test_name"],
                        "missing_fields": missing_fields
                    })
            
            logging.info(f"Найдено {len(missing_data)} тестов с недостающими данными")
            return missing_data
            
        except Exception as e:
            logging.error(f"Ошибка при определении недостающих данных: {e}")
            return []


class StructuredTestAgent:
    """Агент для работы со структурированными данными анализов"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.extraction_agent = TestExtractionAgent(supabase_client)
        
    async def get_test_results_table(self, user_id: str) -> str:
        """
        Возвращает отформатированную таблицу с результатами анализов
        """
        try:
            logging.info(f"Формирование таблицы анализов для пользователя: {user_id}")
            
            # Сначала обновляем структурированные данные
            await self.extraction_agent.extract_and_structure_tests(user_id)
            
            # Получаем все структурированные тесты
            tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).order("test_name").execute()
            
            if not tests.data:
                return "У вас нет сохраненных результатов анализов."
            
            logging.info(f"Найдено {len(tests.data)} тестов для таблицы")
            
            # Формируем таблицу
            table = "📊 **Ваши анализы:**\n\n"
            table += "| Анализ | Результат | Референсные значения | Единицы | Дата |\n"
            table += "|--------|----------|---------------------|---------|------|\n"
            
            for test in tests.data:
                test_name = test.get("test_name", "").replace("|", "\\|")
                result = test.get("result", "").replace("|", "\\|")
                ref_values = test.get("reference_values", "").replace("|", "\\|")
                units = test.get("units", "").replace("|", "\\|")
                test_date = test.get("test_date", "")
                
                if test_date:
                    try:
                        test_date = datetime.strptime(test_date, "%Y-%m-%d").strftime("%d.%m.%Y")
                    except:
                        test_date = str(test_date)
                else:
                    test_date = "Не указана"
                
                table += f"| {test_name} | {result} | {ref_values} | {units} | {test_date} |\n"
            
            logging.info("Таблица анализов сформирована")
            return table
            
        except Exception as e:
            logging.error(f"Ошибка при формировании таблицы анализов: {e}")
            return "Не удалось получить результаты анализов."
    
    async def get_specific_test_result(self, user_id: str, test_name: str) -> Dict[str, Any]:
        """
        Возвращает результат конкретного анализа
        """
        try:
            logging.info(f"Поиск анализа '{test_name}' для пользователя: {user_id}")
            
            # Сначала обновляем структурированные данные
            await self.extraction_agent.extract_and_structure_tests(user_id)
            
            # Ищем анализ по названию
            tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).ilike("test_name", f"%{test_name}%").execute()
            
            if tests.data:
                logging.info(f"Найден анализ: {tests.data[0].get('test_name')}")
                return {
                    "found": True,
                    "test": tests.data[0]
                }
            else:
                logging.info(f"Анализ '{test_name}' не найден")
                return {
                    "found": False,
                    "message": f"Анализ '{test_name}' не найден в ваших результатах."
                }
                
        except Exception as e:
            logging.error(f"Ошибка при поиске анализа: {e}")
            return {
                "found": False,
                "message": "Произошла ошибка при поиске анализа."
            }
    
    async def request_missing_data(self, user_id: str, test_id: int) -> str:
        """
        Формирует запрос на добавление недостающих данных
        """
        try:
            # Получаем информацию о тесте
            test = self.supabase.table("doc_structured_test_results").select("*").eq("id", test_id).execute()
            
            if not test.data:
                return "Анализ не найден."
            
            test_data = test.data[0]
            
            # Формируем сообщение с запросом данных
            message = f"📋 **Необходимо дополнить информацию об анализе:**\n\n"
            message += f"**Название анализа:** {test_data.get('test_name', '')}\n"
            message += f"**Результат:** {test_data.get('result', '')}\n\n"
            message += "Пожалуйста, укажите следующую информацию:\n"
            
            if not test_data.get("test_date"):
                message += "- Дата сдачи анализа (в формате ДД.ММ.ГГГГ)\n"
            
            if not test_data.get("reference_values"):
                message += "- Референсные значения (норма)\n"
            
            message += "\nОтправьте информацию в формате:\n"
            message += "```\n"
            message += f"Анализ: {test_data.get('test_name', '')}\n"
            if not test_data.get("test_date"):
                message += "Дата: ДД.ММ.ГГГГ\n"
            if not test_data.get("reference_values"):
                message += "Норма: [референсные значения]\n"
            message += "```"
            
            return message
            
        except Exception as e:
            logging.error(f"Ошибка при формировании запроса недостающих данных: {e}")
            return "Произошла ошибка при формировании запроса."
    
    async def update_test_data(self, user_id: str, test_id: int, update_data: Dict[str, Any]) -> bool:
        """
        Обновляет данные анализа
        """
        try:
            logging.info(f"Обновление данных анализа {test_id} для пользователя {user_id}")
            
            # Проверяем, что тест принадлежит пользователю
            test = self.supabase.table("doc_structured_test_results").select("*").eq(
                "id", test_id).eq("user_id", user_id).execute()
            
            if not test.data:
                logging.warning(f"Анализ {test_id} не найден или не принадлежит пользователю {user_id}")
                return False
            
            # Обрабатываем дату, если она передана
            if "test_date" in update_data and update_data["test_date"]:
                try:
                    # Пытаемся распарсить дату в разных форматах
                    date_str = update_data["test_date"]
                    for fmt in ('%d.%m.%Y', '%d/%m/%Y', '%d-%m-%Y'):
                        try:
                            parsed_date = datetime.strptime(date_str, fmt)
                            update_data['test_date'] = parsed_date.strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue
                except Exception as e:
                    logging.error(f"Ошибка при парсинге даты {update_data['test_date']}: {e}")
                    return False
            
            # Обновляем данные
            self.supabase.table("doc_structured_test_results").update({
                **update_data,
                "updated_at": datetime.now().isoformat()
            }).eq("id", test_id).execute()
            
            logging.info(f"Данные анализа {test_id} успешно обновлены")
            return True
            
        except Exception as e:
            logging.error(f"Ошибка при обновлении данных анализа: {e}")
            return False
    
    async def get_tests_summary(self, user_id: str) -> str:
        """
        Возвращает сводку по всем анализам пользователя
        """
        try:
            logging.info(f"Формирование сводки анализов для пользователя: {user_id}")
            
            # Получаем все структурированные тесты
            tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).order("test_name").execute()
            
            if not tests.data:
                return "У вас нет сохраненных результатов анализов."
            
            # Группируем анализы по типам
            test_groups = {}
            for test in tests.data:
                test_type = self._categorize_test(test.get("test_name", ""))
                if test_type not in test_groups:
                    test_groups[test_type] = []
                test_groups[test_type].append(test)
            
            # Формируем сводку
            summary = f"📊 **Сводка по анализам**\n\n"
            summary += f"Всего анализов: {len(tests.data)}\n\n"
            
            for test_type, type_tests in test_groups.items():
                summary += f"**{test_type}** ({len(type_tests)} анализов):\n"
                for test in type_tests:
                    result = test.get("result", "")
                    date = test.get("test_date", "")
                    if date:
                        try:
                            date = datetime.strptime(date, "%Y-%m-%d").strftime("%d.%m.%Y")
                        except:
                            date = str(date)
                    else:
                        date = "дата не указана"
                    
                    summary += f"• {test.get('test_name', '')}: {result} (от {date})\n"
                summary += "\n"
            
            logging.info("Сводка анализов сформирована")
            return summary
            
        except Exception as e:
            logging.error(f"Ошибка при формировании сводки анализов: {e}")
            return "Не удалось получить сводку анализов."
    
    def _categorize_test(self, test_name: str) -> str:
        """Категоризирует анализ по названию"""
        test_name_lower = test_name.lower()
        
        if any(keyword in test_name_lower for keyword in ['anti-', 'гепатит', 'hcv', 'hbv', 'hev']):
            return "Анализы на гепатиты"
        elif any(keyword in test_name_lower for keyword in ['ige', 'аллергия', 'аллерген']):
            return "Аллергологические анализы"
        elif any(keyword in test_name_lower for keyword in ['общий', 'гемоглобин', 'лейкоциты', 'эритроциты']):
            return "Общий анализ крови"
        elif any(keyword in test_name_lower for keyword in ['биохимия', 'глюкоза', 'холестерин', 'креатинин']):
            return "Биохимический анализ"
        else:
            return "Другие анализы"
