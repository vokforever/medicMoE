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
                
                # Ищем строки с результатами анализов (расширяем поиск)
                if any(keyword in line.lower() for keyword in [
                    'anti-', 'igg', 'igm', 'ige', 'гепатит', 'аллергия', 'opisthorchis', 
                    'toxocara', 'lamblia', 'ascaris', 'hepatitis', 'ferritin', 'tsh',
                    'церулоплазмин', 'с-реактивный белок', 'c-реактивный белок'
                ]):
                    test_data = self._extract_test_from_line_improved(line, lines, i)
                    if test_data:
                        tests.append(test_data)
                        logging.info(f"Извлечен тест: {test_data['test_name']} = {test_data['result']}")
                    else:
                        # Если не удалось извлечь, попробуем простой метод
                        test_data = self._extract_test_from_line(line)
                        if test_data:
                            tests.append(test_data)
                            logging.info(f"Извлечен тест простым методом: {test_data['test_name']} = {test_data['result']}")
            
            logging.info(f"Всего извлечено тестов: {len(tests)}")
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
                    
                    # Очищаем результат с улучшенной логикой
                    clean_result = self._clean_result_enhanced(result, all_lines, line_index)
                    
                    # Ищем дополнительную информацию в соседних строках
                    test_system = self._find_test_system(all_lines, line_index)
                    equipment = self._find_equipment(all_lines, line_index)
                    
                    # Если не нашли тест-систему или оборудование, ищем в контексте
                    if not test_system or test_system == "**" or test_system == "*":
                        real_test_system = self._extract_real_value_from_context(all_lines, line_index, "test_system")
                        if real_test_system:
                            test_system = real_test_system
                    
                    if not equipment or equipment == "**" or equipment == "*":
                        real_equipment = self._extract_real_value_from_context(all_lines, line_index, "equipment")
                        if real_equipment:
                            equipment = real_equipment
                    
                    # Проверяем, что результат не пустой и не содержит только звездочки
                    if test_name and clean_result and clean_result != "Не указан":
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
                    
                    # Очищаем результат с улучшенной логикой (используем простую версию для обратной совместимости)
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
        if not result:
            return "Не указан"
        
        # Убираем звездочки
        clean_result = result.replace('**', '').replace('*', '').strip()
        
        # Убираем единицы измерения и референсные значения
        clean_result = re.sub(r'\s*\d+\s*(МЕ/мл|мл|мг/л|ммоль/л|г/л|%)', '', clean_result)
        clean_result = re.sub(r'\s*норма[:\s]*[^,\n]+', '', clean_result)
        clean_result = re.sub(r'\s*референс[:\s]*[^,\n]+', '', clean_result)
        
        # Очищаем от лишних пробелов
        clean_result = re.sub(r'\s+', ' ', clean_result).strip()
        
        # Если результат пустой после очистки, возвращаем "Не указан"
        if not clean_result:
            clean_result = "Не указан"
        
        return clean_result
    
    def _clean_result_enhanced(self, result: str, all_lines: List[str], line_index: int) -> str:
        """Улучшенная очистка результата с поиском в контексте"""
        if not result:
            return "Не указан"
        
        # Сначала пробуем обычную очистку
        clean_result = self._clean_result(result)
        
        # Если результат содержит только звездочки или пустой, ищем в контексте
        if clean_result == "Не указан" and ("**" in result or "*" in result):
            # Ищем реальное значение в соседних строках
            real_result = self._extract_real_value_from_context(all_lines, line_index, "result")
            if real_result:
                # Очищаем найденное значение
                clean_result = self._clean_result(real_result)
                logging.info(f"Найдено реальное значение в контексте: {real_result} -> {clean_result}")
        
        # Если все еще не указан, ищем по ключевым словам в контексте
        if clean_result == "Не указан":
            context_result = self._search_result_in_context(all_lines, line_index)
            if context_result:
                clean_result = context_result
                logging.info(f"Найдено значение по ключевым словам: {clean_result}")
        
        return clean_result
    
    def _search_result_in_context(self, all_lines: List[str], line_index: int) -> Optional[str]:
        """Ищет результат анализа по ключевым словам в контексте"""
        try:
            search_range = 10  # Увеличиваем диапазон поиска
            
            for i in range(max(0, line_index - search_range), min(len(all_lines), line_index + search_range + 1)):
                line = all_lines[i].strip()
                if not line:
                    continue
                
                # Ищем строки с результатами анализов
                if any(keyword in line.lower() for keyword in [
                    'отрицательно', 'положительно', 'negative', 'positive',
                    'норма', 'норме', 'в норме', 'в пределах нормы',
                    'повышен', 'понижен', 'высокий', 'низкий',
                    'нормальный', 'патологический', 'патология'
                ]):
                    # Если строка содержит двоеточие, извлекаем значение
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            value = parts[1].strip()
                            if value and value != "**" and value != "*":
                                # Очищаем найденное значение
                                clean_value = self._clean_result(value)
                                if clean_value != "Не указан":
                                    return clean_value
                    else:
                        # Если нет двоеточия, проверяем всю строку
                        if line and line != "**" and line != "*":
                            # Проверяем, что это не название теста
                            if not any(test_keyword in line.lower() for test_keyword in [
                                'anti-', 'igg', 'igm', 'ige', 'гепатит', 'аллергия',
                                'тест-система', 'оборудование', 'abbott', 'roche'
                            ]):
                                clean_value = self._clean_result(line)
                                if clean_value != "Не указан":
                                    return clean_value
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка при поиске результата в контексте: {e}")
            return None
    
    def _extract_real_value_from_context(self, all_lines: List[str], line_index: int, field_name: str) -> Optional[str]:
        """Извлекает реальное значение поля из контекста строк"""
        try:
            # Ищем в текущей и соседних строках (расширяем поиск)
            search_range = 5  # Увеличиваем диапазон поиска
            
            for i in range(max(0, line_index - search_range), min(len(all_lines), line_index + search_range + 1)):
                line = all_lines[i].strip()
                if not line:
                    continue
                
                # Ищем конкретные поля
                if field_name == "test_system":
                    if any(keyword in line.lower() for keyword in ['тест-система', 'test-system', 'abbott', 'roche', 'cobas']):
                        if ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                value = parts[1].strip()
                                if value and value != "**" and value != "*":
                                    return value
                        else:
                            # Если нет двоеточия, берем всю строку
                            if line and line != "**" and line != "*":
                                return line
                
                elif field_name == "equipment":
                    if any(keyword in line.lower() for keyword in ['оборудование', 'equipment', 'alinity', 'cobas']):
                        if ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                value = parts[1].strip()
                                if value and value != "**" and value != "*":
                                    return value
                        else:
                            if line and line != "**" and line != "*":
                                return line
                
                elif field_name == "result":
                    # Ищем результат в соседних строках
                    if any(keyword in line.lower() for keyword in ['отрицательно', 'положительно', 'negative', 'positive', 'норма', 'норме']):
                        if ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                value = parts[1].strip()
                                if value and value != "**" and value != "*":
                                    return value
                        else:
                            if line and line != "**" and line != "*":
                                return line
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении реального значения для {field_name}: {e}")
            return None
    
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

    async def cleanup_existing_test_results(self, user_id: str) -> Dict[str, Any]:
        """
        Очищает существующие результаты анализов от лишних символов и форматирования
        """
        try:
            logging.info(f"Начинаю очистку результатов анализов для пользователя: {user_id}")
            
            # Получаем все структурированные тесты пользователя
            tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).execute()
            
            if not tests.data:
                logging.info("Нет анализов для очистки")
                return {"success": True, "message": "Нет анализов для очистки", "cleaned_count": 0}
            
            cleaned_count = 0
            updated_tests = []
            
            for test in tests.data:
                test_id = test.get("id")
                test_name = test.get("test_name", "")
                result = test.get("result", "")
                test_system = test.get("test_system", "")
                equipment = test.get("equipment", "")
                
                # Проверяем, нужна ли очистка
                needs_cleaning = False
                cleaned_result = result
                cleaned_test_system = test_system
                cleaned_equipment = equipment
                
                # Очищаем результат
                if result and ("**" in result or "*" in result):
                    cleaned_result = self._clean_result(result)
                    if cleaned_result != result:
                        needs_cleaning = True
                
                # Очищаем тест-систему
                if test_system and ("**" in test_system or "*" in test_system):
                    cleaned_test_system = self._clean_result(test_system)
                    if cleaned_test_system != test_system:
                        needs_cleaning = True
                
                # Очищаем оборудование
                if equipment and ("**" in equipment or "*" in equipment):
                    cleaned_equipment = self._clean_result(equipment)
                    if cleaned_equipment != equipment:
                        needs_cleaning = True
                
                # Если нужна очистка, обновляем запись
                if needs_cleaning:
                    try:
                        update_data = {
                            "result": cleaned_result,
                            "test_system": cleaned_test_system,
                            "equipment": cleaned_equipment,
                            "updated_at": datetime.now().isoformat()
                        }
                        
                        # Обновляем запись в базе
                        self.supabase.table("doc_structured_test_results").update(update_data).eq(
                            "id", test_id).execute()
                        
                        cleaned_count += 1
                        updated_tests.append({
                            "id": test_id,
                            "test_name": test_name,
                            "old_result": result,
                            "new_result": cleaned_result,
                            "old_test_system": test_system,
                            "new_test_system": cleaned_test_system,
                            "old_equipment": equipment,
                            "new_equipment": cleaned_equipment
                        })
                        
                        logging.info(f"Очищен анализ {test_id}: {test_name}")
                        
                    except Exception as e:
                        logging.error(f"Ошибка при очистке анализа {test_id}: {e}")
            
            result = {
                "success": True,
                "message": f"Очистка завершена. Очищено {cleaned_count} анализов",
                "cleaned_count": cleaned_count,
                "updated_tests": updated_tests
            }
            
            logging.info(f"Очистка завершена: {cleaned_count} анализов очищено")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при очистке результатов анализов: {e}")
            return {"success": False, "message": str(e), "cleaned_count": 0}

    async def reprocess_medical_records(self, user_id: str) -> Dict[str, Any]:
        """
        Переобрабатывает медицинские записи для улучшения структурированных данных
        """
        try:
            logging.info(f"Начинаю переобработку медицинских записей для пользователя: {user_id}")
            
            # Получаем все медицинские записи пользователя
            medical_records = self._get_medical_records(user_id)
            
            if not medical_records:
                logging.info("Нет медицинских записей для переобработки")
                return {"success": False, "message": "Нет медицинских записей для переобработки"}
            
            # Удаляем старые структурированные данные
            old_tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).execute()
            
            if old_tests.data:
                for test in old_tests.data:
                    self.supabase.table("doc_structured_test_results").delete().eq("id", test.get("id")).execute()
                
                logging.info(f"Удалено {len(old_tests.data)} старых записей анализов")
            
            # Переобрабатываем записи с улучшенной логикой
            result = await self.extract_and_structure_tests(user_id)
            
            if result.get("success"):
                result["message"] = f"Переобработка завершена. Извлечено {result.get('tests_count', 0)} анализов"
                result["reprocessed"] = True
            
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при переобработке медицинских записей: {e}")
            return {"success": False, "message": str(e)}


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

    async def cleanup_existing_test_results(self, user_id: str) -> Dict[str, Any]:
        """
        Очищает существующие результаты анализов от лишних символов и форматирования
        """
        try:
            logging.info(f"Начинаю очистку результатов анализов для пользователя: {user_id}")
            
            # Получаем все структурированные тесты пользователя
            tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).execute()
            
            if not tests.data:
                logging.info("Нет анализов для очистки")
                return {"success": True, "message": "Нет анализов для очистки", "cleaned_count": 0}
            
            cleaned_count = 0
            updated_tests = []
            
            for test in tests.data:
                test_id = test.get("id")
                test_name = test.get("test_name", "")
                result = test.get("result", "")
                test_system = test.get("test_system", "")
                equipment = test.get("equipment", "")
                
                # Проверяем, нужна ли очистка
                needs_cleaning = False
                cleaned_result = result
                cleaned_test_system = test_system
                cleaned_equipment = equipment
                
                # Очищаем результат
                if result and ("**" in result or "*" in result):
                    cleaned_result = self._clean_result(result)
                    if cleaned_result != result:
                        needs_cleaning = True
                
                # Очищаем тест-систему
                if test_system and ("**" in test_system or "*" in test_system):
                    cleaned_test_system = self._clean_result(test_system)
                    if cleaned_test_system != test_system:
                        needs_cleaning = True
                
                # Очищаем оборудование
                if equipment and ("**" in equipment or "*" in equipment):
                    cleaned_equipment = self._clean_result(equipment)
                    if cleaned_equipment != equipment:
                        needs_cleaning = True
                
                # Если нужна очистка, обновляем запись
                if needs_cleaning:
                    try:
                        update_data = {
                            "result": cleaned_result,
                            "test_system": cleaned_test_system,
                            "equipment": cleaned_equipment,
                            "updated_at": datetime.now().isoformat()
                        }
                        
                        # Обновляем запись в базе
                        self.supabase.table("doc_structured_test_results").update(update_data).eq(
                            "id", test_id).execute()
                        
                        cleaned_count += 1
                        updated_tests.append({
                            "id": test_id,
                            "test_name": test_name,
                            "old_result": result,
                            "new_result": cleaned_result,
                            "old_test_system": test_system,
                            "new_test_system": cleaned_test_system,
                            "old_equipment": equipment,
                            "new_equipment": cleaned_equipment
                        })
                        
                        logging.info(f"Очищен анализ {test_id}: {test_name}")
                        
                    except Exception as e:
                        logging.error(f"Ошибка при очистке анализа {test_id}: {e}")
            
            result = {
                "success": True,
                "message": f"Очистка завершена. Очищено {cleaned_count} анализов",
                "cleaned_count": cleaned_count,
                "updated_tests": updated_tests
            }
            
            logging.info(f"Очистка завершена: {cleaned_count} анализов очищено")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при очистке результатов анализов: {e}")
            return {"success": False, "message": str(e), "cleaned_count": 0}
    
    async def reprocess_medical_records(self, user_id: str) -> Dict[str, Any]:
        """
        Переобрабатывает медицинские записи для улучшения структурированных данных
        """
        try:
            logging.info(f"Начинаю переобработку медицинских записей для пользователя: {user_id}")
            
            # Получаем все медицинские записи пользователя
            medical_records = self._get_medical_records(user_id)
            
            if not medical_records:
                logging.info("Нет медицинских записей для переобработки")
                return {"success": False, "message": "Нет медицинских записей для переобработки"}
            
            # Удаляем старые структурированные данные
            old_tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).execute()
            
            if old_tests.data:
                for test in old_tests.data:
                    self.supabase.table("doc_structured_test_results").delete().eq("id", test.get("id")).execute()
                
                logging.info(f"Удалено {len(old_tests.data)} старых записей анализов")
            
            # Переобрабатываем записи с улучшенной логикой
            result = await self.extract_and_structure_tests(user_id)
            
            if result.get("success"):
                result["message"] = f"Переобработка завершена. Извлечено {result.get('tests_count', 0)} анализов"
                result["reprocessed"] = True
            
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при переобработке медицинских записей: {e}")
            return {"success": False, "message": str(e)}
