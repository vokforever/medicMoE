"""
Enhanced Database Cleanup Module
Улучшенные функции для очистки и исправления данных в базе данных
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from supabase import Client
from enhanced_test_extractor import EnhancedTestExtractor

class EnhancedDatabaseCleanup:
    """Улучшенный модуль для очистки базы данных"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.extractor = EnhancedTestExtractor()
    
    async def cleanup_all_test_results(self, user_id: str) -> Dict[str, Any]:
        """
        Комплексная очистка всех результатов анализов пользователя
        """
        try:
            logging.info(f"Начинаю комплексную очистку результатов анализов для пользователя: {user_id}")
            
            cleanup_result = {
                "success": True,
                "cleaned_tests": 0,
                "fixed_tests": 0,
                "deleted_duplicates": 0,
                "details": []
            }
            
            # 1. Очистка от символов форматирования
            formatting_result = await self.cleanup_formatting_issues(user_id)
            cleanup_result["cleaned_tests"] = formatting_result.get("cleaned_count", 0)
            cleanup_result["details"].extend(formatting_result.get("details", []))
            
            # 2. Исправление некорректных данных
            fixing_result = await self.fix_incorrect_data(user_id)
            cleanup_result["fixed_tests"] = fixing_result.get("fixed_count", 0)
            cleanup_result["details"].extend(fixing_result.get("details", []))
            
            # 3. Удаление дубликатов
            deduplication_result = await self.remove_duplicates(user_id)
            cleanup_result["deleted_duplicates"] = deduplication_result.get("deleted_count", 0)
            cleanup_result["details"].extend(deduplication_result.get("details", []))
            
            # 4. Переобработка медицинских записей
            reprocessing_result = await self.reprocess_medical_records(user_id)
            cleanup_result["details"].extend(reprocessing_result.get("details", []))
            
            total_improvements = (
                cleanup_result["cleaned_tests"] + 
                cleanup_result["fixed_tests"] + 
                cleanup_result["deleted_duplicates"]
            )
            
            cleanup_result["message"] = (
                f"Комплексная очистка завершена!\n\n"
                f"🧹 Очищено от форматирования: {cleanup_result['cleaned_tests']}\n"
                f"🔧 Исправлено данных: {cleanup_result['fixed_tests']}\n"
                f"🗑️ Удалено дубликатов: {cleanup_result['deleted_duplicates']}\n"
                f"📊 Всего улучшено: {total_improvements} записей"
            )
            
            logging.info(f"Комплексная очистка завершена: {total_improvements} улучшений")
            return cleanup_result
            
        except Exception as e:
            logging.error(f"Ошибка при комплексной очистке: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Произошла ошибка при очистке: {str(e)}"
            }
    
    async def cleanup_formatting_issues(self, user_id: str) -> Dict[str, Any]:
        """
        Очищает результаты анализов от проблем форматирования
        """
        try:
            logging.info("Начинаю очистку от проблем форматирования")
            
            # Получаем все анализы пользователя
            tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).execute()
            
            if not tests.data:
                return {"cleaned_count": 0, "details": ["Нет анализов для очистки"]}
            
            cleaned_count = 0
            details = []
            
            for test in tests.data:
                test_id = test.get("id")
                test_name = test.get("test_name", "")
                
                needs_update = False
                update_data = {}
                
                # Проверяем и очищаем каждое поле
                for field in ["result", "reference_values", "units", "test_system", "equipment"]:
                    current_value = test.get(field, "")
                    if current_value and ("**" in current_value or "*" in current_value):
                        cleaned_value = self._clean_field_value(current_value)
                        if cleaned_value != current_value:
                            update_data[field] = cleaned_value
                            needs_update = True
                            details.append(f"Очищено поле '{field}' в анализе '{test_name}'")
                
                # Если есть что обновлять
                if needs_update:
                    update_data["updated_at"] = datetime.now().isoformat()
                    
                    self.supabase.table("doc_structured_test_results").update(
                        update_data
                    ).eq("id", test_id).execute()
                    
                    cleaned_count += 1
                    logging.info(f"Очищен анализ {test_id}: {test_name}")
            
            result = {
                "cleaned_count": cleaned_count,
                "details": details
            }
            
            logging.info(f"Очистка форматирования завершена: {cleaned_count} анализов")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при очистке форматирования: {e}")
            return {"cleaned_count": 0, "details": [f"Ошибка: {str(e)}"]}
    
    async def fix_incorrect_data(self, user_id: str) -> Dict[str, Any]:
        """
        Исправляет некорректные данные в анализах
        """
        try:
            logging.info("Начинаю исправление некорректных данных")
            
            # Получаем анализы с проблемными данными
            tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).execute()
            
            if not tests.data:
                return {"fixed_count": 0, "details": ["Нет анализов для исправления"]}
            
            fixed_count = 0
            details = []
            
            for test in tests.data:
                test_id = test.get("id")
                test_name = test.get("test_name", "")
                result = test.get("result", "")
                
                needs_fix = False
                update_data = {}
                
                # Исправляем пустые или некорректные результаты
                if not result or result in ["**", "*", "Не указан", "", "null"]:
                    # Пытаемся извлечь результат из медицинских записей
                    fixed_result = await self._extract_result_from_medical_records(
                        user_id, test_name, test.get("source_record_id")
                    )
                    
                    if fixed_result:
                        update_data["result"] = fixed_result
                        needs_fix = True
                        fixed_count += 1
                        details.append(f"Исправлен результат в анализе '{test_name}': '{fixed_result}'")
                
                # Исправляем некорректные даты
                test_date = test.get("test_date", "")
                if test_date and not self._is_valid_date(test_date):
                    fixed_date = self._extract_date_from_medical_records(
                        user_id, test.get("source_record_id")
                    )
                    if fixed_date:
                        update_data["test_date"] = fixed_date
                        needs_fix = True
                        details.append(f"Исправлена дата в анализе '{test_name}': '{fixed_date}'")
                
                # Если есть исправления
                if needs_fix:
                    update_data["updated_at"] = datetime.now().isoformat()
                    
                    self.supabase.table("doc_structured_test_results").update(
                        update_data
                    ).eq("id", test_id).execute()
                    
                    logging.info(f"Исправлен анализ {test_id}: {test_name}")
            
            result = {
                "fixed_count": fixed_count,
                "details": details
            }
            
            logging.info(f"Исправление данных завершено: {fixed_count} анализов")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при исправлении данных: {e}")
            return {"fixed_count": 0, "details": [f"Ошибка: {str(e)}"]}
    
    async def remove_duplicates(self, user_id: str) -> Dict[str, Any]:
        """
        Удаляет дублирующиеся анализы
        """
        try:
            logging.info("Начинаю удаление дубликатов")
            
            # Получаем все анализы пользователя
            tests = self.supabase.table("doc_structured_test_results").select("*").eq(
                "user_id", user_id).order("created_at", desc=True).execute()
            
            if not tests.data:
                return {"deleted_count": 0, "details": ["Нет анализов для проверки"]}
            
            deleted_count = 0
            details = []
            seen_tests = {}
            
            for test in tests.data:
                test_name = test.get("test_name", "").lower().strip()
                result = test.get("result", "").lower().strip()
                test_date = test.get("test_date", "")
                
                # Создаем уникальный ключ
                unique_key = f"{test_name}_{result}_{test_date}"
                
                if unique_key in seen_tests:
                    # Это дубликат - удаляем
                    self.supabase.table("doc_structured_test_results").delete().eq(
                        "id", test.get("id")
                    ).execute()
                    
                    deleted_count += 1
                    details.append(f"Удален дубликат анализа: '{test.get('test_name')}'")
                    logging.info(f"Удален дубликат анализа: {test.get('id')}")
                else:
                    seen_tests[unique_key] = test
            
            result = {
                "deleted_count": deleted_count,
                "details": details
            }
            
            logging.info(f"Удаление дубликатов завершено: {deleted_count} записей")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при удалении дубликатов: {e}")
            return {"deleted_count": 0, "details": [f"Ошибка: {str(e)}"]}
    
    async def reprocess_medical_records(self, user_id: str) -> Dict[str, Any]:
        """
        Переобрабатывает медицинские записи для извлечения пропущенных данных
        """
        try:
            logging.info("Начинаю переобработку медицинских записей")
            
            # Получаем медицинские записи
            records = self.supabase.table("doc_medical_records").select("*").eq(
                "user_id", user_id).order("created_at", desc=True).limit(10).execute()
            
            if not records.data:
                return {"details": ["Нет медицинских записей для переобработки"]}
            
            details = []
            
            for record in records.data:
                record_id = record.get("id")
                content = record.get("content", "")
                
                # Используем улучшенный экстрактор для переобработки
                try:
                    extraction_result = await self.extractor.extract_tests_from_image(
                        f"data:text/plain,{content[:2000]}", ""
                    )
                    
                    if extraction_result.get("success"):
                        tests = extraction_result.get("structured_tests", [])
                        
                        # Сохраняем новые структурированные данные
                        for test in tests:
                            # Проверяем, нет ли уже такого анализа
                            existing = self.supabase.table("doc_structured_test_results").select("*").eq(
                                "user_id", user_id
                            ).eq("test_name", test.get("test_name")).execute()
                            
                            if not existing.data:
                                # Создаем новую запись
                                self.supabase.table("doc_structured_test_results").insert({
                                    "user_id": user_id,
                                    "test_name": test.get("test_name"),
                                    "result": test.get("result"),
                                    "reference_values": test.get("reference_values"),
                                    "units": test.get("units"),
                                    "test_system": test.get("test_system"),
                                    "equipment": test.get("equipment"),
                                    "test_date": test.get("test_date"),
                                    "notes": test.get("notes"),
                                    "source_record_id": record_id
                                }).execute()
                                
                                details.append(f"Добавлен новый анализ: '{test.get('test_name')}'")
                                logging.info(f"Добавлен новый анализ из записи {record_id}")
                
                except Exception as e:
                    logging.warning(f"Не удалось переобработать запись {record_id}: {e}")
            
            return {"details": details}
            
        except Exception as e:
            logging.error(f"Ошибка при переобработке медицинских записей: {e}")
            return {"details": [f"Ошибка: {str(e)}"]}
    
    def _clean_field_value(self, value: str) -> str:
        """Очищает значение поля от лишних символов"""
        if not value:
            return ""
        
        # Убираем символы форматирования
        cleaned = value.replace("**", "").replace("*", "")
        cleaned = cleaned.strip()
        
        # Дополнительная очистка
        if cleaned.lower() in ["не указан", "null", "none"]:
            return ""
        
        return cleaned
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Проверяет валидность даты"""
        if not date_str:
            return False
        
        try:
            # Пытаемся распарсить дату
            from datetime import datetime
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except:
            return False
    
    async def _extract_result_from_medical_records(
        self, user_id: str, test_name: str, source_record_id: int
    ) -> Optional[str]:
        """Извлекает результат из медицинских записей"""
        try:
            if not source_record_id:
                return None
            
            # Получаем медицинскую запись
            record = self.supabase.table("doc_medical_records").select("*").eq(
                "id", source_record_id
            ).eq("user_id", user_id).execute()
            
            if not record.data:
                return None
            
            content = record.data[0].get("content", "")
            
            # Используем экстрактор для поиска результата
            result = await self.extractor.extract_specific_test(content, test_name)
            
            if result:
                return result.get("result")
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении результата из записей: {e}")
            return None
    
    async def _extract_date_from_medical_records(
        self, user_id: str, source_record_id: int
    ) -> Optional[str]:
        """Извлекает дату из медицинских записей"""
        try:
            if not source_record_id:
                return None
            
            # Получаем медицинскую запись
            record = self.supabase.table("doc_medical_records").select("*").eq(
                "id", source_record_id
            ).eq("user_id", user_id).execute()
            
            if not record.data:
                return None
            
            content = record.data[0].get("content", "")
            
            # Ищем дату в тексте
            import re
            date_patterns = [
                r'(\d{2})\.(\d{2})\.(\d{4})',  # DD.MM.YYYY
                r'(\d{2})/(\d{2})/(\d{4})',  # DD/MM/YYYY
                r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, content)
                if match:
                    if pattern == date_patterns[0]:  # DD.MM.YYYY
                        day, month, year = match.groups()
                        return f"{year}-{month}-{day}"
                    elif pattern == date_patterns[1]:  # DD/MM/YYYY
                        day, month, year = match.groups()
                        return f"{year}-{month}-{day}"
                    elif pattern == date_patterns[2]:  # YYYY-MM-DD
                        return content[match.start():match.end()]
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении даты из записей: {e}")
            return None

# Удобная функция для использования
async def enhanced_cleanup_all_tests(user_id: str, supabase_client: Client) -> Dict[str, Any]:
    """
    Выполняет комплексную очистку всех анализов пользователя
    
    Args:
        user_id: ID пользователя
        supabase_client: Клиент Supabase
        
    Returns:
        Dict с результатами очистки
    """
    cleanup = EnhancedDatabaseCleanup(supabase_client)
    return await cleanup.cleanup_all_test_results(user_id)