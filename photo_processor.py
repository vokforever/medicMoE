import re
import json
import logging
from typing import Dict, List, Optional
from models import call_model_with_failover

class SimplePhotoProcessor:
    """Упрощенный и надежный обработчик медицинских анализов с фото"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Простая категоризация по ключевым словам
        self.categories = {
            'Биохимические анализы': ['билирубин', 'алат', 'асат', 'ггт', 'мочевина', 'креатинин', 'холестерин', 'глюкоза', 'с-реактивный белок', 'crp'],
            'Гормональные анализы': ['ттг', 'т3', 'т4', 'пролактин', 'эстрадиол', 'тестостерон', 'кортизол'],
            'Анализы на гепатиты': ['hbsag', 'anti-hcv', 'anti-hbc', 'hbeag', 'anti-hbe'],
            'Общий анализ крови': ['гемоглобин', 'эритроциты', 'лейкоциты', 'тромбоциты', 'соэ'],
            'Аллергологические анализы': ['ige', 'аллерг'],
            'Паразитологические анализы': ['opisthorchis', 'toxocara', 'lamblia', 'ascaris']
        }
    
    async def process_photo(self, image_url: str) -> Dict:
        """Основной метод обработки фото"""
        try:
            self.logger.info("Начинаю обработку фото")
            
            # Шаг 1: Извлекаем текст из изображения
            extracted_text = await self._extract_text_from_image(image_url)
            
            if not extracted_text:
                return {"success": False, "error": "Не удалось извлечь текст из изображения"}
            
            # Шаг 2: Извлекаем структурированные данные
            structured_data = await self._extract_structured_data(extracted_text)
            
            # Шаг 3: Генерируем простой ответ
            response = self._generate_simple_response(structured_data)
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "structured_data": structured_data,
                "response": response
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки фото: {e}")
            return {"success": False, "error": f"Ошибка обработки: {str(e)}"}
    
    async def _extract_text_from_image(self, image_url: str) -> Optional[str]:
        """Извлечение текста из изображения"""
        try:
            # Добавляем URL изображения к сообщению для vision модели
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Извлеки весь текст с этого медицинского документа. Возвращай только текст, без комментариев."
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
            
            response, provider, metadata = await call_model_with_failover(
                messages=messages,
                model_type="vision",
                system_prompt="Ты - помощник для извлечения текста. Возвращай только текст с изображения."
            )
            
            if response and isinstance(response, str):
                text = response.strip()
                self.logger.info(f"Извлечен текст длиной {len(text)} символов провайдером {provider}")
                return text
            elif response and isinstance(response, tuple):
                text = response[0].strip()
                self.logger.info(f"Извлечен текст длиной {len(text)} символов")
                return text
            else:
                self.logger.warning(f"Неожиданный формат ответа: {type(response)}")
                return None
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения текста: {e}")
            return None
    
    async def _extract_structured_data(self, text: str) -> List[Dict]:
        """Извлечение структурированных данных из текста"""
        try:
            # Простой и четкий промпт
            prompt = f"""
Извлеки из этого медицинского текста анализы в формате JSON:

{text}

Верни только JSON массив:
[
  {{"test_name": "Название анализа", "result": "Результат", "reference_values": "Норма", "units": "Единицы"}},
  ...
]
"""
            
            response, provider, metadata = await call_model_with_failover(
                messages=[{"role": "user", "content": prompt}],
                model_type="text",
                system_prompt="Ты - медицинский аналитик. Возвращай только JSON без комментариев."
            )
            
            if response:
                # Обрабатываем оба формата ответа
                if isinstance(response, str):
                    json_text = response
                elif isinstance(response, tuple):
                    json_text = response[0]
                else:
                    self.logger.warning(f"Неожиданный формат ответа: {type(response)}")
                    return self._simple_parse(text)
                
                # Ищем JSON в ответе
                json_match = re.search(r'\[.*\]', json_text, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        self.logger.info(f"Успешно извлечено {len(data)} анализов из JSON")
                        return self._validate_and_clean_data(data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Ошибка парсинга JSON: {e}")
                        self.logger.error(f"JSON текст: {json_text[:200]}...")
            
            # Fallback к простому парсингу
            self.logger.info("Используем простой парсинг как fallback")
            return self._simple_parse(text)
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения структурированных данных: {e}")
            return []
    
    def _simple_parse(self, text: str) -> List[Dict]:
        """Простой парсинг как fallback"""
        tests = []
        lines = text.split('\n')
        
        for line in lines:
            # Ищем строки с медицинскими терминами
            if any(keyword in line.lower() for keyword in ['белок', 'ттг', 'гепатит', 'hbsag', 'anti']):
                
                # Простые регулярные выражения
                test_name_match = re.search(r'([А-Яа-я\s\-\(\)]+(?:анализ|белок|фермент|гормон|вирус|антитела|показатель))', line)
                result_match = re.search(r'([0-9]+\.?[0-9]*)', line)
                reference_match = re.search(r'([0-9]+\.?[0-9]*\s*[-–]\s*[0-9]+\.?[0-9]*|<\s*[0-9]+\.?[0-9]*|>\s*[0-9]+\.?[0-9]*)', line)
                units_match = re.search(r'(мг/л|мЕд/л|нг/мл|ммоль/л|г/л|ед/л)', line)
                
                if test_name_match and result_match:
                    test = {
                        "test_name": test_name_match.group(1).strip(),
                        "result": result_match.group(1),
                        "reference_values": reference_match.group(1) if reference_match else "",
                        "units": units_match.group(1) if units_match else ""
                        # Убираем category так как нет такой колонки в БД
                    }
                    tests.append(test)
        
        return tests
    
    def _validate_and_clean_data(self, data: List[Dict]) -> List[Dict]:
        """Валидация и очистка данных"""
        cleaned_data = []
        
        for item in data:
            if isinstance(item, dict) and item.get('test_name') and item.get('result'):
                # Очищаем данные
                cleaned_item = {
                    'test_name': item['test_name'].strip(),
                    'result': item['result'].strip(),
                    'reference_values': item.get('reference_values', '').strip(),
                    'units': item.get('units', '').strip()
                    # Убираем category так как нет такой колонки в БД
                }
                
                cleaned_data.append(cleaned_item)
        
        return cleaned_data
    
    def _categorize_test(self, test_name: str) -> str:
        """Категоризация теста по ключевым словам"""
        test_name_lower = test_name.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in test_name_lower for keyword in keywords):
                return category
        
        return "Другие анализы"
    
    def _generate_simple_response(self, structured_data: List[Dict]) -> str:
        """Генерация простого ответа"""
        if not structured_data:
            return "📊 Не удалось извлечь данные анализов. Попробуйте сделать фото более четким."
        
        response = "📊 **Результаты анализов:**\n\n"
        
        # Группируем по категориям
        categorized = {}
        for test in structured_data:
            category = test.get('category', 'Другие анализы')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(test)
        
        # Формируем ответ
        for category, tests in categorized.items():
            response += f"🔬 **{category}:**\n"
            
            for test in tests:
                name = test.get('test_name', 'Неизвестный анализ')
                result = test.get('result', 'Не указан')
                reference = test.get('reference_values', '')
                units = test.get('units', '')
                
                response += f"• {name}: {result}"
                if units:
                    response += f" {units}"
                if reference:
                    response += f" (норма: {reference})"
                response += "\n"
            
            response += "\n"
        
        # Добавляем рекомендации
        response += "💡 **Рекомендации:**\n"
        response += "• Проконсультируйтесь с врачом для детальной интерпретации\n"
        response += "• Обратите внимание на показатели, выходящие за пределы нормы\n"
        response += "• Сохраните результаты для отслеживания динамики"
        
        return response
