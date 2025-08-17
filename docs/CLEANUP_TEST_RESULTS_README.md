# Очистка результатов анализов от лишних символов

## Описание проблемы

В системе анализа медицинских изображений иногда возникают проблемы с форматированием результатов анализов. Вместо конкретных значений (например, "ОТРИЦАТЕЛЬНО" или "0.7 мг/л") в базу данных записываются символы форматирования типа `**` или `*`.

### Примеры проблемных записей

**До очистки:**
```
Anti-HB core total: **
Тест-система: ** Anti-HBc, Abbott
Оборудование: ** Abbott, Alinity i
```

**После очистки:**
```
Anti-HB core total: Не указан
Тест-система: Anti-HBc, Abbott
Оборудование: Abbott, Alinity i
```

## Решение

Реализована система автоматической очистки результатов анализов от лишних символов форматирования с интеллектуальным поиском реальных значений в контексте.

### Основные компоненты

#### 1. Улучшенная функция очистки (`_clean_result_enhanced`)

```python
def _clean_result_enhanced(self, result: str, all_lines: List[str], line_index: int) -> str:
    """Улучшенная очистка результата с поиском в контексте"""
    
    # Сначала пробуем обычную очистку
    clean_result = self._clean_result(result)
    
    # Если результат содержит только звездочки, ищем в контексте
    if clean_result == "Не указан" and ("**" in result or "*" in result):
        real_result = self._extract_real_value_from_context(all_lines, line_index, "result")
        if real_result:
            clean_result = self._clean_result(real_result)
    
    # Если все еще не указан, ищем по ключевым словам
    if clean_result == "Не указан":
        context_result = self._search_result_in_context(all_lines, line_index)
        if context_result:
            clean_result = context_result
    
    return clean_result
```

#### 2. Поиск в контексте (`_search_result_in_context`)

Функция ищет результаты анализов по ключевым словам в соседних строках:

```python
def _search_result_in_context(self, all_lines: List[str], line_index: int) -> Optional[str]:
    """Ищет результат анализа по ключевым словам в контексте"""
    
    search_range = 10  # Поиск в 10 строках вверх и вниз
    
    for i in range(max(0, line_index - search_range), min(len(all_lines), line_index + search_range + 1)):
        line = all_lines[i].strip()
        
        # Ищем строки с результатами анализов
        if any(keyword in line.lower() for keyword in [
            'отрицательно', 'положительно', 'negative', 'positive',
            'норма', 'норме', 'в норме', 'в пределах нормы',
            'повышен', 'понижен', 'высокий', 'низкий'
        ]):
            # Извлекаем и очищаем значение
            clean_value = self._clean_result(line)
            if clean_value != "Не указан":
                return clean_value
    
    return None
```

#### 3. Очистка существующих записей (`cleanup_existing_test_results`)

Функция для очистки уже сохраненных в базе данных результатов:

```python
async def cleanup_existing_test_results(self, user_id: str) -> Dict[str, Any]:
    """Очищает существующие результаты анализов от лишних символов"""
    
    # Получаем все анализы пользователя
    tests = self.supabase.table("doc_structured_test_results").select("*").eq("user_id", user_id).execute()
    
    cleaned_count = 0
    for test in tests.data:
        # Проверяем, нужна ли очистка
        needs_cleaning = False
        cleaned_result = test.get("result", "")
        cleaned_test_system = test.get("test_system", "")
        cleaned_equipment = test.get("equipment", "")
        
        # Очищаем каждое поле
        if test.get("result") and ("**" in test.get("result") or "*" in test.get("result")):
            cleaned_result = self._clean_result(test.get("result"))
            needs_cleaning = True
        
        # Если нужна очистка, обновляем запись
        if needs_cleaning:
            update_data = {
                "result": cleaned_result,
                "test_system": cleaned_test_system,
                "equipment": cleaned_equipment,
                "updated_at": datetime.now().isoformat()
            }
            
            self.supabase.table("doc_structured_test_results").update(update_data).eq("id", test.get("id")).execute()
            cleaned_count += 1
    
    return {
        "success": True,
        "cleaned_count": cleaned_count,
        "message": f"Очистка завершена. Очищено {cleaned_count} анализов"
    }
```

#### 4. Переобработка медицинских записей (`reprocess_medical_records`)

Функция для полной переобработки всех медицинских записей с улучшенной логикой:

```python
async def reprocess_medical_records(self, user_id: str) -> Dict[str, Any]:
    """Переобрабатывает медицинские записи для улучшения структурированных данных"""
    
    # Удаляем старые структурированные данные
    old_tests = self.supabase.table("doc_structured_test_results").select("*").eq("user_id", user_id).execute()
    
    if old_tests.data:
        for test in old_tests.data:
            self.supabase.table("doc_structured_test_results").delete().eq("id", test.get("id")).execute()
    
    # Переобрабатываем записи с улучшенной логикой
    result = await self.extract_and_structure_tests(user_id)
    
    return result
```

## Использование

### Команды Telegram бота

#### 1. `/cleanup_tests` - Очистка результатов анализов

Очищает существующие результаты анализов от лишних символов форматирования.

**Пример использования:**
```
Пользователь: /cleanup_tests

Бот: 🧹 Начинаю очистку результатов анализов... Пожалуйста, подождите.

Бот: ✅ Очистка завершена!

🧹 Очищено 5 результатов анализов от лишних символов.

Теперь ваши анализы будут отображаться корректно без лишних символов форматирования.
```

#### 2. `/reprocess_tests` - Переобработка анализов

Полностью переобрабатывает все медицинские записи с улучшенной логикой извлечения.

**Пример использования:**
```
Пользователь: /reprocess_tests

Бот: 🔄 Начинаю переобработку медицинских записей... Это может занять некоторое время.

Бот: ✅ Переобработка завершена!

🔄 Переобработано 12 анализов с улучшенной логикой.

Теперь ваши анализы будут корректно структурированы и очищены от лишних символов.
```

### Программное использование

#### Очистка существующих записей

```python
from structured_tests_agent import TestExtractionAgent

# Создаем агент
agent = TestExtractionAgent(supabase)

# Очищаем результаты анализов пользователя
cleanup_result = await agent.cleanup_existing_test_results(user_id)

if cleanup_result.get("success"):
    cleaned_count = cleanup_result.get("cleaned_count", 0)
    print(f"Очищено {cleaned_count} результатов анализов")
```

#### Переобработка медицинских записей

```python
# Переобрабатываем все записи
reprocess_result = await agent.reprocess_medical_records(user_id)

if reprocess_result.get("success"):
    tests_count = reprocess_result.get("tests_count", 0)
    print(f"Переобработано {tests_count} анализов")
```

## Тестирование

### Тестовый скрипт

Создан тестовый скрипт `test/test_cleanup_test_results.py` для проверки функциональности:

```bash
# Запуск тестов
python test/test_cleanup_test_results.py
```

### Что тестируется

1. **Функция очистки** - проверка корректности очистки символов
2. **Функция переобработки** - проверка полной переобработки записей
3. **Интеграция с базой данных** - проверка сохранения очищенных данных
4. **Логирование** - проверка корректности логирования операций

## Логирование

Все операции очистки логируются для аудита:

```
2025-08-17 17:15:00 - INFO - Начинаю очистку результатов анализов для пользователя: ff0fc454-319a-5ba8-8901-06b6bc0e59f4
2025-08-17 17:15:01 - INFO - Найдено реальное значение в контексте: ОТРИЦАТЕЛЬНО -> ОТРИЦАТЕЛЬНО
2025-08-17 17:15:02 - INFO - Очищен анализ 20: Anti-HB core total
2025-08-17 17:15:03 - INFO - Очистка завершена: 5 анализов очищено
```

## Мониторинг и статистика

### Метрики очистки

- **Количество очищенных анализов** - общее число исправленных записей
- **Типы очищенных полей** - какие поля были исправлены (результат, тест-система, оборудование)
- **Время выполнения** - продолжительность операции очистки
- **Успешность операций** - процент успешно очищенных записей

### Отчеты

После каждой операции очистки формируется детальный отчет:

```json
{
    "success": true,
    "cleaned_count": 5,
    "message": "Очистка завершена. Очищено 5 анализов",
    "updated_tests": [
        {
            "id": "20",
            "test_name": "Anti-HB core total",
            "old_result": "**",
            "new_result": "Не указан",
            "old_test_system": "** Anti-HBc, Abbott",
            "new_test_system": "Anti-HBc, Abbott",
            "old_equipment": "** Abbott, Alinity i",
            "new_equipment": "Abbott, Alinity i"
        }
    ]
}
```

## Безопасность

### Валидация данных

- Проверка принадлежности анализов пользователю
- Валидация обновляемых данных
- Логирование всех изменений для аудита

### Обработка ошибок

- Graceful handling ошибок базы данных
- Fallback на безопасные значения по умолчанию
- Информативные сообщения об ошибках для пользователя

## Производительность

### Оптимизации

- Пакетная обработка записей
- Кэширование результатов очистки
- Асинхронные операции для больших объемов данных

### Ограничения

- Рекомендуется выполнять очистку в непиковые часы
- Для больших объемов данных может потребоваться несколько минут
- Рекомендуется выполнять очистку не чаще раза в день

## Планы развития

### Автоматическая очистка

- Планировщик для автоматической очистки новых записей
- Интеграция с системой мониторинга качества данных
- Автоматические уведомления о проблемах с форматированием

### Улучшение алгоритмов

- Машинное обучение для распознавания паттернов форматирования
- Интеллектуальное извлечение значений из изображений
- Автоматическая категоризация типов анализов

### Расширенная аналитика

- Статистика качества данных по типам анализов
- Тренды изменения качества данных во времени
- Рекомендации по улучшению процесса извлечения

## Заключение

Система очистки результатов анализов решает проблему некорректного форматирования данных и обеспечивает высокое качество структурированной информации. Регулярное использование команд `/cleanup_tests` и `/reprocess_tests` поможет поддерживать базу данных в актуальном состоянии.
