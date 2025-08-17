import logging
import requests
from typing import List, Tuple, Dict, Any
from config import MODEL_CONFIG, TOKEN_LIMITS

# Функция для проверки доступности модели
async def check_model_availability(provider: str, model_name: str) -> bool:
    """Проверяет доступность модели и наличие токенов"""
    try:
        logging.info(f"Проверка доступности модели {model_name} у провайдера {provider}")
        
        config = MODEL_CONFIG.get(provider)
        if not config or not config.get("client"):
            logging.warning(f"Провайдер {provider} не настроен")
            return False

        # Проверка лимитов токенов
        token_limit = TOKEN_LIMITS.get(provider, {})
        if token_limit.get("daily_limit", 0) > 0 and token_limit.get("used_today", 0) >= token_limit["daily_limit"]:
            logging.warning(f"Достигнут лимит токенов для провайдера {provider}")
            return False

        # Для OpenRouter можно проверить доступность модели через API
        if provider == "openrouter":
            try:
                headers = {
                    "Authorization": f"Bearer {config['api_key']}"
                }
                response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    available_models = [m["id"] for m in models]
                    is_available = model_name in available_models
                    logging.info(f"Модель {model_name} доступна в OpenRouter: {is_available}")
                    return is_available
                else:
                    logging.warning(f"Ошибка при проверке доступности модели OpenRouter: {response.status_code}")
            except Exception as e:
                logging.error(f"Ошибка при проверке доступности модели OpenRouter: {e}")

        # Для других провайдеров просто проверяем, что API ключ существует
        logging.info(f"Модель {model_name} считается доступной для провайдера {provider}")
        return True
    except Exception as e:
        logging.error(f"Ошибка при проверке доступности модели {model_name} у провайдера {provider}: {e}")
        return False

# Функция для обновления счетчика использованных токенов
def update_token_usage(provider: str, tokens_used: int):
    """Обновляет счетчик использованных токенов для провайдера"""
    if provider in TOKEN_LIMITS:
        TOKEN_LIMITS[provider]["used_today"] += tokens_used
        logging.info(f"Обновлен счетчик токенов для {provider}: +{tokens_used}, всего сегодня: {TOKEN_LIMITS[provider]['used_today']}")
    else:
        logging.warning(f"Провайдер {provider} не найден в TOKEN_LIMITS")

# Функция для сброса счетчиков токенов (можно вызывать раз в день)
def reset_token_usage():
    """Сбрасывает ежедневные счетчики токенов"""
    logging.info("Сброс счетчиков токенов")
    
    for provider in TOKEN_LIMITS:
        old_value = TOKEN_LIMITS[provider]["used_today"]
        TOKEN_LIMITS[provider]["used_today"] = 0
        logging.info(f"Сброшен счетчик токенов для {provider}: {old_value} -> 0")
    
    logging.info("Все счетчики токенов сброшены")

# Универсальная функция для вызова моделей с failover
async def call_model_with_failover(
    messages: List[Dict[str, str]],
    model_preference: str = None,
    model_type: str = None,  # Новый параметр для указания типа модели
    system_prompt: str = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Универсальная функция для вызова моделей с failover.
    
    Args:
        messages: Список сообщений для модели
        model_preference: Предпочтительная модель (опционально)
        model_type: Тип модели (например, "vision" для анализа изображений)
        system_prompt: Системный промпт (опционально)
    
    Returns:
        (response, provider, metadata)
    """
    logging.info(f"call_model_with_failover: тип модели: {model_type}, предпочтение: {model_preference}")
    logging.info(f"Количество сообщений: {len(messages)}")
    
    # Формируем список всех моделей с учетом приоритета
    all_models = []
    for provider, config in MODEL_CONFIG.items():
        for model in config["models"]:
            # Добавляем информацию о типе модели
            model_info = {
                "provider": provider,
                "name": model["name"],
                "priority": model["priority"],
                "type": model.get("type", "text"),  # По умолчанию text
                "client": config["client"]
            }
            all_models.append(model_info)
    
    logging.info(f"Всего доступных моделей: {len(all_models)}")
    
    # Фильтруем модели по типу, если указан
    if model_type:
        original_count = len(all_models)
        all_models = [m for m in all_models if m.get("type") == model_type]
        logging.info(f"Отфильтровано по типу '{model_type}': {len(all_models)} из {original_count}")
        
        if not all_models:
            logging.warning(f"Нет доступных моделей типа '{model_type}'")
            # Для vision задач не используем text модели как fallback
            if model_type == "vision":
                logging.error("Vision модели недоступны. Text модели не подходят для анализа изображений.")
                return "😔 К сожалению, все vision модели временно недоступны (закончились лимиты на сегодня у всех провайдеров). Попробуйте повторить запрос завтра.", "", {}
            else:
                # Для text задач используем все доступные модели
                logging.info("Используем все доступные модели для text задач")
                for provider, config in MODEL_CONFIG.items():
                    for model in config["models"]:
                        model_info = {
                            "provider": provider,
                            "name": model["name"],
                            "priority": model["priority"],
                            "type": model.get("type", "text"),
                            "client": config["client"]
                        }
                        all_models.append(model_info)
                logging.info(f"Восстановлено общее количество моделей: {len(all_models)}")
    
    # Сортируем по приоритету
    all_models.sort(key=lambda x: x["priority"])
    logging.info(f"Модели отсортированы по приоритету")
    
    # Если указана предпочтительная модель, перемещаем её в начало
    if model_preference:
        preferred_models = [m for m in all_models if m["name"] == model_preference]
        other_models = [m for m in all_models if m["name"] != model_preference]
        all_models = preferred_models + other_models
        logging.info(f"Предпочтительная модель '{model_preference}' перемещена в начало списка")
    
    last_error = None
    logging.info(f"Начинаю попытки вызова моделей, всего моделей: {len(all_models)}")
    
    # Пробуем модели в порядке приоритета
    for i, model_info in enumerate(all_models):
        provider = model_info["provider"]
        model_name = model_info["name"]
        client = model_info["client"]
        
        logging.info(f"Попытка {i+1}/{len(all_models)}: модель {model_name} от провайдера {provider}")
        
        # Проверяем доступность модели
        if not await check_model_availability(provider, model_name):
            logging.info(f"Модель {model_name} провайдера {provider} недоступна, пробуем следующую")
            continue
        
        try:
            logging.info(f"Пробую модель {model_name} от провайдера {provider}")
            
            # Дополнительная диагностика для Cerebras
            if provider == "cerebras":
                config = MODEL_CONFIG[provider]
                logging.info(f"Cerebras API Key: {config.get('api_key', '')[:10]}...")
                logging.info(f"Cerebras Base URL: {config.get('base_url', '')}")
                logging.info(f"Client initialized: {config.get('client') is not None}")
            
            # Добавляем системный промпт, если он указан
            if system_prompt:
                # Проверяем, есть ли уже системный промпт в сообщениях
                has_system = any(msg.get("role") == "system" for msg in messages)
                if not has_system:
                    messages = [{"role": "system", "content": system_prompt}] + messages
                    logging.info("Добавлен системный промпт в сообщения")
            
            # Добавляем заголовки для OpenRouter
            extra_headers = {}
            if provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://github.com/vokforever/ai-doctor",
                    "X-Title": "AI Doctor Bot"
                }
                logging.info("Добавлены специальные заголовки для OpenRouter")
            
            # Выполняем запрос
            # Для Qwen 3 235B Thinking модели добавляем специальные параметры
            extra_params = {}
            if provider == "cerebras" and "qwen-3-235b" in model_name:
                extra_params = {
                    "max_tokens": 64000,  # Cerebras API использует max_tokens, не max_completion_tokens
                    "temperature": 0.7,
                    "top_p": 0.9
                }
                logging.info(f"Добавлены специальные параметры для Qwen 3 235B: {extra_params}")
            
            logging.info(f"Вызываю модель {model_name} с {len(messages)} сообщениями")
            
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **extra_params,
                **({"extra_headers": extra_headers} if extra_headers else {})
            )
            
            logging.info(f"Модель {model_name} успешно ответила")
            
            # Получаем ответ
            current_answer = completion.choices[0].message.content
            logging.info(f"Получен ответ длиной {len(current_answer)} символов")
            
            # Для некоторых моделей (например, Cerebras) может быть цепочка размышлений
            thinking_process = ""
            if provider == "cerebras" and hasattr(completion.choices[0], 'thinking'):
                thinking_process = completion.choices[0].thinking
                logging.info("Найдена цепочка размышлений в choices[0].thinking")
            elif provider == "cerebras" and hasattr(completion.choices[0].message, 'thinking'):
                thinking_process = completion.choices[0].message.thinking
                logging.info("Найдена цепочка размышлений в choices[0].message.thinking")
            
            # Обновляем счетчик токенов (если есть информация)
            if hasattr(completion, 'usage') and completion.usage:
                tokens_used = completion.usage.total_tokens
                update_token_usage(provider, tokens_used)
                logging.info(f"Использовано токенов {provider}: {tokens_used}")
            
            # Сохраняем информацию о модели
            metadata = {
                "provider": provider,
                "model": model_name,
                "type": model_info.get("type", "text"),
                "thinking": thinking_process,
                "usage": getattr(completion, 'usage', None)
            }
            
            logging.info(f"Успешно завершена работа с моделью {model_name} от провайдера {provider}")
            
            # Возвращаем первый успешный ответ
            return current_answer, provider, metadata
        except Exception as e:
            last_error = e
            error_msg = f"Ошибка при использовании модели {model_name} от провайдера {provider}: {e}"
            
            logging.error(f"Ошибка при вызове модели {model_name}: {e}")
            
            # Дополнительная диагностика для Cerebras
            if provider == "cerebras":
                error_msg += f"\nПроверьте:\n"
                error_msg += f"- Правильность API ключа\n"
                error_msg += f"- Доступность модели {model_name}\n"
                error_msg += f"- Лимиты токенов\n"
                error_msg += f"- Статус API Cerebras"
                
                # Проверяем конкретные ошибки Cerebras
                if "model_not_found" in str(e):
                    error_msg += f"\n❌ Модель {model_name} не найдена. Проверьте правильность названия."
                    logging.error(f"Модель {model_name} не найдена в Cerebras")
                elif "authentication" in str(e).lower():
                    error_msg += f"\n❌ Ошибка аутентификации. Проверьте API ключ."
                    logging.error("Ошибка аутентификации в Cerebras")
                elif "rate_limit" in str(e).lower():
                    error_msg += f"\n❌ Превышен лимит запросов."
                    logging.error("Превышен лимит запросов в Cerebras")
            
            # Дополнительная диагностика для OpenRouter
            elif provider == "openrouter":
                error_msg += f"\nПроверьте:\n"
                error_msg += f"- Правильность API ключа\n"
                error_msg += f"- Лимиты токенов\n"
                error_msg += f"- Статус API OpenRouter"
                
                # Проверяем конкретные ошибки OpenRouter
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    error_msg += f"\n❌ Превышен лимит запросов OpenRouter. Попробуйте завтра или добавьте кредиты."
                    logging.error("Превышен лимит запросов в OpenRouter")
                elif "model_not_found" in str(e).lower():
                    error_msg += f"\n❌ Модель {model_name} не найдена в OpenRouter."
                    logging.error(f"Модель {model_name} не найдена в OpenRouter")
            
            # Дополнительная диагностика для Groq
            elif provider == "groq":
                error_msg += f"\nПроверьте:\n"
                error_msg += f"- Правильность API ключа\n"
                error_msg += f"- Доступность модели {model_name}\n"
                error_msg += f"- Лимиты токенов\n"
                error_msg += f"- Статус API Groq"
                
                # Проверяем конкретные ошибки Groq
                if "model_not_found" in str(e).lower():
                    error_msg += f"\n❌ Модель {model_name} не найдена в Groq. Проверьте правильность названия."
                    logging.error(f"Модель {model_name} не найдена в Groq")
                elif "authentication" in str(e).lower():
                    error_msg += f"\n❌ Ошибка аутентификации. Проверьте API ключ."
                    logging.error("Ошибка аутентификации в Groq")
            
            logging.warning(error_msg)
            continue
    
    # Если все модели не сработали
    logging.error(f"Все модели недоступны. Последняя ошибка: {last_error}")
    
    if model_type == "vision":
        error_message = "😔 К сожалению, все vision модели временно недоступны (закончились лимиты на сегодня у всех провайдеров). Попробуйте повторить запрос завтра."
    else:
        error_message = "😔 К сожалению, произошла ошибка при генерации ответа. Все модели временно недоступны. Попробуйте повторить запрос позже."
    
    return error_message, "", {}
