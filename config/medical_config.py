"""
Конфигурационная система для медицинских анализов с архитектурой Low-Code
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalConfig:
    """Конфигурационная система для медицинских анализов"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "config/medical_analysis_config.json")
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Загрузка конфигурации из файла"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Конфигурация загружена из {self.config_path}")
            else:
                logger.info("Файл конфигурации не найден, используется конфигурация по умолчанию")
                self.config = self.get_default_config()
                self.save_config()
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            "version": "1.0.0",
            "categories": {
                "biochemical": {
                    "name": "Биохимические анализы",
                    "keywords": ["билирубин", "алат", "асат", "ггт", "мочевина", "креатинин", "холестерин", "глюкоза", "мочевая кислота"],
                    "patterns": [
                        r"(билирубин|алат|асат|ггт)",
                        r"(мочевина|креатинин|холестерин)",
                        r"(глюкоза|с-реактивный белок|crp)",
                        r"(мочевая кислота|амилаза|липаза)"
                    ],
                    "priority": 1,
                    "description": "Показатели функции печени, почек и обмена веществ"
                },
                "hormonal": {
                    "name": "Гормональные анализы",
                    "keywords": ["ттг", "т3", "т4", "пролактин", "эстрадиол", "тестостерон", "кортизол"],
                    "patterns": [r"(ттг|т3|т4|пролактин)", r"(эстрадиол|тестостерон|кортизол)"],
                    "priority": 2,
                    "description": "Анализы гормонального статуса"
                },
                "hepatitis": {
                    "name": "Анализы на гепатиты",
                    "keywords": ["hbsag", "anti-hcv", "anti-hbc", "hbeag", "anti-hav", "anti-hev"],
                    "patterns": [r"(hbsag|anti-hcv|anti-hbc)", r"(hbeag|anti-hav|anti-hev)"],
                    "priority": 3,
                    "description": "Маркеры вирусных гепатитов"
                },
                "parasitic": {
                    "name": "Паразитологические анализы",
                    "keywords": ["opisthorchis", "toxocara", "lamblia", "ascaris", "echinococcus"],
                    "patterns": [r"(opisthorchis|toxocara|lamblia)", r"(ascaris|echinococcus)"],
                    "priority": 4,
                    "description": "Анализы на паразитарные инфекции"
                },
                "allergic": {
                    "name": "Аллергологические анализы",
                    "keywords": ["ige", "аллерг", "эозинофилы"],
                    "patterns": [r"(ige|аллерг)", r"эозинофилы"],
                    "priority": 5,
                    "description": "Показатели аллергического статуса"
                },
                "hematology": {
                    "name": "Гематологические анализы",
                    "keywords": ["гемоглобин", "эритроциты", "лейкоциты", "тромбоциты", "гематокрит"],
                    "patterns": [r"(гемоглобин|эритроциты)", r"(лейкоциты|тромбоциты|гематокрит)"],
                    "priority": 6,
                    "description": "Показатели крови"
                }
            },
            "extraction_patterns": {
                "test_name": [
                    r"([А-Яа-я\s\-\(\)]+(?:анализ|белок|фермент|гормон|вирус|антитела))",
                    r"([A-Z][a-z\s]+(?:test|antibody|antigen))",
                    r"([А-Яа-яA-Za-z\s\-\(\)]+\d*(?:\.\d+)?\s*(?:мг/л|мЕд/л|нг/мл|ммоль/л|г/л|ед/л))",
                    r"(HBsAg|Anti-HCV|Anti-HBc|Anti-HAV|Anti-HEV|HBeAg)",
                    r"(ТТГ|Т3|Т4|IgE|IgG|IgM|IgA)"
                ],
                "result": [
                    r"([0-9]+\.?[0-9]*)",
                    r"(отрицательно|положительно|negative|positive)",
                    r"(не обнаружено|обнаружено)",
                    r"(норма|в норме|повышен|понижен)"
                ],
                "reference": [
                    r"([0-9]+\.?[0-9]*\s*[-–—]\s*[0-9]+\.?[0-9]*)",
                    r"(<\s*[0-9]+\.?[0-9]*)",
                    r"(>\s*[0-9]+\.?[0-9]*)",
                    r"(референсные значения|норма)\s*[:\-]?\s*([0-9\.\s<>\-—–]+)"
                ],
                "units": [
                    r"(мг/л|мЕд/л|нг/мл|ммоль/л|г/л|ед/л|мм/ч|мкмоль/л|пг/мл)",
                    r"(mg/l|mIU/l|ng/ml|mmol/l|g/l|U/l|mm/h|µmol/l|pg/ml)",
                    r"(×10[⁹³]/л|×10¹²/л)"
                ],
                "date": [
                    r"(\d{2}\.\d{2}\.\d{4})",
                    r"(\d{4}-\d{2}-\d{2})",
                    r"(\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4})"
                ]
            },
            "llm_functions": {
                "extract_medical_data": {
                    "name": "extract_medical_data",
                    "description": "Извлекает структурированные медицинские данные из документа",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_info": {"$ref": "#/definitions/patient_info"},
                            "document_info": {"$ref": "#/definitions/document_info"},
                            "medical_tests": {"$ref": "#/definitions/medical_tests"},
                            "summary": {"type": "string", "description": "Краткое резюме анализа"}
                        },
                        "required": ["medical_tests"]
                    }
                },
                "categorize_test": {
                    "name": "categorize_test",
                    "description": "Определяет категорию медицинского анализа",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_name": {"type": "string", "description": "Название анализа"},
                            "category": {"type": "string", "description": "Категория анализа"},
                            "confidence": {"type": "number", "description": "Уверенность в классификации (0-1)"}
                        },
                        "required": ["test_name", "category"]
                    }
                }
            },
            "definitions": {
                "patient_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Имя пациента"},
                        "birth_date": {"type": "string", "description": "Дата рождения"},
                        "age": {"type": "integer", "description": "Возраст"},
                        "gender": {"type": "string", "description": "Пол"},
                        "document_number": {"type": "string", "description": "Номер документа"}
                    }
                },
                "document_info": {
                    "type": "object",
                    "properties": {
                        "test_date": {"type": "string", "description": "Дата анализа"},
                        "laboratory": {"type": "string", "description": "Лаборатория"},
                        "doctor": {"type": "string", "description": "Врач"},
                        "document_type": {"type": "string", "description": "Тип документа"}
                    }
                },
                "medical_tests": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/medical_test"},
                    "description": "Список медицинских анализов"
                },
                "medical_test": {
                    "type": "object",
                    "properties": {
                        "test_name": {"type": "string", "description": "Название анализа"},
                        "result": {"type": "string", "description": "Результат"},
                        "reference_values": {"type": "string", "description": "Референсные значения"},
                        "units": {"type": "string", "description": "Единицы измерения"},
                        "category": {"type": "string", "description": "Категория"},
                        "abnormal": {"type": "boolean", "description": "Отклонение от нормы"},
                        "test_date": {"type": "string", "description": "Дата анализа"},
                        "laboratory": {"type": "string", "description": "Лаборатория"}
                    },
                    "required": ["test_name", "result"]
                }
            },
            "response_templates": {
                "success": {
                    "header": "📊 **Анализ медицинских документов:**\n\n",
                    "patient_section": "👤 **Пациент:** {name}\n🎂 **Дата рождения:** {birth_date}\n📅 **Возраст:** {age} лет\n⚧️ **Пол:** {gender}\n\n",
                    "document_section": "📋 **Информация о документе:**\n📅 **Дата анализа:** {test_date}\n🏥 **Лаборатория:** {laboratory}\n👨‍⚕️ **Врач:** {doctor}\n\n",
                    "category_section": "🔬 **{category_name}:**\n{tests}\n\n",
                    "test_item": "{status} **{test_name}:** {result} {units} (норма: {reference})",
                    "summary_section": "📋 **Медицинское резюме:**\n{summary}\n\n",
                    "recommendations_section": "💡 **Рекомендации:**\n{recommendations}",
                    "footer": "\n⚠️ *Данный анализ не является медицинской диагнозом. Для точной интерпретации результатов проконсультируйтесь с врачом.*"
                },
                "error": {
                    "extraction_failed": "❌ Не удалось извлечь данные из документа",
                    "processing_failed": "❌ Ошибка обработки документа: {error}",
                    "no_text_found": "❌ Не удалось извлечь текст из документа",
                    "invalid_format": "❌ Неверный формат документа",
                    "ai_error": "❌ Ошибка ИИ-обработки: {error}"
                },
                "status": {
                    "normal": "✅",
                    "abnormal": "⚠️",
                    "critical": "🚨"
                }
            },
            "processing": {
                "max_text_length": 50000,
                "max_tests_per_document": 100,
                "confidence_threshold": 0.7,
                "fallback_enabled": True,
                "caching_enabled": True,
                "validation_enabled": True
            },
            "quality_control": {
                "min_test_name_length": 2,
                "max_test_name_length": 200,
                "required_fields": ["test_name", "result"],
                "normalization_enabled": True,
                "duplicate_detection": True
            }
        }
    
    def save_config(self):
        """Сохранение конфигурации в файл"""
        try:
            # Создаем директорию если нужно
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            logger.info(f"Конфигурация сохранена в {self.config_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    def get_categories(self) -> Dict[str, Dict]:
        """Получить все категории"""
        return self.config.get("categories", {})
    
    def get_category(self, category_key: str) -> Optional[Dict]:
        """Получить категорию по ключу"""
        return self.config.get("categories", {}).get(category_key)
    
    def get_patterns(self, pattern_type: str) -> List[str]:
        """Получить паттерны для извлечения"""
        patterns = self.config.get("extraction_patterns", {}).get(pattern_type, [])
        return [re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern for pattern in patterns]
    
    def get_llm_functions(self) -> Dict[str, Any]:
        """Получить функции для LLM"""
        return self.config.get("llm_functions", {})
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """Получить шаблон ответа"""
        return self.config.get("response_templates", {}).get(template_name, {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Получить конфигурацию обработки"""
        return self.config.get("processing", {})
    
    def get_quality_config(self) -> Dict[str, Any]:
        """Получить конфигурацию качества"""
        return self.config.get("quality_control", {})
    
    def get_definitions(self) -> Dict[str, Any]:
        """Получить определения для JSON schema"""
        return self.config.get("definitions", {})
    
    def add_category(self, category_key: str, category_data: Dict[str, Any]):
        """Добавить новую категорию"""
        self.config.setdefault("categories", {})[category_key] = category_data
        logger.info(f"Добавлена категория: {category_key}")
    
    def update_category(self, category_key: str, category_data: Dict[str, Any]):
        """Обновить категорию"""
        if category_key in self.config.get("categories", {}):
            self.config["categories"][category_key].update(category_data)
            logger.info(f"Обновлена категория: {category_key}")
        else:
            logger.warning(f"Категория {category_key} не найдена")
    
    def remove_category(self, category_key: str):
        """Удалить категорию"""
        if category_key in self.config.get("categories", {}):
            del self.config["categories"][category_key]
            logger.info(f"Удалена категория: {category_key}")
        else:
            logger.warning(f"Категория {category_key} не найдена")
    
    def validate_config(self) -> Dict[str, Any]:
        """Валидация конфигурации"""
        issues = []
        warnings = []
        
        # Проверяем основные секции
        required_sections = ["categories", "extraction_patterns", "llm_functions", "response_templates"]
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Отсутствует обязательная секция: {section}")
        
        # Проверяем категории
        categories = self.get_categories()
        for cat_key, cat_data in categories.items():
            if not cat_data.get("name"):
                issues.append(f"У категории {cat_key} отсутствует имя")
            if not cat_data.get("keywords"):
                warnings.append(f"У категории {cat_key} отсутствуют ключевые слова")
        
        # Проверяем паттерны
        patterns = self.config.get("extraction_patterns", {})
        for pattern_type, pattern_list in patterns.items():
            if not isinstance(pattern_list, list):
                issues.append(f"Паттерны для {pattern_type} должны быть списком")
            else:
                for i, pattern in enumerate(pattern_list):
                    try:
                        re.compile(pattern, re.IGNORECASE)
                    except re.error as e:
                        issues.append(f"Неверный паттерн {pattern_type}[{i}]: {e}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику конфигурации"""
        categories = self.get_categories()
        patterns = self.config.get("extraction_patterns", {})
        
        return {
            "version": self.config.get("version", "unknown"),
            "categories_count": len(categories),
            "patterns_count": sum(len(patterns.get(pt, [])) for pt in patterns),
            "llm_functions_count": len(self.get_llm_functions()),
            "categories": list(categories.keys()),
            "pattern_types": list(patterns.keys()),
            "last_validated": self.validate_config().get("timestamp")
        }
    
    def export_config(self, export_path: str):
        """Экспорт конфигурации в файл"""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            logger.info(f"Конфигурация экспортирована в {export_path}")
        except Exception as e:
            logger.error(f"Ошибка экспорта конфигурации: {e}")
    
    def import_config(self, import_path: str):
        """Импорт конфигурации из файла"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # Валидация импортированной конфигурации
            self.config = imported_config
            validation = self.validate_config()
            
            if validation["valid"]:
                logger.info(f"Конфигурация импортирована из {import_path}")
                return True
            else:
                logger.error(f"Импортированная конфигурация невалидна: {validation['issues']}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка импорта конфигурации: {e}")
            return False

# Глобальный экземпляр конфигурации
medical_config = MedicalConfig()
