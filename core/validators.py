"""
Система валидации медицинских данных
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from config.medical_config import medical_config

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Результат валидации"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0

class MedicalDataValidator:
    """Валидатор медицинских данных"""
    
    def __init__(self, config=None):
        self.config = config or medical_config
        self.quality_config = self.config.get_quality_config()
        self.processing_config = self.config.get_processing_config()
    
    def validate_test_name(self, test_name: str) -> ValidationResult:
        """Валидация названия анализа"""
        errors = []
        warnings = []
        confidence = 0.0
        
        if not test_name or not isinstance(test_name, str):
            errors.append("Название анализа отсутствует или имеет неверный тип")
            return ValidationResult(False, errors, warnings, None, 0.0)
        
        test_name = test_name.strip()
        
        # Проверка длины
        min_len = self.quality_config.get("min_test_name_length", 2)
        max_len = self.quality_config.get("max_test_name_length", 200)
        
        if len(test_name) < min_len:
            errors.append(f"Название анализа слишком короткое (минимум {min_len} символа)")
        elif len(test_name) > max_len:
            errors.append(f"Название анализа слишком длинное (максимум {max_len} символов)")
        
        # Проверка на пустые символы
        if not test_name or test_name.isspace():
            errors.append("Название анализа пустое")
        
        # Проверка на стоп-слова
        stop_words = ["тест", "test", "анализ", "analysis", "показатель", "indicator"]
        if test_name.lower() in stop_words:
            warnings.append("Название анализа слишком общее")
            confidence = 0.3
        else:
            confidence = 0.8
        
        # Проверка на наличие цифр (обычно в названиях анализов цифр нет)
        if re.search(r'\d', test_name):
            warnings.append("Название анализа содержит цифры")
        
        # Нормализация названия
        cleaned_name = self.normalize_test_name(test_name)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={"test_name": cleaned_name},
            confidence=confidence
        )
    
    def validate_result(self, result: str) -> ValidationResult:
        """Валидация результата анализа"""
        errors = []
        warnings = []
        confidence = 0.0
        
        if not result or not isinstance(result, str):
            errors.append("Результат анализа отсутствует или имеет неверный тип")
            return ValidationResult(False, errors, warnings, None, 0.0)
        
        result = result.strip()
        
        if not result:
            errors.append("Результат анализа пустой")
            return ValidationResult(False, errors, warnings, None, 0.0)
        
        # Проверка числовых значений
        numeric_pattern = r'^[0-9]+\.?[0-9]*$'
        if re.match(numeric_pattern, result):
            # Проверка диапазона для медицинских значений
            try:
                value = float(result)
                if value < 0:
                    warnings.append("Отрицательное значение результата")
                elif value > 1000000:
                    warnings.append("Слишком большое значение результата")
                confidence = 0.9
            except ValueError:
                errors.append("Ошибка преобразования числового значения")
        
        # Проверка качественных значений
        elif result.lower() in ['отрицательно', 'положительно', 'negative', 'positive', 
                               'не обнаружено', 'обнаружено', 'норма', 'в норме']:
            confidence = 0.8
        
        # Проверка на пороговые значения
        elif result.startswith('<') or result.startswith('>'):
            number_part = result[1:].strip()
            if re.match(numeric_pattern, number_part):
                confidence = 0.85
            else:
                errors.append("Неверный формат порогового значения")
        
        # Проверка на диапазоны
        elif '-' in result or '—' in result:
            range_pattern = r'^[0-9]+\.?[0-9]*\s*[-—–]\s*[0-9]+\.?[0-9]*$'
            if re.match(range_pattern, result):
                confidence = 0.85
            else:
                warnings.append("Неверный формат диапазона")
        
        # Нормализация результата
        cleaned_result = self.normalize_result(result)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={"result": cleaned_result},
            confidence=confidence
        )
    
    def validate_date(self, date_str: str) -> ValidationResult:
        """Валидация даты"""
        errors = []
        warnings = []
        confidence = 0.0
        
        if not date_str or not isinstance(date_str, str):
            errors.append("Дата отсутствует или имеет неверный тип")
            return ValidationResult(False, errors, warnings, None, 0.0)
        
        date_str = date_str.strip()
        
        # Пробуем различные форматы даты
        date_formats = [
            '%d.%m.%Y',    # 01.01.2024
            '%Y-%m-%d',     # 2024-01-01
            '%d/%m/%Y',     # 01/01/2024
            '%d %B %Y',     # 01 января 2024
            '%d %b %Y',     # 01 Jan 2024
        ]
        
        parsed_date = None
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue
        
        if not parsed_date:
            errors.append("Неверный формат даты")
            return ValidationResult(False, errors, warnings, None, 0.0)
        
        # Проверка разумности даты
        today = date.today()
        min_date = date(1900, 1, 1)
        
        if parsed_date > today:
            warnings.append("Дата анализа в будущем")
        elif parsed_date < min_date:
            warnings.append("Дата анализа слишком давняя")
        
        # Проверка на слишком старые анализы
        days_diff = (today - parsed_date).days
        if days_diff > 365:
            warnings.append(f"Анализ выполнен более года назад ({days_diff} дней)")
        
        confidence = 0.9 if len(warnings) == 0 else 0.7
        
        # Форматируем дату в стандартный формат
        formatted_date = parsed_date.strftime('%d.%m.%Y')
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={"date": formatted_date, "date_obj": parsed_date},
            confidence=confidence
        )
    
    def validate_units(self, units: str) -> ValidationResult:
        """Валидация единиц измерения"""
        errors = []
        warnings = []
        confidence = 0.0
        
        if not units or not isinstance(units, str):
            # Единицы измерения могут отсутствовать - это не ошибка
            return ValidationResult(True, [], [], {"units": ""}, 0.5)
        
        units = units.strip()
        
        if not units:
            return ValidationResult(True, [], [], {"units": ""}, 0.5)
        
        # Проверка на стандартные медицинские единицы
        standard_units = [
            'мг/л', 'мЕд/л', 'нг/мл', 'ммоль/л', 'г/л', 'ед/л',
            'мм/ч', 'мкмоль/л', 'пг/мл', 'мкг/л', 'мг/дл',
            'mg/l', 'mIU/l', 'ng/ml', 'mmol/l', 'g/l', 'U/l',
            'mm/h', 'µmol/l', 'pg/ml', 'µg/l', 'mg/dl',
            '×10⁹/л', '×10³/л', '×10¹²/л'
        ]
        
        if units.lower() in [u.lower() for u in standard_units]:
            confidence = 0.9
        else:
            warnings.append("Нестандартная единица измерения")
            confidence = 0.6
        
        # Нормализация единиц измерения
        cleaned_units = self.normalize_units(units)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data={"units": cleaned_units},
            confidence=confidence
        )
    
    def validate_medical_test(self, test: Dict[str, Any]) -> ValidationResult:
        """Валидация полного медицинского теста"""
        errors = []
        warnings = []
        cleaned_data = {}
        total_confidence = 0.0
        field_count = 0
        
        # Проверка обязательных полей
        required_fields = self.quality_config.get("required_fields", ["test_name", "result"])
        for field in required_fields:
            if field not in test or not test[field]:
                errors.append(f"Отсутствует обязательное поле: {field}")
        
        # Валидация каждого поля
        field_validations = {
            "test_name": self.validate_test_name,
            "result": self.validate_result,
            "reference_values": self.validate_result,
            "units": self.validate_units,
            "test_date": self.validate_date
        }
        
        for field, validator in field_validations.items():
            if field in test and test[field]:
                field_result = validator(test[field])
                
                if field_result.errors:
                    errors.extend([f"{field}: {error}" for error in field_result.errors])
                
                if field_result.warnings:
                    warnings.extend([f"{field}: {warning}" for warning in field_result.warnings])
                
                if field_result.cleaned_data:
                    cleaned_data.update(field_result.cleaned_data)
                
                total_confidence += field_result.confidence
                field_count += 1
        
        # Дополнительные проверки
        if "test_name" in cleaned_data and "result" in cleaned_data:
            # Проверка на дубликаты
            if self.is_duplicate_test(cleaned_data["test_name"], cleaned_data["result"]):
                warnings.append("Возможный дубликат анализа")
            
            # Проверка аномальных комбинаций
            if self.is_abnormal_combination(cleaned_data["test_name"], cleaned_data["result"]):
                warnings.append("Аномальная комбинация названия и результата")
        
        # Расчет общей уверенности
        if field_count > 0:
            total_confidence /= field_count
        
        # Добавление категорий если есть
        if "category" in test:
            cleaned_data["category"] = test["category"]
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_data,
            confidence=total_confidence
        )
    
    def normalize_test_name(self, test_name: str) -> str:
        """Нормализация названия анализа"""
        if not test_name:
            return ""
        
        # Удаляем лишние пробелы
        normalized = re.sub(r'\s+', ' ', test_name.strip())
        
        # Приводим к стандартному виду аббревиатуры
        abbreviations = {
            'алт': 'АЛТ',
            'аст': 'АСТ',
            'ггт': 'ГГТ',
            'ттг': 'ТТГ',
            'иге': 'IgE',
            'игг': 'IgG',
            'игм': 'IgM',
            'ига': 'IgA'
        }
        
        for abbr, full in abbreviations.items():
            normalized = re.sub(rf'\b{abbr}\b', full, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def normalize_result(self, result: str) -> str:
        """Нормализация результата"""
        if not result:
            return ""
        
        normalized = result.strip()
        
        # Стандартизация разделителей
        normalized = re.sub(r'[—–]', '-', normalized)
        
        # Удаление лишних пробелов вокруг разделителей
        normalized = re.sub(r'\s*-\s*', '-', normalized)
        
        # Стандартизация пороговых значений
        normalized = re.sub(r'<\s*', '<', normalized)
        normalized = re.sub(r'>\s*', '>', normalized)
        
        return normalized
    
    def normalize_units(self, units: str) -> str:
        """Нормализация единиц измерения"""
        if not units:
            return ""
        
        normalized = units.strip()
        
        # Стандартизация символов
        replacements = {
            'мк': 'µ',
            'мкг': 'µg',
            'мкл': 'µL',
            'мл': 'ml',
            'л': 'L',
            'г': 'g',
            'мг': 'mg',
            'нг': 'ng',
            'пг': 'pg',
            'ммоль': 'mmol',
            'мкмоль': 'µmol',
            'мЕд': 'mIU',
            'ед': 'U'
        }
        
        for old, new in replacements.items():
            normalized = re.sub(rf'\b{old}\b', new, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def is_duplicate_test(self, test_name: str, result: str) -> bool:
        """Проверка на дубликат теста"""
        # Упрощенная проверка - в реальной системе нужно сравнивать с базой данных
        common_combinations = [
            ("Общий белок", "65"),
            ("Гемоглобин", "120"),
            ("Глюкоза", "5.0")
        ]
        
        for name, res in common_combinations:
            if name.lower() in test_name.lower() and res in result:
                return True
        
        return False
    
    def is_abnormal_combination(self, test_name: str, result: str) -> bool:
        """Проверка на аномальную комбинацию"""
        # Проверка на невозможные комбинации
        abnormal_patterns = [
            (r"гемоглобин", r"[a-zA-Z]"),  # Гемоглобин не может содержать буквы
            (r"сахар|глюкоза", r"[a-zA-Z]"),  # Глюкоза не может содержать буквы
            (r"hb.?sag", r"[0-9]"),  # HBsAg не может быть числом
        ]
        
        for pattern, invalid_result in abnormal_patterns:
            if re.search(pattern, test_name, re.IGNORECASE) and re.search(invalid_result, result):
                return True
        
        return False
    
    def batch_validate(self, tests: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Пакетная валидация"""
        results = []
        
        for test in tests:
            result = self.validate_medical_test(test)
            results.append(result)
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Получение сводки валидации"""
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        invalid = total - valid
        
        all_errors = []
        all_warnings = []
        avg_confidence = 0.0
        
        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            avg_confidence += result.confidence
        
        if total > 0:
            avg_confidence /= total
        
        return {
            "total_tests": total,
            "valid_tests": valid,
            "invalid_tests": invalid,
            "validity_rate": valid / total * 100 if total > 0 else 0,
            "average_confidence": avg_confidence,
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
            "errors": all_errors,
            "warnings": all_warnings
        }

# Глобальный экземпляр валидатора
medical_validator = MedicalDataValidator()
