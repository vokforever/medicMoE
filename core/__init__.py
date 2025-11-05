"""
Ядро системы обработки медицинских документов
"""

from .universal_processor import universal_processor, ProcessingResult
from .validators import medical_validator, ValidationResult
from .monitoring import processing_monitor, health_checker
from .bot_handlers import create_bot_handlers

__all__ = [
    'universal_processor',
    'ProcessingResult',
    'medical_validator',
    'ValidationResult',
    'processing_monitor',
    'health_checker',
    'create_bot_handlers'
]
