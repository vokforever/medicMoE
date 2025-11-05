"""
Система мониторинга и логирования обработки документов
"""
import logging
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from config.medical_config import medical_config

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Метрики обработки"""
    document_type: str
    processing_time: float
    success: bool
    extraction_method: str
    tests_extracted: int
    confidence_score: float
    error: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    tokens_used: Optional[int] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class SystemMetrics:
    """Системные метрики"""
    total_processed: int
    successful_processed: int
    failed_processed: int
    average_processing_time: float
    total_tests_extracted: int
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_updated: str
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

class ProcessingMonitor:
    """Мониторинг обработки документов"""
    
    def __init__(self, metrics_file: str = "data/processing_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics_history: List[ProcessingMetrics] = []
        self.session_start_time = time.time()
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {},
            "by_method": {},
            "total_tests": 0,
            "total_time": 0.0
        }
        self.load_metrics()
    
    def start_processing(self, document_type: str) -> str:
        """Начать обработку"""
        session_id = f"{document_type}_{int(time.time())}"
        logger.info(f"Начало обработки {document_type}, сессия: {session_id}")
        return session_id
    
    def end_processing(self, 
                    session_id: str, 
                    document_type: str, 
                    start_time: float,
                    success: bool,
                    extraction_method: str,
                    tests_count: int,
                    confidence_score: float = 0.0,
                    error: Optional[str] = None,
                    model_provider: Optional[str] = None,
                    model_name: Optional[str] = None,
                    tokens_used: Optional[int] = None) -> ProcessingMetrics:
        """Завершить обработку"""
        
        processing_time = time.time() - start_time
        
        metrics = ProcessingMetrics(
            document_type=document_type,
            processing_time=processing_time,
            success=success,
            extraction_method=extraction_method,
            tests_extracted=tests_count,
            confidence_score=confidence_score,
            error=error,
            model_provider=model_provider,
            model_name=model_name,
            tokens_used=tokens_used
        )
        
        # Сохраняем метрики
        self.metrics_history.append(metrics)
        self.update_processing_stats(metrics)
        
        # Логируем метрики
        status = "успешно" if success else "с ошибкой"
        logger.info(
            f"Обработка {document_type} завершена {status} за {processing_time:.2f}с, "
            f"извлечено тестов: {tests_count}, метод: {extraction_method}, "
            f"уверенность: {confidence_score:.2f}"
        )
        
        if error:
            logger.error(f"Ошибка обработки: {error}")
        
        # Сохраняем в файл
        self.save_metrics()
        
        return metrics
    
    def update_processing_stats(self, metrics: ProcessingMetrics):
        """Обновление статистики обработки"""
        self.processing_stats["total_processed"] += 1
        
        if metrics.success:
            self.processing_stats["successful"] += 1
        else:
            self.processing_stats["failed"] += 1
        
        # Статистика по типам документов
        doc_type = metrics.document_type
        if doc_type not in self.processing_stats["by_type"]:
            self.processing_stats["by_type"][doc_type] = {
                "total": 0, "successful": 0, "failed": 0, "avg_time": 0.0
            }
        
        type_stats = self.processing_stats["by_type"][doc_type]
        type_stats["total"] += 1
        if metrics.success:
            type_stats["successful"] += 1
        else:
            type_stats["failed"] += 1
        
        # Обновляем среднее время
        current_total = type_stats.get("total_time", 0.0) + metrics.processing_time
        type_stats["total_time"] = current_total
        type_stats["avg_time"] = current_total / type_stats["total"]
        
        # Статистика по методам извлечения
        method = metrics.extraction_method
        if method not in self.processing_stats["by_method"]:
            self.processing_stats["by_method"][method] = {
                "total": 0, "successful": 0, "avg_confidence": 0.0
            }
        
        method_stats = self.processing_stats["by_method"][method]
        method_stats["total"] += 1
        if metrics.success:
            method_stats["successful"] += 1
        
        # Обновляем среднюю уверенность
        current_conf_total = method_stats.get("total_confidence", 0.0) + metrics.confidence_score
        method_stats["total_confidence"] = current_conf_total
        method_stats["avg_confidence"] = current_conf_total / method_stats["total"]
        
        # Общая статистика
        self.processing_stats["total_tests"] += metrics.tests_extracted
        self.processing_stats["total_time"] += metrics.processing_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику"""
        if not self.metrics_history:
            return {
                "total_processed": 0,
                "success_rate": 0.0,
                "average_processing_time": 0.0,
                "average_tests_extracted": 0.0,
                "average_confidence": 0.0,
                "uptime_minutes": 0.0,
                "by_type": {},
                "by_method": {}
            }
        
        total = len(self.metrics_history)
        successful = sum(1 for m in self.metrics_history if m.success)
        failed = total - successful
        
        avg_time = sum(m.processing_time for m in self.metrics_history) / total
        avg_tests = sum(m.tests_extracted for m in self.metrics_history) / total
        avg_confidence = sum(m.confidence_score for m in self.metrics_history) / total
        
        uptime = time.time() - self.session_start_time
        
        return {
            "total_processed": total,
            "successful_processed": successful,
            "failed_processed": failed,
            "success_rate": successful / total * 100,
            "average_processing_time": avg_time,
            "average_tests_extracted": avg_tests,
            "average_confidence": avg_confidence,
            "uptime_minutes": uptime / 60,
            "by_type": self.processing_stats["by_type"],
            "by_method": self.processing_stats["by_method"],
            "last_updated": datetime.now().isoformat()
        }
    
    def get_recent_metrics(self, limit: int = 100) -> List[ProcessingMetrics]:
        """Получить последние метрики"""
        return self.metrics_history[-limit:]
    
    def get_metrics_by_type(self, document_type: str) -> List[ProcessingMetrics]:
        """Получить метрики по типу документа"""
        return [m for m in self.metrics_history if m.document_type == document_type]
    
    def get_metrics_by_method(self, extraction_method: str) -> List[ProcessingMetrics]:
        """Получить метрики по методу извлечения"""
        return [m for m in self.metrics_history if m.extraction_method == extraction_method]
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Получить сводку ошибок за последние часы"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [
            m for m in self.metrics_history 
            if not m.success and datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        error_counts = {}
        for error in recent_errors:
            if error.error:
                error_type = error.error.split(':')[0]  # Первая часть ошибки
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "error_types": error_counts,
            "error_rate": len(recent_errors) / max(1, len(self.get_recent_metrics())) * 100,
            "period_hours": hours
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Получить тренды производительности"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        if not recent_metrics:
            return {"trend": "no_data", "message": "Нет данных за указанный период"}
        
        # Группируем по часам
        hourly_stats = {}
        for metric in recent_metrics:
            hour = datetime.fromisoformat(metric.timestamp).hour
            if hour not in hourly_stats:
                hourly_stats[hour] = {
                    "count": 0, "successful": 0, "avg_time": 0.0, "total_time": 0.0
                }
            
            stats = hourly_stats[hour]
            stats["count"] += 1
            if metric.success:
                stats["successful"] += 1
            stats["total_time"] += metric.processing_time
            stats["avg_time"] = stats["total_time"] / stats["count"]
        
        # Определяем тренд
        if len(hourly_stats) < 2:
            return {"trend": "insufficient_data", "hourly_stats": hourly_stats}
        
        # Простая логика определения тренда
        success_rates = [
            stats["successful"] / stats["count"] * 100 
            for stats in hourly_stats.values()
        ]
        
        if len(success_rates) >= 2:
            if success_rates[-1] > success_rates[0]:
                trend = "improving"
            elif success_rates[-1] < success_rates[0]:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        return {
            "trend": trend,
            "success_rate_change": success_rates[-1] - success_rates[0] if len(success_rates) >= 2 else 0,
            "hourly_stats": hourly_stats
        }
    
    def save_metrics(self):
        """Сохранить метрики в файл"""
        try:
            # Создаем директорию если нужно
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем последние 1000 метрик
            metrics_to_save = self.metrics_history[-1000:]
            
            data = {
                "metrics": [asdict(m) for m in metrics_to_save],
                "statistics": self.get_statistics(),
                "processing_stats": self.processing_stats,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения метрик: {e}")
    
    def load_metrics(self):
        """Загрузить метрики из файла"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Восстанавливаем метрики
                metrics_data = data.get("metrics", [])
                self.metrics_history = [
                    ProcessingMetrics(**m) for m in metrics_data
                ]
                
                # Восстанавливаем статистику
                self.processing_stats = data.get("processing_stats", self.processing_stats)
                
                logger.info(f"Загружено {len(self.metrics_history)} метрик из файла")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки метрик: {e}")
    
    def clear_metrics(self):
        """Очистить метрики"""
        self.metrics_history.clear()
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {},
            "by_method": {},
            "total_tests": 0,
            "total_time": 0.0
        }
        
        # Удаляем файл метрик
        try:
            if self.metrics_file.exists():
                self.metrics_file.unlink()
                logger.info("Файл метрик удален")
        except Exception as e:
            logger.error(f"Ошибка удаления файла метрик: {e}")
    
    def export_metrics(self, export_path: str, format: str = "json"):
        """Экспорт метрик"""
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "statistics": self.get_statistics(),
                "processing_stats": self.processing_stats,
                "error_summary": self.get_error_summary(24),
                "performance_trends": self.get_performance_trends(24),
                "recent_metrics": [asdict(m) for m in self.get_recent_metrics(50)]
            }
            
            if format.lower() == "json":
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Метрики экспортированы в {export_path}")
            
        except Exception as e:
            logger.error(f"Ошибка экспорта метрик: {e}")

class HealthChecker:
    """Проверка здоровья системы"""
    
    def __init__(self, monitor: ProcessingMonitor):
        self.monitor = monitor
        self.health_checks = {
            "processing_rate": self._check_processing_rate,
            "error_rate": self._check_error_rate,
            "processing_time": self._check_processing_time,
            "confidence_score": self._check_confidence_score
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Комплексная проверка здоровья"""
        results = {}
        overall_health = "healthy"
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[check_name] = result
                
                if result["status"] == "critical":
                    overall_health = "critical"
                elif result["status"] == "warning" and overall_health != "critical":
                    overall_health = "warning"
                    
            except Exception as e:
                results[check_name] = {
                    "status": "error",
                    "message": f"Ошибка проверки: {e}"
                }
                overall_health = "critical"
        
        return {
            "overall_status": overall_health,
            "timestamp": datetime.now().isoformat(),
            "checks": results
        }
    
    def _check_processing_rate(self) -> Dict[str, Any]:
        """Проверка скорости обработки"""
        stats = self.monitor.get_statistics()
        total = stats["total_processed"]
        
        if total == 0:
            return {"status": "warning", "message": "Нет обработанных документов"}
        
        # Проверяем, что система обрабатывает документы
        recent_count = len(self.monitor.get_recent_metrics(10))
        
        if recent_count == 0:
            return {"status": "warning", "message": "Нет недавней активности"}
        
        return {"status": "healthy", "message": f"Обработано {total} документов"}
    
    def _check_error_rate(self) -> Dict[str, Any]:
        """Проверка частоты ошибок"""
        error_summary = self.monitor.get_error_summary(1)  # Последний час
        
        if error_summary["total_errors"] == 0:
            return {"status": "healthy", "message": "Ошибок нет"}
        
        error_rate = error_summary["error_rate"]
        
        if error_rate > 50:
            return {
                "status": "critical", 
                "message": f"Высокая частота ошибок: {error_rate:.1f}%"
            }
        elif error_rate > 20:
            return {
                "status": "warning", 
                "message": f"Повышенная частота ошибок: {error_rate:.1f}%"
            }
        
        return {"status": "healthy", "message": f"Нормальная частота ошибок: {error_rate:.1f}%"}
    
    def _check_processing_time(self) -> Dict[str, Any]:
        """Проверка времени обработки"""
        stats = self.monitor.get_statistics()
        avg_time = stats["average_processing_time"]
        
        if avg_time == 0:
            return {"status": "warning", "message": "Нет данных о времени обработки"}
        
        if avg_time > 30:
            return {
                "status": "warning", 
                "message": f"Медленная обработка: {avg_time:.1f}с"
            }
        elif avg_time > 60:
            return {
                "status": "critical", 
                "message": f"Очень медленная обработка: {avg_time:.1f}с"
            }
        
        return {"status": "healthy", "message": f"Нормальное время обработки: {avg_time:.1f}с"}
    
    def _check_confidence_score(self) -> Dict[str, Any]:
        """Проверка уверенности извлечения"""
        stats = self.monitor.get_statistics()
        avg_confidence = stats["average_confidence"]
        
        if avg_confidence == 0:
            return {"status": "warning", "message": "Нет данных об уверенности"}
        
        if avg_confidence < 0.5:
            return {
                "status": "critical", 
                "message": f"Низкая уверенность извлечения: {avg_confidence:.2f}"
            }
        elif avg_confidence < 0.7:
            return {
                "status": "warning", 
                "message": f"Средняя уверенность извлечения: {avg_confidence:.2f}"
            }
        
        return {"status": "healthy", "message": f"Хорошая уверенность извлечения: {avg_confidence:.2f}"}

# Глобальные экземпляры
processing_monitor = ProcessingMonitor()
health_checker = HealthChecker(processing_monitor)
