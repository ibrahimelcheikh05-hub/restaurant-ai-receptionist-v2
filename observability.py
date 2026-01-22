"""
Observability
=============
Logging, metrics, and monitoring infrastructure.

Responsibilities:
- Structured logging
- Metrics collection
- Event tracking
- Health monitoring
- Alerting hooks
"""

import logging
import logging.config
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import json
import sys

# Structured logging format
STRUCTURED_LOG_FORMAT = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        },
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}


def setup_logging(
    level: str = "INFO",
    use_json: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Log level
        use_json: Use JSON formatting
        log_file: Optional log file path
    """
    # Basic config
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": level,
            "handlers": ["console"]
        }
    }
    
    # Add file handler if specified
    if log_file:
        log_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "standard",
            "filename": log_file,
            "maxBytes": 10485760,
            "backupCount": 5
        }
        log_config["root"]["handlers"].append("file")
    
    logging.config.dictConfig(log_config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {level} level")


class MetricsCollector:
    """
    Simple metrics collector.
    
    Collects metrics for monitoring.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
    
    def increment(self, metric: str, value: int = 1) -> None:
        """
        Increment counter.
        
        Args:
            metric: Metric name
            value: Increment value
        """
        if metric not in self._counters:
            self._counters[metric] = 0
        self._counters[metric] += value
    
    def set_gauge(self, metric: str, value: float) -> None:
        """
        Set gauge value.
        
        Args:
            metric: Metric name
            value: Gauge value
        """
        self._gauges[metric] = value
    
    def record_histogram(self, metric: str, value: float) -> None:
        """
        Record histogram value.
        
        Args:
            metric: Metric name
            value: Value to record
        """
        if metric not in self._histograms:
            self._histograms[metric] = []
        self._histograms[metric].append(value)
        
        # Keep last 1000 values
        if len(self._histograms[metric]) > 1000:
            self._histograms[metric] = self._histograms[metric][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "mean": sum(values) / len(values) if values else 0
                }
                for name, values in self._histograms.items()
            }
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


class EventLogger:
    """
    Event logging for important system events.
    
    Logs structured events for analysis.
    """
    
    def __init__(self, logger_name: str = "events"):
        """
        Initialize event logger.
        
        Args:
            logger_name: Logger name
        """
        self.logger = logging.getLogger(logger_name)
    
    def log_event(
        self,
        event_type: str,
        **kwargs
    ) -> None:
        """
        Log event.
        
        Args:
            event_type: Type of event
            **kwargs: Event data
        """
        event = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        
        self.logger.info(
            f"Event: {event_type}",
            extra=event
        )
    
    def log_call_started(self, call_id: str, **kwargs) -> None:
        """Log call started event."""
        self.log_event("call_started", call_id=call_id, **kwargs)
    
    def log_call_ended(
        self,
        call_id: str,
        duration: float,
        **kwargs
    ) -> None:
        """Log call ended event."""
        self.log_event(
            "call_ended",
            call_id=call_id,
            duration=duration,
            **kwargs
        )
    
    def log_order_created(self, order_id: str, **kwargs) -> None:
        """Log order created event."""
        self.log_event("order_created", order_id=order_id, **kwargs)
    
    def log_transfer(self, call_id: str, reason: str, **kwargs) -> None:
        """Log call transfer event."""
        self.log_event(
            "transfer",
            call_id=call_id,
            reason=reason,
            **kwargs
        )
    
    def log_error(self, error_type: str, **kwargs) -> None:
        """Log error event."""
        self.log_event("error", error_type=error_type, **kwargs)


class HealthMonitor:
    """
    System health monitoring.
    
    Tracks system health status.
    """
    
    def __init__(self):
        """Initialize health monitor."""
        self._healthy = True
        self._issues: Dict[str, str] = {}
        self._last_check = datetime.now(timezone.utc)
    
    def mark_healthy(self, component: str) -> None:
        """
        Mark component as healthy.
        
        Args:
            component: Component name
        """
        if component in self._issues:
            del self._issues[component]
        
        # Update overall health
        self._healthy = len(self._issues) == 0
        self._last_check = datetime.now(timezone.utc)
    
    def mark_unhealthy(self, component: str, reason: str) -> None:
        """
        Mark component as unhealthy.
        
        Args:
            component: Component name
            reason: Reason for unhealthy status
        """
        self._issues[component] = reason
        self._healthy = False
        self._last_check = datetime.now(timezone.utc)
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self._healthy
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get health status.
        
        Returns:
            Status dictionary
        """
        return {
            "healthy": self._healthy,
            "issues": self._issues.copy(),
            "last_check": self._last_check.isoformat()
        }


# Global instances
_metrics = MetricsCollector()
_events = EventLogger()
_health = HealthMonitor()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics


def get_events() -> EventLogger:
    """Get global event logger."""
    return _events


def get_health() -> HealthMonitor:
    """Get global health monitor."""
    return _health
