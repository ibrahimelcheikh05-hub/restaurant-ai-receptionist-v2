"""
Latency Tracker
===============
Latency measurement and reporting.

Responsibilities:
- Track operation latencies
- Measure end-to-end timings
- Identify bottlenecks
- Generate latency reports
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class LatencyMetric(Enum):
    """Types of latency metrics."""
    STT_LATENCY = "stt_latency"
    LLM_LATENCY = "llm_latency"
    TTS_LATENCY = "tts_latency"
    TRANSLATION_LATENCY = "translation_latency"
    DATABASE_LATENCY = "database_latency"
    TOTAL_RESPONSE = "total_response"
    CALL_DURATION = "call_duration"


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    
    metric: LatencyMetric
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class LatencyTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str):
        """
        Initialize timer.
        
        Args:
            name: Operation name
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer."""
        self.end_time = time.perf_counter()
        return False
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.perf_counter()
        return (end - self.start_time) * 1000


class LatencyTracker:
    """
    Tracks latencies for operations.
    
    Features:
    - Per-call latency tracking
    - Aggregate statistics
    - Percentile calculations
    - Bottleneck identification
    """
    
    def __init__(self, call_id: str):
        """
        Initialize tracker.
        
        Args:
            call_id: Call identifier
        """
        self.call_id = call_id
        
        # Measurements
        self._measurements: List[LatencyMeasurement] = []
        
        # Timers
        self._active_timers: Dict[str, LatencyTimer] = {}
        
        # Start time
        self._call_start = datetime.now(timezone.utc)
        
        logger.debug(
            "LatencyTracker created",
            extra={"call_id": call_id}
        )
    
    def start_timer(self, name: str) -> LatencyTimer:
        """
        Start named timer.
        
        Args:
            name: Timer name
            
        Returns:
            Timer instance
        """
        timer = LatencyTimer(name)
        timer.__enter__()
        self._active_timers[name] = timer
        return timer
    
    def stop_timer(self, name: str) -> Optional[float]:
        """
        Stop named timer.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed milliseconds or None
        """
        if name not in self._active_timers:
            logger.warning(
                f"Timer not found: {name}",
                extra={"call_id": self.call_id}
            )
            return None
        
        timer = self._active_timers[name]
        timer.__exit__(None, None, None)
        
        del self._active_timers[name]
        
        return timer.elapsed_ms
    
    def record(
        self,
        metric: LatencyMetric,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record latency measurement.
        
        Args:
            metric: Metric type
            latency_ms: Latency in milliseconds
            metadata: Optional metadata
        """
        measurement = LatencyMeasurement(
            metric=metric,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )
        
        self._measurements.append(measurement)
        
        logger.debug(
            f"{metric.value}: {latency_ms:.2f}ms",
            extra={"call_id": self.call_id}
        )
    
    def record_stt(self, latency_ms: float) -> None:
        """Record STT latency."""
        self.record(LatencyMetric.STT_LATENCY, latency_ms)
    
    def record_llm(self, latency_ms: float, tokens: Optional[int] = None) -> None:
        """Record LLM latency."""
        metadata = {"tokens": tokens} if tokens else {}
        self.record(LatencyMetric.LLM_LATENCY, latency_ms, metadata)
    
    def record_tts(self, latency_ms: float) -> None:
        """Record TTS latency."""
        self.record(LatencyMetric.TTS_LATENCY, latency_ms)
    
    def record_translation(self, latency_ms: float) -> None:
        """Record translation latency."""
        self.record(LatencyMetric.TRANSLATION_LATENCY, latency_ms)
    
    def record_database(self, latency_ms: float, query: Optional[str] = None) -> None:
        """Record database latency."""
        metadata = {"query": query} if query else {}
        self.record(LatencyMetric.DATABASE_LATENCY, latency_ms, metadata)
    
    def record_total_response(self, latency_ms: float) -> None:
        """Record total response time."""
        self.record(LatencyMetric.TOTAL_RESPONSE, latency_ms)
    
    def get_measurements(
        self,
        metric: Optional[LatencyMetric] = None
    ) -> List[LatencyMeasurement]:
        """
        Get measurements.
        
        Args:
            metric: Filter by metric type
            
        Returns:
            List of measurements
        """
        if metric is None:
            return self._measurements.copy()
        
        return [
            m for m in self._measurements
            if m.metric == metric
        ]
    
    def get_average(self, metric: LatencyMetric) -> Optional[float]:
        """
        Get average latency for metric.
        
        Args:
            metric: Metric type
            
        Returns:
            Average latency in ms or None
        """
        measurements = self.get_measurements(metric)
        if not measurements:
            return None
        
        return sum(m.latency_ms for m in measurements) / len(measurements)
    
    def get_percentile(
        self,
        metric: LatencyMetric,
        percentile: float
    ) -> Optional[float]:
        """
        Get percentile latency.
        
        Args:
            metric: Metric type
            percentile: Percentile (0-100)
            
        Returns:
            Percentile latency or None
        """
        measurements = self.get_measurements(metric)
        if not measurements:
            return None
        
        sorted_latencies = sorted(m.latency_ms for m in measurements)
        index = int(len(sorted_latencies) * (percentile / 100))
        index = min(index, len(sorted_latencies) - 1)
        
        return sorted_latencies[index]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get latency summary.
        
        Returns:
            Summary dict with statistics
        """
        summary = {
            "call_id": self.call_id,
            "call_duration_seconds": (
                datetime.now(timezone.utc) - self._call_start
            ).total_seconds(),
            "measurement_count": len(self._measurements),
            "metrics": {}
        }
        
        # Add stats for each metric type
        for metric in LatencyMetric:
            measurements = self.get_measurements(metric)
            if not measurements:
                continue
            
            latencies = [m.latency_ms for m in measurements]
            
            summary["metrics"][metric.value] = {
                "count": len(latencies),
                "avg_ms": sum(latencies) / len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "p50_ms": self.get_percentile(metric, 50),
                "p95_ms": self.get_percentile(metric, 95),
                "p99_ms": self.get_percentile(metric, 99)
            }
        
        return summary
    
    def identify_bottlenecks(
        self,
        threshold_ms: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """
        Identify operations exceeding threshold.
        
        Args:
            threshold_ms: Threshold in milliseconds
            
        Returns:
            List of bottlenecks
        """
        bottlenecks = []
        
        for measurement in self._measurements:
            if measurement.latency_ms > threshold_ms:
                bottlenecks.append({
                    "metric": measurement.metric.value,
                    "latency_ms": measurement.latency_ms,
                    "threshold_ms": threshold_ms,
                    "exceeded_by_ms": measurement.latency_ms - threshold_ms,
                    "timestamp": measurement.timestamp.isoformat()
                })
        
        return bottlenecks
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Full tracker data
        """
        return {
            "call_id": self.call_id,
            "call_start": self._call_start.isoformat(),
            "measurements": [m.to_dict() for m in self._measurements],
            "summary": self.get_summary()
        }


# Global tracker storage
_trackers: Dict[str, LatencyTracker] = {}


def get_tracker(call_id: str) -> LatencyTracker:
    """
    Get or create latency tracker.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Latency tracker
    """
    if call_id not in _trackers:
        _trackers[call_id] = LatencyTracker(call_id)
    
    return _trackers[call_id]


def remove_tracker(call_id: str) -> None:
    """
    Remove latency tracker.
    
    Args:
        call_id: Call identifier
    """
    if call_id in _trackers:
        del _trackers[call_id]
