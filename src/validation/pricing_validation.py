"""Framework for pricing validation and regression testing."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from statistics import fmean
import math
import datetime as dt


class ToleranceMode(str, Enum):
    """Enumeration describing how pricing errors should be interpreted."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass
class PricingTolerance:
    """Container for tolerance configuration with runtime mutability."""

    absolute_threshold: float = 0.05
    relative_threshold: float = 0.01
    mode: ToleranceMode = ToleranceMode.ABSOLUTE

    def __post_init__(self) -> None:
        if self.absolute_threshold < 0:
            raise ValueError("absolute_threshold must be non-negative")
        if self.relative_threshold < 0:
            raise ValueError("relative_threshold must be non-negative")

    def use_absolute(self, threshold: Optional[float] = None) -> None:
        """Switch to absolute tolerance with an optional new threshold."""

        if threshold is not None:
            if threshold < 0:
                raise ValueError("Absolute tolerance cannot be negative")
            self.absolute_threshold = threshold
        self.mode = ToleranceMode.ABSOLUTE

    def use_relative(self, threshold: Optional[float] = None) -> None:
        """Switch to relative tolerance with an optional new threshold."""

        if threshold is not None:
            if threshold < 0:
                raise ValueError("Relative tolerance cannot be negative")
            self.relative_threshold = threshold
        self.mode = ToleranceMode.RELATIVE

    def evaluate(self, model: float, benchmark: float) -> Tuple[float, float, bool]:
        """Return (error, tolerance_value, pass?)."""

        error = model - benchmark
        tolerance_value: float
        if self.mode is ToleranceMode.ABSOLUTE:
            tolerance_value = self.absolute_threshold
        else:
            ref = abs(benchmark) if benchmark != 0 else 1.0
            tolerance_value = self.relative_threshold * ref
        passed = abs(error) <= tolerance_value
        return error, tolerance_value, passed


@dataclass(frozen=True)
class ValidationObservation:
    """Represents a single pricing comparison against a benchmark."""

    symbol: str
    option_id: str
    model_price: float
    benchmark_price: float
    confidence_interval: Optional[Tuple[float, float]] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def deviation(self) -> float:
        """Return the signed deviation between model and benchmark."""

        return self.model_price - self.benchmark_price

    def deviation_magnitude(self) -> float:
        return abs(self.deviation())


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a single observation."""

    symbol: str
    option_id: str
    mode: ToleranceMode
    error: float
    tolerance: float
    passed: bool
    model_price: float
    benchmark_price: float
    confidence_interval: Optional[Tuple[float, float]] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationSchedule:
    """Describes when validation runs should execute."""

    requested_times: Sequence[dt.time]
    recommended_times: Sequence[dt.time]
    timezone: dt.tzinfo

    def merged_schedule(self) -> List[dt.time]:
        """Return the union of requested and recommended run times."""

        seen: Dict[Tuple[int, int, int], dt.time] = {}
        for time_list in (self.requested_times, self.recommended_times):
            for t in time_list:
                seen[(t.hour, t.minute, t.second)] = t
        return list(seen.values())


PRIORITY_SYMBOLS: Tuple[str, ...] = (
    "SPX",
    "SPY",
    "NDX",
    "QQQ",
    "TSLA",
    "AAPL",
    "GOOGL",
    "META",
    "PLTR",
    "ORCL",
)

DEFAULT_REQUESTED_TIMES = (
    dt.time(hour=6, minute=45),  # pre-market calibration check
    dt.time(hour=12, minute=0),  # mid-session validation
    dt.time(hour=15, minute=30),  # pre-close audit
)

RECOMMENDED_TIMES = (
    dt.time(hour=6, minute=45),
    dt.time(hour=10, minute=30),
    dt.time(hour=12, minute=0),
    dt.time(hour=15, minute=30),
    dt.time(hour=16, minute=15),  # post-close reconciliation
)


class PricingValidationHarness:
    """Coordinates validation runs and prioritisation logic."""

    def __init__(
        self,
        tolerance: Optional[PricingTolerance] = None,
        prioritized_symbols: Sequence[str] = PRIORITY_SYMBOLS,
        top_deviation_count: int = 50,
        timezone: dt.tzinfo = dt.timezone(dt.timedelta(hours=-8)),
    ) -> None:
        if top_deviation_count < 0:
            raise ValueError("top_deviation_count must be non-negative")
        self.tolerance = tolerance or PricingTolerance()
        self.prioritized_symbols = tuple(dict.fromkeys(prioritized_symbols))
        self.top_deviation_count = top_deviation_count
        self.timezone = timezone

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_targets(self, observations: Iterable[ValidationObservation]) -> List[ValidationObservation]:
        """Return observations limited to the prioritised validation universe."""

        prioritized: Dict[str, ValidationObservation] = {}
        others: List[ValidationObservation] = []
        for obs in observations:
            if math.isnan(obs.model_price) or math.isnan(obs.benchmark_price):
                continue
            key = obs.symbol.upper()
            if key in self.prioritized_symbols and key not in prioritized:
                prioritized[key] = obs
            else:
                others.append(obs)

        others.sort(key=lambda o: o.deviation_magnitude(), reverse=True)
        selected_others = others[: self.top_deviation_count]
        return list(prioritized.values()) + selected_others

    def validate(self, observations: Iterable[ValidationObservation]) -> List[ValidationResult]:
        """Validate observations and return pass/fail diagnostics."""

        results: List[ValidationResult] = []
        for obs in observations:
            if math.isnan(obs.model_price) or math.isnan(obs.benchmark_price):
                continue
            error, tolerance_value, passed = self.tolerance.evaluate(
                obs.model_price, obs.benchmark_price
            )
            results.append(
                ValidationResult(
                    symbol=obs.symbol,
                    option_id=obs.option_id,
                    mode=self.tolerance.mode,
                    error=error,
                    tolerance=tolerance_value,
                    passed=passed,
                    model_price=obs.model_price,
                    benchmark_price=obs.benchmark_price,
                    confidence_interval=obs.confidence_interval,
                    diagnostics=obs.diagnostics,
                )
            )
        return results

    def failure_rate(self, results: Sequence[ValidationResult]) -> float:
        """Return the share of validations that failed."""

        if not results:
            return 0.0
        failures = sum(1 for result in results if not result.passed)
        return failures / len(results)

    def aggregate_error(self, results: Sequence[ValidationResult]) -> Dict[str, float]:
        """Summarise error statistics for reporting."""

        if not results:
            return {"mean_error": 0.0, "rmse": 0.0}
        errors = [result.error for result in results]
        return {
            "mean_error": fmean(errors),
            "rmse": math.sqrt(fmean([err ** 2 for err in errors])),
        }

    def schedule(self) -> ValidationSchedule:
        """Return both the requested cadence and an enhanced recommended set."""

        return ValidationSchedule(
            requested_times=DEFAULT_REQUESTED_TIMES,
            recommended_times=RECOMMENDED_TIMES,
            timezone=self.timezone,
        )

