"""Calibration engine orchestrating Schwab data and model parameter estimation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from pricing.heston_jump_diffusion import HestonParams, JumpParams, MarketParams

from .schwab_client import CalibrationData

logger = logging.getLogger(__name__)

PACIFIC_TZ = "America/Los_Angeles"


class MissingDataError(RuntimeError):
    """Raised when required calibration inputs are unavailable."""

    def __init__(self, fields: Sequence[str]):
        super().__init__(f"Missing calibration data: {', '.join(fields)}")
        self.fields = list(fields)


@dataclass
class CalibrationResult:
    """Outputs of the calibration process."""

    as_of: datetime
    market_params: MarketParams
    heston_params: HestonParams
    jump_params: JumpParams
    missing_fields: List[str] = field(default_factory=list)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "market_params": {
                "spot": self.market_params.spot,
                "rate": self.market_params.rate,
                "dividend": self.market_params.dividend,
            },
            "heston_params": {
                "kappa": self.heston_params.kappa,
                "theta": self.heston_params.theta,
                "sigma": self.heston_params.sigma,
                "rho": self.heston_params.rho,
                "v0": self.heston_params.v0,
            },
            "jump_params": {
                "intensity": self.jump_params.intensity,
                "mean": self.jump_params.mean,
                "std": self.jump_params.std,
            },
            "missing_fields": list(self.missing_fields),
            "source_metadata": self.source_metadata,
        }


class CalibrationStore(Protocol):
    """Persistence interface for calibration results."""

    def save(self, result: CalibrationResult) -> None:
        ...

    def load_latest(self) -> Optional[CalibrationResult]:  # pragma: no cover - interface
        ...


class JsonCalibrationStore:
    """Simple filesystem-backed storage for calibration results."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: CalibrationResult) -> None:
        timestamp = result.as_of.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_path = self.directory / f"calibration_{timestamp}.json"
        file_path.write_text(json.dumps(result.to_dict(), indent=2))

    def load_latest(self) -> Optional[CalibrationResult]:
        files = sorted(self.directory.glob("calibration_*.json"))
        if not files:
            return None
        latest = files[-1]
        payload = json.loads(latest.read_text())
        as_of = datetime.fromisoformat(payload["as_of"])
        market = MarketParams(**payload["market_params"])
        heston = HestonParams(**payload["heston_params"])
        jump = JumpParams(**payload["jump_params"])
        return CalibrationResult(
            as_of=as_of,
            market_params=market,
            heston_params=heston,
            jump_params=jump,
            missing_fields=payload.get("missing_fields", []),
            source_metadata=payload.get("source_metadata", {}),
        )


@dataclass
class CalibrationEngine:
    """Estimate Heston and jump parameters from Schwab market data."""

    store: CalibrationStore
    require_complete: bool = True

    def calibrate(self, data: CalibrationData, *, symbol: str) -> CalibrationResult:
        if data.vol_surface is None:
            missing = ["vol_surface"]
            if self.require_complete:
                raise MissingDataError(missing)
            logger.warning("Vol surface missing; using previous calibration fallback if available")
        market_params = self._calibrate_market_params(data, symbol)
        heston_params = self._calibrate_heston_params(data)
        jump_params = self._calibrate_jump_params(data)
        result = CalibrationResult(
            as_of=data.as_of,
            market_params=market_params,
            heston_params=heston_params,
            jump_params=jump_params,
            missing_fields=list(data.missing_fields),
            source_metadata={
                "symbol": symbol,
                "vol_surface_keys": list((data.vol_surface or {}).keys()),
                "rates_curve_keys": list((data.rates_curve or {}).keys()),
                "dividend_fields": list((data.dividend_yield or {}).keys()) if isinstance(data.dividend_yield, dict) else [],
            },
        )
        self.store.save(result)
        return result

    def _calibrate_market_params(self, data: CalibrationData, symbol: str) -> MarketParams:
        spot = self._extract_float(data.market_snapshot, ["lastPrice", "price", "close"], default=0.0)
        rate = self._extract_curve_rate(data.rates_curve)
        dividend = self._extract_float(data.dividend_yield, ["indicatedYield", "yield"], default=0.0)
        if spot <= 0:
            fallback = self.store.load_latest()
            if fallback:
                logger.warning("Using spot fallback from previous calibration for %s", symbol)
                spot = fallback.market_params.spot
        return MarketParams(spot=spot, rate=rate, dividend=dividend)

    def _calibrate_heston_params(self, data: CalibrationData) -> HestonParams:
        vol_surface = data.vol_surface or {}
        vols = self._extract_all_floats(vol_surface)
        if not vols:
            fallback = self.store.load_latest()
            if fallback:
                logger.warning("Missing vol data; falling back to previous Heston params")
                return fallback.heston_params
            raise MissingDataError(["vol_surface"])
        avg_vol = float(np.mean(vols))
        var_vol = float(np.var(vols))
        kappa = max(0.1, 2.0 * avg_vol)
        theta = max(1e-6, avg_vol**2)
        sigma = max(1e-6, np.sqrt(var_vol) if var_vol > 0 else 0.5 * avg_vol)
        rho = float(np.clip(-0.5 * np.sign(theta - avg_vol**2), -0.95, 0.95))
        v0 = theta
        return HestonParams(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)

    def _calibrate_jump_params(self, data: CalibrationData) -> JumpParams:
        snapshot = data.market_snapshot or {}
        price_changes = []
        intraday = snapshot.get("intraday")
        if isinstance(intraday, Iterable):
            for point in intraday:
                if not isinstance(point, dict):
                    continue
                change = self._extract_float(point, ["change", "delta"], default=None)
                if change is not None:
                    price_changes.append(change)
        if price_changes:
            jumps = np.array(price_changes)
            intensity = float(np.mean(np.abs(jumps) > 0.02))  # heuristic threshold
            mean = float(np.mean(jumps))
            std = float(np.std(jumps))
        else:
            fallback = self.store.load_latest()
            if fallback:
                logger.warning("Missing jump data; falling back to previous jump params")
                return fallback.jump_params
            intensity = 0.0
            mean = 0.0
            std = 0.01
        return JumpParams(intensity=intensity, mean=mean, std=max(std, 1e-4))

    def _extract_curve_rate(self, rates_curve: Optional[Dict[str, Any]]) -> float:
        if not rates_curve:
            fallback = self.store.load_latest()
            if fallback:
                logger.warning("Missing rates data; using fallback rate")
                return fallback.market_params.rate
            return 0.0
        rates = self._extract_all_floats(rates_curve)
        if not rates:
            return 0.0
        return float(np.mean(rates))

    def _extract_all_floats(self, container: Any) -> List[float]:
        values: List[float] = []
        if isinstance(container, dict):
            for value in container.values():
                values.extend(self._extract_all_floats(value))
        elif isinstance(container, Iterable) and not isinstance(container, (str, bytes)):
            for item in container:
                values.extend(self._extract_all_floats(item))
        else:
            maybe = self._to_float(container)
            if maybe is not None:
                values.append(maybe)
        return values

    def _extract_float(
        self,
        container: Optional[Any],
        keys: Sequence[str],
        *,
        default: Optional[float],
    ) -> Optional[float]:
        if not isinstance(container, dict):
            return default
        for key in keys:
            if key in container:
                maybe = self._to_float(container[key])
                if maybe is not None:
                    return maybe
        return default

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class CalibrationScheduler(Protocol):
    """Protocol for scheduling calibration runs."""

    def next_run(self, now: Optional[datetime] = None) -> datetime:
        ...

    def should_run(self, now: Optional[datetime] = None) -> bool:
        ...


@dataclass
class DailyCalibrationScheduler:
    """Schedule calibration at a fixed Pacific Time each day."""

    run_time: time = time(23, 50)
    timezone_name: str = PACIFIC_TZ
    last_run: Optional[datetime] = None

    def next_run(self, now: Optional[datetime] = None) -> datetime:
        tz = self._timezone()
        if now is None:
            now = datetime.now(tz)
        else:
            now = now.astimezone(tz)
        target = datetime.combine(now.date(), self.run_time, tzinfo=tz)
        if now >= target:
            target += timedelta(days=1)
        return target

    def should_run(self, now: Optional[datetime] = None) -> bool:
        tz = self._timezone()
        current = datetime.now(tz) if now is None else now.astimezone(tz)
        next_run_time = self.next_run(current)
        if self.last_run and self.last_run >= next_run_time:
            return False
        if current >= next_run_time:
            self.last_run = current
            return True
        return False

    def _timezone(self) -> tzinfo:
        if self.timezone_name == "UTC":
            return timezone.utc
        try:
            from zoneinfo import ZoneInfo
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("zoneinfo is required for timezone calculations") from exc
        return ZoneInfo(self.timezone_name)
