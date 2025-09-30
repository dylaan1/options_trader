"""Client and data fetcher for the Schwab Developer API."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


@dataclass
class SchwabAPIConfig:
    """Configuration bundle for the Schwab Developer API."""

    base_url: str
    access_token: str
    account_id: Optional[str] = None
    timeout: float = 10.0
    rate_limit_sleep: float = 0.2

    def headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers


class SchwabAPIClient:
    """Minimal Schwab API client with retry-aware GET helper."""

    def __init__(
        self,
        config: SchwabAPIConfig,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self._session = session or requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        expected_status: Sequence[int] = (200,),
    ) -> Optional[Dict[str, Any]]:
        url = urljoin(self.config.base_url.rstrip("/") + "/", path.lstrip("/"))
        try:
            response = self._session.request(
                method.upper(),
                url,
                headers=self.config.headers(),
                params=params,
                json=data,
                timeout=self.config.timeout,
            )
        except requests.RequestException as exc:
            logger.warning("Schwab API request failed: %s", exc, exc_info=exc)
            return None

        if response.status_code not in expected_status:
            logger.warning(
                "Schwab API unexpected status %s for %s", response.status_code, url
            )
            return None

        if not response.text:
            return None

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON from Schwab API: %s", exc, exc_info=exc)
            return None

    # ------------------------------------------------------------------
    # Public endpoint helpers
    # ------------------------------------------------------------------
    def get_vol_surface(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve implied volatility surface for an underlier."""

        path = f"/markets/options/{symbol}/volatility"
        return self._request("GET", path)

    def get_rate_curve(self) -> Optional[Dict[str, Any]]:
        """Retrieve interest-rate curve data."""

        path = "/markets/rates"
        return self._request("GET", path)

    def get_dividend_yield(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve dividend yield data for an equity symbol."""

        path = f"/markets/stocks/{symbol}/dividends"
        return self._request("GET", path)

    def get_market_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve latest market snapshot for an underlier."""

        path = f"/markets/quotes/{symbol}"
        return self._request("GET", path)

    def get_account_values(self) -> Optional[Dict[str, Any]]:
        """Retrieve account balances and risk metrics."""

        if not self.config.account_id:
            logger.warning("Account ID missing for Schwab account values request")
            return None
        path = f"/accounts/{self.config.account_id}/balances"
        return self._request("GET", path)


@dataclass
class CalibrationData:
    """Container for calibration inputs and missing-data indicators."""

    vol_surface: Optional[Dict[str, Any]] = None
    rates_curve: Optional[Dict[str, Any]] = None
    dividend_yield: Optional[Dict[str, Any]] = None
    market_snapshot: Optional[Dict[str, Any]] = None
    account_values: Optional[Dict[str, Any]] = None
    as_of: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    missing_fields: List[str] = field(default_factory=list)

    def mark_missing(self, field: str) -> None:
        if field not in self.missing_fields:
            self.missing_fields.append(field)

    def has_missing(self) -> bool:
        return bool(self.missing_fields)


class CalibrationDataFetcher:
    """High-level orchestration around Schwab API data fetching."""

    def __init__(self, client: SchwabAPIClient) -> None:
        self.client = client

    def fetch_all(self, symbol: str) -> CalibrationData:
        data = CalibrationData()
        data.vol_surface = self._safe_fetch("vol_surface", lambda: self.client.get_vol_surface(symbol), data)
        data.rates_curve = self._safe_fetch("rates_curve", self.client.get_rate_curve, data)
        data.dividend_yield = self._safe_fetch(
            "dividend_yield", lambda: self.client.get_dividend_yield(symbol), data
        )
        data.market_snapshot = self._safe_fetch(
            "market_snapshot", lambda: self.client.get_market_snapshot(symbol), data
        )
        data.account_values = self._safe_fetch("account_values", self.client.get_account_values, data)
        return data

    def retrieve_missing(self, symbol: str, data: CalibrationData) -> CalibrationData:
        """Attempt to refill any missing calibration data from the API."""

        missing = list(data.missing_fields)
        data.missing_fields.clear()
        for field in missing:
            fetch_map: Dict[str, Callable[[], Optional[Dict[str, Any]]]] = {
                "vol_surface": lambda: self.client.get_vol_surface(symbol),
                "rates_curve": self.client.get_rate_curve,
                "dividend_yield": lambda: self.client.get_dividend_yield(symbol),
                "market_snapshot": lambda: self.client.get_market_snapshot(symbol),
                "account_values": self.client.get_account_values,
            }
            fetcher = fetch_map.get(field)
            if not fetcher:
                continue
            result = fetcher()
            if result is None:
                data.mark_missing(field)
                continue
            setattr(data, field, result)
        return data

    def _safe_fetch(
        self,
        field: str,
        fetcher: Callable[[], Optional[Dict[str, Any]]],
        data: CalibrationData,
    ) -> Optional[Dict[str, Any]]:
        try:
            result = fetcher()
        except Exception as exc:  # pragma: no cover - defensive catch
            logger.exception("Unexpected error fetching %s: %s", field, exc)
            result = None
        if result is None:
            data.mark_missing(field)
        return result
