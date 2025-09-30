"""Monte Carlo engine for options using a Heston model with jump diffusion."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np

SECONDS_IN_YEAR = 365.0 * 24 * 60 * 60
Z_SCORE_99 = 2.576
OptionType = Literal["call", "put"]
OptionStyle = Literal["european", "american"]
UnderlyingType = Literal["equity", "index", "vix"]


@dataclass(frozen=True)
class HestonParams:
    """Parameters describing the Heston stochastic volatility process."""

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float


@dataclass(frozen=True)
class JumpParams:
    """Parameters for the jump component of the price process."""

    intensity: float
    mean: float
    std: float

    def drift_adjustment(self) -> float:
        """Return the compensator for the jump process under risk-neutral measure."""

        return np.exp(self.mean + 0.5 * self.std ** 2) - 1.0


@dataclass(frozen=True)
class MarketParams:
    """Market configuration for the Monte Carlo simulation."""

    spot: float
    rate: float
    dividend: float = 0.0


@dataclass(frozen=True)
class OptionContract:
    """Describes the option being priced."""

    strike: float
    maturity: float  # expressed in years
    option_type: OptionType
    style: OptionStyle
    underlying_type: UnderlyingType = "equity"


def _payoff(option_type: OptionType, prices: np.ndarray, strike: float) -> np.ndarray:
    if option_type == "call":
        return np.maximum(prices - strike, 0.0)
    return np.maximum(strike - prices, 0.0)


class HestonJumpDiffusionMonteCarlo:
    """Monte Carlo pricer supporting European and American options."""

    def __init__(
        self,
        heston: HestonParams,
        jumps: JumpParams,
        market: MarketParams,
        n_paths: int = 5000,
        dt_seconds: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        self.heston = heston
        self.jumps = jumps
        self.market = market
        self.dt_seconds = float(dt_seconds)
        self.n_paths = int(n_paths)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def price(self, contract: OptionContract) -> Dict[str, float]:
        """Price the option and compute Greeks and diagnostics."""

        base_state = copy.deepcopy(self._rng.bit_generator.state)
        base_result = self._price_core(
            contract, self.market, self.heston, rng_state=base_state
        )
        diagnostics: Dict[str, float] = {
            "fair_value_price": base_result["fair_value"],
            "std_error": base_result["std_error"],
            "payoff_variance": base_result["std_error"] ** 2 * base_result["num_samples"],
            "ci_low": base_result["ci_low"],
            "ci_high": base_result["ci_high"],
            "mean_payoff": base_result["mean_payoff"],
            "discount_factor": base_result["discount_factor"],
            "num_paths": float(self.n_paths),
            "time_steps": float(base_result["time_steps"]),
        }
        if base_result.get("exercise_probability") is not None:
            diagnostics["exercise_probability"] = base_result["exercise_probability"]

        delta, gamma = self._spot_greeks(
            contract, base_result["fair_value"], base_state
        )
        vega = self._vega(contract, base_state)
        theta = self._theta(contract, base_result["fair_value"], base_state)

        diagnostics.update({
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
        })

        return diagnostics

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    def _price_core(
        self,
        contract: OptionContract,
        market: MarketParams,
        heston: HestonParams,
        rng_state: Optional[dict] = None,
    ) -> Dict[str, float]:
        if rng_state is not None:
            self._rng.bit_generator.state = copy.deepcopy(rng_state)

        price_paths, variance_paths, dt = self._simulate_paths(
            contract.maturity, market, heston
        )
        payoff_fn = lambda prices: _payoff(contract.option_type, prices, contract.strike)
        discount_factor = np.exp(-market.rate * contract.maturity)

        exercise_probability: Optional[float] = None
        if contract.style == "european":
            payoffs = payoff_fn(price_paths[-1])
            discounted = discount_factor * payoffs
            mean_payoff = float(np.mean(payoffs))
        else:
            discounted, exercise_probability = self._least_squares_american(
                price_paths, payoff_fn, dt, market.rate
            )
            mean_payoff = float(np.mean(discounted))
        fair_value = float(np.mean(discounted))
        payoff_std = float(np.std(discounted, ddof=1))
        stderr = payoff_std / np.sqrt(discounted.size)
        ci_low = fair_value - Z_SCORE_99 * stderr
        ci_high = fair_value + Z_SCORE_99 * stderr

        return {
            "fair_value": fair_value,
            "std_error": stderr,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_payoff": mean_payoff,
            "discount_factor": discount_factor,
            "time_steps": price_paths.shape[0] - 1,
            "exercise_probability": exercise_probability,
            "num_samples": discounted.size,
            "discounted_payoffs": discounted,
            "price_paths": price_paths,
            "variance_paths": variance_paths,
        }

    def _simulate_paths(
        self,
        maturity: float,
        market: MarketParams,
        heston: HestonParams,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if maturity <= 0:
            raise ValueError("Maturity must be positive for simulation")
        dt = self.dt_seconds / SECONDS_IN_YEAR
        n_steps = max(1, int(np.ceil(maturity / dt)))
        dt = maturity / n_steps

        sqrt_dt = np.sqrt(dt)
        rho = np.clip(heston.rho, -0.999, 0.999)
        drift_adjustment = self.jumps.drift_adjustment()
        mu = market.rate - market.dividend - self.jumps.intensity * drift_adjustment

        prices = np.empty((n_steps + 1, self.n_paths))
        variances = np.empty_like(prices)
        prices[0] = market.spot
        variances[0] = max(heston.v0, 1e-8)

        prev_prices = prices[0]
        prev_variances = variances[0]

        for step in range(1, n_steps + 1):
            z1 = self._rng.standard_normal(self.n_paths)
            z2 = self._rng.standard_normal(self.n_paths)
            dW1 = sqrt_dt * z1
            dW2 = sqrt_dt * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)

            sqrt_prev_v = np.sqrt(np.maximum(prev_variances, 0.0))
            next_variances = (
                prev_variances
                + heston.kappa * (heston.theta - prev_variances) * dt
                + heston.sigma * sqrt_prev_v * dW2
            )
            next_variances = np.maximum(next_variances, 1e-10)

            jump_counts = self._rng.poisson(self.jumps.intensity * dt, size=self.n_paths)
            jump_normals = self._rng.standard_normal(self.n_paths)
            log_jump = np.where(
                jump_counts > 0,
                jump_counts * self.jumps.mean + np.sqrt(jump_counts) * self.jumps.std * jump_normals,
                0.0,
            )

            drift = (mu - 0.5 * prev_variances) * dt
            diffusion = sqrt_prev_v * dW1
            next_prices = prev_prices * np.exp(drift + diffusion + log_jump)

            prices[step] = next_prices
            variances[step] = next_variances
            prev_prices = next_prices
            prev_variances = next_variances

        return prices, variances, dt

    # ------------------------------------------------------------------
    # Greeks via finite differences
    # ------------------------------------------------------------------
    def _spot_greeks(
        self,
        contract: OptionContract,
        base_price: float,
        rng_state: dict,
        bump_size: float = 0.01,
    ) -> Tuple[float, float]:
        bump = max(self.market.spot * bump_size, 1e-6)
        up_market = MarketParams(
            spot=self.market.spot + bump,
            rate=self.market.rate,
            dividend=self.market.dividend,
        )
        down_market = MarketParams(
            spot=max(self.market.spot - bump, 1e-8),
            rate=self.market.rate,
            dividend=self.market.dividend,
        )
        up = self._price_core(contract, up_market, self.heston, rng_state)["fair_value"]
        down = self._price_core(contract, down_market, self.heston, rng_state)["fair_value"]
        delta = (up - down) / (2 * bump)
        gamma = (up - 2 * base_price + down) / (bump ** 2)
        return delta, gamma

    def _vega(
        self,
        contract: OptionContract,
        rng_state: dict,
        bump_size: float = 0.01,
    ) -> float:
        bump = max(self.heston.v0 * bump_size, 1e-6)
        up_heston = HestonParams(
            kappa=self.heston.kappa,
            theta=self.heston.theta,
            sigma=self.heston.sigma,
            rho=self.heston.rho,
            v0=self.heston.v0 + bump,
        )
        down_heston = HestonParams(
            kappa=self.heston.kappa,
            theta=self.heston.theta,
            sigma=self.heston.sigma,
            rho=self.heston.rho,
            v0=max(self.heston.v0 - bump, 1e-8),
        )
        up = self._price_core(contract, self.market, up_heston, rng_state)["fair_value"]
        down = self._price_core(contract, self.market, down_heston, rng_state)["fair_value"]
        return (up - down) / (2 * bump)

    def _theta(
        self,
        contract: OptionContract,
        base_price: float,
        rng_state: dict,
        bump_days: float = 1.0,
    ) -> float:
        bump_years = bump_days / 365.0
        if bump_years >= contract.maturity:
            bump_years = 0.5 * contract.maturity
        bumped_contract = OptionContract(
            strike=contract.strike,
            maturity=max(contract.maturity - bump_years, 1e-6),
            option_type=contract.option_type,
            style=contract.style,
            underlying_type=contract.underlying_type,
        )
        bumped_price = self._price_core(
            bumped_contract, self.market, self.heston, rng_state
        )["fair_value"]
        return (bumped_price - base_price) / bump_days

    # ------------------------------------------------------------------
    # American option pricing
    # ------------------------------------------------------------------
    def _least_squares_american(
        self,
        price_paths: np.ndarray,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        dt: float,
        rate: float,
    ) -> Tuple[np.ndarray, float]:
        discount = np.exp(-rate * dt)
        n_steps = price_paths.shape[0] - 1
        cashflows = payoff_fn(price_paths[-1])
        exercised = np.zeros(price_paths.shape[1], dtype=bool)

        for step in range(n_steps - 1, 0, -1):
            cashflows *= discount
            prices = price_paths[step]
            immediate = payoff_fn(prices)
            itm = immediate > 0
            if not np.any(itm):
                continue
            basis = np.column_stack((np.ones(np.sum(itm)), prices[itm], prices[itm] ** 2))
            continuation_coeffs = np.linalg.lstsq(basis, cashflows[itm], rcond=None)[0]
            continuation = (
                continuation_coeffs[0]
                + continuation_coeffs[1] * prices[itm]
                + continuation_coeffs[2] * prices[itm] ** 2
            )
            exercise_now = immediate[itm] > continuation
            exercise_indices = np.where(itm)[0][exercise_now]
            cashflows[exercise_indices] = immediate[exercise_indices]
            exercised[exercise_indices] = True

        cashflows *= discount
        exercise_probability = float(np.mean(exercised))
        return cashflows, exercise_probability
