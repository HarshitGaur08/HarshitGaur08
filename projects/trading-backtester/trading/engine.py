"""Vectorized long/cash moving-average crossover backtesting engine."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    fast_window: int = 20
    slow_window: int = 50
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    periods_per_year: int = 252

    def validate(self) -> None:
        integer_fields = (
            self.fast_window,
            self.slow_window,
            self.periods_per_year,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_fields
        ):
            raise ValueError("Windows and periods_per_year must be integers")
        if self.fast_window < 1 or self.slow_window < 2:
            raise ValueError("Moving-average windows must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be less than slow_window")
        numeric_fields = (
            self.initial_cash,
            self.commission_bps,
            self.slippage_bps,
        )
        if any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in numeric_fields
        ):
            raise ValueError("Cash, fees, and slippage must be finite numbers")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Fees and slippage cannot be negative")
        if self.commission_bps + self.slippage_bps >= 10_000:
            raise ValueError("Combined fees and slippage must be below 10,000 bps")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")


def run_backtest(
    bars: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    """Run a close-only long/cash strategy and return metrics and equity points.

    Signals computed at a bar close are applied to the next bar, preventing
    same-bar look-ahead. Transaction costs apply to each change in exposure.
    """
    settings = config or BacktestConfig()
    settings.validate()
    if "close" not in bars.columns:
        raise ValueError("Input data must contain a 'close' column")
    if len(bars) < settings.slow_window + 1:
        raise ValueError(
            f"At least {settings.slow_window + 1} bars are required"
        )

    prices = pd.to_numeric(bars["close"], errors="coerce").astype("float64")
    if prices.isna().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Close prices must be finite numeric values")
    if (prices <= 0).any():
        raise ValueError("Close prices must be greater than zero")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("Bar index must be unique and chronologically sorted")

    fast_average = prices.rolling(
        window=settings.fast_window,
        min_periods=settings.fast_window,
    ).mean()
    slow_average = prices.rolling(
        window=settings.slow_window,
        min_periods=settings.slow_window,
    ).mean()
    target_position = (fast_average > slow_average).astype("float64")
    position = target_position.shift(1, fill_value=0.0)

    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    if not np.isfinite(asset_returns.to_numpy()).all():
        raise ValueError("Price returns must be finite")
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    cost_rate = (settings.commission_bps + settings.slippage_bps) / 10_000.0
    strategy_returns = position * asset_returns - turnover * cost_rate
    if not np.isfinite(strategy_returns.to_numpy()).all():
        raise ValueError("Strategy returns must be finite")
    equity = settings.initial_cash * (1.0 + strategy_returns).cumprod()
    if not np.isfinite(equity.to_numpy()).all():
        raise ValueError("Equity curve overflowed; review prices and fee settings")

    nonzero_returns = strategy_returns.iloc[1:]
    if nonzero_returns.empty:
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
    else:
        standard_deviation = float(nonzero_returns.std(ddof=1))
        annualized_volatility = (
            standard_deviation * np.sqrt(settings.periods_per_year)
            if np.isfinite(standard_deviation)
            else 0.0
        )
        mean_return = float(nonzero_returns.mean())
        sharpe_ratio = (
            mean_return / standard_deviation * np.sqrt(settings.periods_per_year)
            if standard_deviation > 0
            else 0.0
        )

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    elapsed_periods = max(len(prices) - 1, 1)
    years = elapsed_periods / settings.periods_per_year
    final_equity = float(equity.iloc[-1])
    total_return = final_equity / settings.initial_cash - 1.0
    cagr = (
        (final_equity / settings.initial_cash) ** (1.0 / years) - 1.0
        if final_equity > 0 and years > 0
        else -1.0
    )

    index_values = bars.index.astype(str)
    equity_curve = [
        {"timestamp": timestamp, "equity": round(float(value), 6)}
        for timestamp, value in zip(index_values, equity.to_numpy(), strict=True)
    ]
    return {
        "config": asdict(settings),
        "metrics": {
            "initial_cash": settings.initial_cash,
            "final_equity": round(final_equity, 6),
            "total_return": round(float(total_return), 8),
            "cagr": round(float(cagr), 8),
            "annualized_volatility": round(float(annualized_volatility), 8),
            "sharpe_ratio": round(float(sharpe_ratio), 8),
            "max_drawdown": round(float(drawdown.min()), 8),
            "trade_count": int((turnover > 0).sum()),
            "market_exposure": round(float(position.mean()), 8),
            "bar_count": int(len(bars)),
        },
        "equity_curve": equity_curve,
    }
