import numpy as np
import pandas as pd
import pytest

from trading.engine import BacktestConfig, run_backtest


def test_backtest_returns_finite_metrics_and_aligned_equity_curve() -> None:
    prices = np.concatenate(
        [
            np.linspace(100, 80, 40),
            np.linspace(80, 140, 70),
            np.linspace(140, 110, 50),
        ]
    )
    bars = pd.DataFrame(
        {"close": prices},
        index=pd.date_range("2024-01-01", periods=len(prices), freq="D"),
    )

    result = run_backtest(
        bars,
        BacktestConfig(
            fast_window=5,
            slow_window=15,
            commission_bps=1,
            slippage_bps=1,
        ),
    )

    assert result["metrics"]["bar_count"] == len(bars)
    assert len(result["equity_curve"]) == len(bars)
    assert np.isfinite(result["metrics"]["sharpe_ratio"])
    assert result["metrics"]["trade_count"] > 0


@pytest.mark.parametrize(
    ("bars", "config", "message"),
    [
        (pd.DataFrame({"price": [1, 2, 3]}), BacktestConfig(1, 2), "close"),
        (pd.DataFrame({"close": [1, 2, 3]}), BacktestConfig(1, 5), "At least"),
        (pd.DataFrame({"close": [1, 0, 3]}), BacktestConfig(1, 2), "greater than zero"),
    ],
)
def test_backtest_validates_input(
    bars: pd.DataFrame,
    config: BacktestConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_backtest(bars, config)


def test_backtest_rejects_non_finite_fees() -> None:
    config = BacktestConfig(
        fast_window=1,
        slow_window=2,
        commission_bps=float("nan"),
    )

    with pytest.raises(ValueError, match="finite numbers"):
        config.validate()
