import time
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from trading.api import create_app
from trading.job_manager import BacktestJobManager


def test_submit_and_poll_backtest_job(tmp_path) -> None:
    manager = BacktestJobManager(tmp_path / "jobs.sqlite3", workers=1)
    client = TestClient(create_app(manager))
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bars = [
        {
            "timestamp": (start + timedelta(days=index)).isoformat(),
            "close": 100 + index,
        }
        for index in range(12)
    ]

    with client:
        submitted = client.post(
            "/api/v1/backtests",
            json={
                "bars": bars,
                "config": {
                    "fast_window": 2,
                    "slow_window": 4,
                    "initial_cash": 1000,
                    "commission_bps": 1,
                    "slippage_bps": 1,
                    "periods_per_year": 252,
                },
            },
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = client.get(f"/api/v1/backtests/{job_id}")
            assert response.status_code in {200, 202}
            result = response.json()
            if result["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.02)

    assert result["status"] == "succeeded"
    assert result["result"]["metrics"]["bar_count"] == len(bars)
