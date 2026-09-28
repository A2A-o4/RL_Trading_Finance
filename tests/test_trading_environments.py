"""Unit tests for the trading environments.

Run with:  pytest
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_environments import ContinuousTradingEnvironment, DiscreteTradingEnvironment

ROOT = Path(__file__).resolve().parent.parent
HOLD, BUY, SELL = 0, 1, 2


def make_df(n=60, start=100.0, step=1.0):
    """Small synthetic price series: the close rises by `step` every day."""
    close = start + step * np.arange(n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": close, "Volume": np.linspace(1e6, 2e6, n)}, index=idx)


@pytest.fixture(params=[DiscreteTradingEnvironment, ContinuousTradingEnvironment])
def env_cls(request):
    return request.param


def buy_action(env):
    return BUY if isinstance(env, DiscreteTradingEnvironment) else np.array([1.0])


def test_observation_shape_and_dtype(env_cls):
    env = env_cls(make_df(), window_size=10)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape == (10, 2 + 2)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_agent_sees_its_own_position(env_cls):
    """Cash and shares must be visible in the observation (they were always 0 before the fix)."""
    env = env_cls(make_df(), window_size=10, max_shares=100)
    obs, _ = env.reset()
    assert np.allclose(obs[:, -2], 1.0)   # all cash
    assert np.allclose(obs[:, -1], 0.0)   # no shares

    obs, *_ = env.step(buy_action(env))
    assert env.shares_held > 0
    assert np.allclose(obs[:, -1], env.shares_held / env.max_shares)
    assert np.allclose(obs[:, -2], env.cash_balance / env.initial_balance)
    assert obs[0, -1] > 0


def test_legacy_mode_hides_position(env_cls):
    """legacy=True reproduces the original environment the saved models were trained in."""
    env = env_cls(make_df(), window_size=10, legacy=True)
    env.reset()
    obs, *_ = env.step(buy_action(env))
    assert env.shares_held > 0
    assert np.allclose(obs[:, -2:], 0.0)


def test_reward_is_next_day_result_of_position():
    """Trade at the last observed close, then earn the move to the next close."""
    df = make_df(step=1.0)
    env = DiscreteTradingEnvironment(df, window_size=10)
    env.reset()
    trade_price = df["Close"].iloc[9]
    next_price = df["Close"].iloc[10]

    _, reward, _, _, info = env.step(BUY)
    shares = env.shares_held
    assert shares == env.trade_size
    assert info["current_price"] == trade_price
    assert info["valuation_price"] == next_price
    cost = shares * trade_price * env.transaction_cost_pct
    assert reward == pytest.approx(shares * (next_price - trade_price) - cost)

    # Holding: reward is exactly the price move on the shares held
    _, reward, *_ = env.step(HOLD)
    assert reward == pytest.approx(shares * (df["Close"].iloc[11] - next_price))


def test_rewards_sum_to_profit(env_cls):
    env = env_cls(make_df(), window_size=10)
    env.reset()
    rng = np.random.default_rng(0)
    total, done = 0.0, False
    while not done:
        action = env.action_space.sample() if isinstance(env, ContinuousTradingEnvironment) else int(rng.integers(3))
        _, reward, term, trunc, _ = env.step(action)
        total += reward
        done = term or trunc
    assert total == pytest.approx(env.total_assets - env.initial_balance)


def test_observation_has_no_future_data():
    """The observation at a step only contains rows up to the last observed close."""
    df = make_df()
    env = DiscreteTradingEnvironment(df, window_size=10)
    env.reset()
    for _ in range(5):
        env.step(HOLD)
    obs_a = env._get_observation()
    changed = df.copy()
    changed.iloc[env.current_step:, :] = 1e9   # tamper with every row after the window
    env.df = changed
    assert np.array_equal(obs_a, env._get_observation())


def test_episode_runs_to_end_of_data(env_cls):
    df = make_df(n=40)
    env = env_cls(df, window_size=10)
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, term, trunc, _ = env.step(HOLD if env_cls is DiscreteTradingEnvironment else np.array([0.0]))
        steps += 1
        done = term or trunc
    assert steps == len(df) - 10
    assert trunc


def test_cannot_overspend_or_oversell():
    env = DiscreteTradingEnvironment(make_df(start=5000.0), window_size=10, initial_balance=10000)
    env.reset()
    env.step(BUY)                        # can only afford 1 share at $5,009
    assert env.shares_held == 1
    assert env.cash_balance >= 0
    for _ in range(3):
        env.step(SELL)
    assert env.shares_held == 0


def test_real_test_data_runs():
    df = pd.read_csv(ROOT / "test_data.csv", index_col=0, parse_dates=True)
    env = DiscreteTradingEnvironment(df)
    obs, _ = env.reset()
    assert obs.shape == (30, df.shape[1] + 2)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(BUY)
        done = term or trunc
    assert np.isfinite(obs).all()
    assert env.total_assets > 0
