"""Evaluate the saved agents on the held-out test period (2023-2024) and compare with baselines.

Usage:
    python evaluate_saved_models.py

The models in models/ were trained on train_data.csv (2013-2022) behind a VecNormalize wrapper whose
statistics were not saved. Observations are therefore evaluated two ways:
  raw   - unnormalised observations, exactly as evaluate_model() in train_ppo_agent.py does
  norm  - observations normalised with mean/std recomputed on the training data
Results go to results/test_metrics.csv and results/*.png.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

from trading_environments import DiscreteTradingEnvironment  # noqa: E402

OUT = Path("results"); OUT.mkdir(exist_ok=True)
train_df = pd.read_csv("train_data.csv", index_col=0, parse_dates=True)
test_df = pd.read_csv("test_data.csv", index_col=0, parse_dates=True)


def metrics(values: np.ndarray) -> dict:
    """Same formulas as compute_metrics() in ppotrading_final.py (drawdown reported in %)."""
    r = np.diff(values) / values[:-1]
    peak = np.maximum.accumulate(values)
    return {
        "cumulative_return": values[-1] / values[0] - 1,
        "sharpe": np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252),
        "max_drawdown": np.max((peak - values) / peak),
        "volatility": np.std(r) * np.sqrt(252),
    }


def obs_stats() -> tuple:
    """Mean / std of observations along the training data (random actions), like VecNormalize."""
    env, rng, obs_all = DiscreteTradingEnvironment(train_df), np.random.default_rng(0), []
    obs, _ = env.reset()
    done = False
    while not done:
        obs_all.append(obs)
        obs, _, term, trunc, _ = env.step(int(rng.integers(3)))
        done = term or trunc
    a = np.stack(obs_all)
    return a.mean(0), a.std(0) + 1e-8


MEAN, STD = obs_stats()


def run(model, mode: str, recurrent: bool) -> tuple:
    env = DiscreteTradingEnvironment(test_df)
    obs, _ = env.reset()
    state, start, done, actions = None, np.ones((1,), dtype=bool), False, []
    while not done:
        o = obs if mode == "raw" else np.clip((obs - MEAN) / STD, -10, 10)
        if recurrent:
            action, state = model.predict(o, state=state, episode_start=start, deterministic=True)
            start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(o, deterministic=True)
        actions.append(int(action))
        obs, _, term, trunc, _ = env.step(int(action))
        done = term or trunc
    return np.array(env.history["total_assets"], dtype=float), np.bincount(actions, minlength=3)


def buy_and_hold() -> np.ndarray:
    env = DiscreteTradingEnvironment(test_df)
    env.reset()
    env.trade_size = 10_000          # buy as many shares as possible on day 1, then hold
    env.step(1)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(0)
        done = term or trunc
    return np.array(env.history["total_assets"], dtype=float)


def random_policy(seed: int) -> np.ndarray:
    env, rng = DiscreteTradingEnvironment(test_df), np.random.default_rng(seed)
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(int(rng.integers(3)))
        done = term or trunc
    return np.array(env.history["total_assets"], dtype=float)


rows, curves = [], {}
bh = buy_and_hold(); curves["Buy & hold AAPL"] = bh
rows.append({"agent": "Buy & hold AAPL", "obs": "-", **metrics(bh)})
rand = [metrics(random_policy(s)) for s in range(100)]
rows.append({"agent": "Random policy (mean of 100)", "obs": "-",
             **{k: float(np.mean([m[k] for m in rand])) for k in rand[0]}})

MODELS = {
    "Recurrent PPO (final)": ("models/recurrent_ppo/final_model.zip", RecurrentPPO, True),
    "Recurrent PPO (best checkpoint)": ("models/recurrent_ppo_best/best_model.zip", RecurrentPPO, True),
    "DQN (final)": ("models/dqn/final_model.zip", DQN, False),
    "DQN (best checkpoint)": ("models/dqn_best/best_model.zip", DQN, False),
}
for name, (path, cls, rec) in MODELS.items():
    try:
        model = cls.load(path, device="cpu", custom_objects={"lr_schedule": lambda _: 0.0,
                                                             "exploration_schedule": lambda _: 0.0})
    except Exception as e:  # noqa: BLE001
        print(f"{name}: could not load ({type(e).__name__}: {e})")
        continue
    for mode in ("raw", "norm"):
        vals, acts = run(model, mode, rec)
        curves[f"{name} [{mode}]"] = vals
        rows.append({"agent": name, "obs": mode, **metrics(vals),
                     "actions_hold_buy_sell": "/".join(map(str, acts))})

res = pd.DataFrame(rows)
res.to_csv(OUT / "test_metrics.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))

dates = test_df.index[DiscreteTradingEnvironment(test_df).window_size - 1:]
plt.figure(figsize=(11, 5))
SHOWN = {"Buy & hold AAPL": ("#888888", "--"), "Recurrent PPO (final) [raw]": ("#1A7A8A", "-"),
         "DQN (best checkpoint) [norm]": ("#C0392B", "-")}
for k, (color, ls) in SHOWN.items():
    v = curves[k]
    plt.plot(dates[:len(v)], v, color=color, ls=ls, lw=2, label=k.replace(" [raw]", "").replace(" [norm]", ""))
plt.axhline(10_000, color="grey", lw=0.8)
plt.title("Portfolio value on the test period (2023-2024), $10,000 start")
plt.ylabel("Portfolio value ($)"); plt.legend(fontsize=7, ncol=2); plt.tight_layout()
plt.savefig(OUT / "test_portfolio_curves.png", dpi=130)
