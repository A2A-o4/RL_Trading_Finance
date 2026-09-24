# Reinforcement Learning Agents for Stock Trading

Custom [Gymnasium](https://gymnasium.farama.org/) trading environments and several deep reinforcement
learning agents that learn to trade Apple (AAPL) stock from daily prices and technical indicators.
The agents are built with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and
[SB3-Contrib](https://sb3-contrib.readthedocs.io/).

> **Status:** research / learning project. The environments and training pipelines are complete; the
> agents' results are still unstable across evaluation checkpoints, so no performance claim is made
> here. See [Limitations](#limitations).

## Overview

```
yfinance ──► stock_analysis.py ──► train_data.csv / test_data.csv
                (indicators, scaling)         │
                                              ▼
                               trading_environments.py
                        (DiscreteTradingEnvironment: hold / buy / sell)
                        (ContinuousTradingEnvironment: action in [-1, 1])
                                              │
          ┌───────────────┬───────────────────┼────────────────────┐
          ▼               ▼                   ▼                    ▼
   Recurrent PPO      A2C + LSTM        DQN + LSTM features       SAC
   (discrete)         (discrete)        (discrete)            (continuous)
```

## Data

`stock_analysis.py` downloads AAPL daily data (2013 to 2025) with `yfinance`, fills missing values, scales
the features and adds technical indicators with `stockstats`: **MACD, RSI and Bollinger Bands**.
It then splits the data chronologically:

| Split | Period |
|---|---|
| Train | 2013-01-01 to 2022-12-31 |
| Test | 2023-01-01 onwards |

It also saves plots of price and volume, MACD, RSI and the Bollinger Bands to `plots/`.

## Trading environments

Both environments live in `trading_environments.py` and share a common abstract base class.

| | Discrete | Continuous |
|---|---|---|
| **Actions** | `0` hold, `1` buy, `2` sell (fixed trade size) | value in `[-1, 1]`: fraction of cash to invest (> 0) or of shares to sell (< 0) |
| **Observation** | last 30 days of prices, volume and indicators, plus cash balance and shares held | same |
| **Reward** | change in total portfolio value (cash + shares × price) at each step | same |
| **Costs** | 0.1% transaction cost on every trade | same |
| **Start** | $10,000 cash, no shares | same |

## Agents

| Script | Algorithm | Environment |
|---|---|---|
| `train_ppo_agent.py`, `ppotrading_final.py` | Recurrent PPO with an LSTM policy (SB3-Contrib) | discrete |
| `train_a2c_lstm_agent.py` | A2C with a custom LSTM feature extractor | discrete |
| `train_rainbow_dqn_agent.py` | DQN with an LSTM feature extractor | discrete |
| `Rainbow-DQN-version1.ipynb` | Rainbow-style DQN experiments (notebook) | discrete |
| `train_sac_agent.py` | Soft Actor-Critic | continuous |

Each script trains the agent, saves the model, then evaluates it on the test period and plots the
portfolio value. The PPO, DQN and SAC scripts normalise observations and rewards with `VecNormalize`.
`ppotrading_final.py` also reports **cumulative return, annualised Sharpe ratio, maximum drawdown and
volatility**, and plots the distribution of actions.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python stock_analysis.py           # 1. download data and build train/test sets
python test_trading_env.py         # 2. sanity-check both environments with random actions
python train_ppo_agent.py          # 3. train and evaluate an agent (or any other train_*.py)
```

Training length is set in each script (10,000 to 100,000 timesteps). TensorBoard logs are written to `logs/`:

```bash
tensorboard --logdir logs
```

## Repository structure

```
stock_analysis.py          data download, indicators, train/test split
trading_environments.py    discrete and continuous Gymnasium environments
test_trading_env.py        random-action checks of both environments
train_ppo_agent.py         Recurrent PPO (LSTM)
ppotrading_final.py        Recurrent PPO, final training and evaluation run
train_a2c_lstm_agent.py    A2C with LSTM feature extractor
train_rainbow_dqn_agent.py DQN with LSTM feature extractor
train_sac_agent.py         SAC on the continuous environment
Rainbow-DQN-version1.ipynb Rainbow DQN notebook experiments
MyCoRL Approach.jpg        research notes on a branching-exploration idea (see below)
```

## Research notes: a branching-exploration idea

`MyCoRL Approach.jpg` sketches an idea explored alongside the standard agents: **active exploration by
branching**. From a market state, the agent explores several short branches of possible actions
(buy / sell / hold), and credit is assigned to each state-action pair from the discounted return of its
branch, normalised by the branch length. It is inspired by how organisms route nutrients towards the most
rewarding paths and adapt in real time.

## Limitations

- **Single asset, daily data.** One stock (AAPL) and one train/test split, so results do not generalise.
- **Unstable evaluation.** Evaluation profit varies strongly between training checkpoints (see the
  notebook), so the agents are not ready for any real use.
- **Simplified market.** Orders fill at the closing price with a flat 0.1% cost. There is no slippage,
  no market impact and no short selling.

This project is for research and learning. It is **not financial advice**.

## Author

**Assim Ayoub**, [@A2A-o4](https://github.com/A2A-o4)
