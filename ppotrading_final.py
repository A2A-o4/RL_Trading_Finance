import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib import RecurrentPPO

from trading_env import DiscreteTradingEnvironment


np.random.seed(0)
torch.manual_seed(0)


def create_env(data_path, log_dir=None):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    def _init():
        env = DiscreteTradingEnvironment(df)
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        training=True
    )
    return vec_env


def train_model(env, total_timesteps=100000, save_path="models/recurrent_ppo"):
    print("Creating RecurrentPPO model with LSTM policy...")
    os.makedirs("logs", exist_ok=True)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=0.0003,
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        policy_kwargs={"lstm_hidden_size": 64, "n_lstm_layers": 1}
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{save_path}_best",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    print(f"Training model for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="recurrent_ppo"
    )

    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/final_model")
    print(f"Model saved to {save_path}/final_model")
    return model


def compute_metrics(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values)
    volatility = np.std(returns) * np.sqrt(252)
    return {
        'Sharpe Ratio': sharpe_ratio,
        'Cumulative Return': cumulative_return,
        'Max Drawdown': max_drawdown,
        'Volatility': volatility
    }


def evaluate_model(model, test_data_path, episodes=3):
    print(f"Evaluating model on test data ({test_data_path})...")
    test_df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    test_env = DiscreteTradingEnvironment(test_df)

    episode_rewards = []
    episode_lengths = []
    portfolio_histories = []
    metrics_list = []

    for i in range(episodes):
        print(f"\nEpisode {i+1}/{episodes}")
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        step = 0
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        actions_taken = []

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            obs, reward, terminated, truncated, info = test_env.step(action.item())
            done = terminated or truncated
            episode_starts = False
            total_reward += reward
            step += 1
            actions_taken.append(action.item())

            if step % 10 == 0:
                print(f"  Step {step}, Total Reward: {total_reward:.2f}, "
                      f"Total Assets: ${info['total_assets']:.2f}")

        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        portfolio_histories.append({
            'total_assets': test_env.history['total_assets'],
            'cash': test_env.history['cash'],
            'shares': test_env.history['shares']
        })

        # Plot portfolio
        test_env.plot_portfolio_performance()
        os.makedirs("ppoagent", exist_ok=True)
        plt.savefig(f"ppoagent/evaluation_episode_{i+1}.png")
        plt.close()

        # Plot action distribution
        plt.figure()
        action_labels = ['Hold', 'Buy', 'Sell']
        counts = [actions_taken.count(i) for i in range(len(action_labels))]
        plt.bar(action_labels, counts)
        plt.title(f'Action Distribution - Episode {i+1}')
        plt.ylabel('Count')
        plt.savefig(f"ppoagent/action_distribution_episode_{i+1}.png")
        plt.close()

        print(f"  Episode {i+1} finished: Steps={step}, Total Reward={total_reward:.2f}, "
              f"Final Portfolio=${test_env.total_assets:.2f}")

        metrics = compute_metrics(test_env.history['total_assets'])
        metrics_list.append(metrics)

    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()

    print("\n=== Evaluation Summary ===")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f}")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot comparison of episodes
    plt.figure(figsize=(12, 6))
    for i, hist in enumerate(portfolio_histories):
        plt.plot(hist['total_assets'], label=f"Episode {i+1}")
    plt.title('Portfolio Value Across Evaluation Episodes')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig("ppoagent/evaluation_comparison.png")
    plt.close()

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'portfolio_histories': portfolio_histories,
        'metrics': avg_metrics
    }


if __name__ == "__main__":
    print("Starting RecurrentPPO with LSTM trading agent training and evaluation...")

    train_data_path = "train_data.csv"
    test_data_path = "test_data.csv"

    print("Creating training environment...")
    env = create_env(train_data_path, log_dir="./logs/monitor")

    print("Training model...")
    model = train_model(env, total_timesteps=100000)

    print("Evaluating model on test data...")
    eval_results = evaluate_model(model, test_data_path, episodes=3)

    print("\nTraining and evaluation complete!")
    print(f"Average reward on test set: {eval_results['avg_reward']:.2f}")
    print(f"Average episode length on test set: {eval_results['avg_length']:.1f}")
