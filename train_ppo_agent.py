import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

# Import stable baselines components
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib import RecurrentPPO

# Import our custom environment
from trading_environments import DiscreteTradingEnvironment

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)


def create_env(data_path, log_dir=None):
    """
    Create and wrap the trading environment
    
    Args:
        data_path: Path to the CSV file with trading data
        log_dir: Directory to save Monitor logs, if None no monitoring will be done
        
    Returns:
        VecNormalize wrapped environment
    """
    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Function to create the environment
    def _init():
        env = DiscreteTradingEnvironment(df)
        if log_dir is not None:
            # Create directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)
        return env
    
    # Create vectorized environment (required for Stable Baselines)
    vec_env = DummyVecEnv([_init])
    
    # Normalize observations and rewards
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
    """
    Train a RecurrentPPO model
    
    Args:
        env: Training environment
        total_timesteps: Total timesteps for training
        save_path: Path to save the trained model
        
    Returns:
        Trained model
    """
    print("Creating RecurrentPPO model with LSTM policy...")
    
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    
    # Create model
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
    
    # Set up evaluation callback with less frequent evaluation
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
    
    # Save the final model
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/final_model")
    print(f"Model saved to {save_path}/final_model")
    
    return model


def evaluate_model(model, test_data_path, episodes=3):
    """
    Evaluate a trained model on test data
    
    Args:
        model: Trained model
        test_data_path: Path to test data CSV
        episodes: Number of evaluation episodes
        
    Returns:
        dict: Performance metrics
    """
    print(f"Evaluating model on test data ({test_data_path})...")
    
    # Load test data
    test_df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    
    # Create test environment (not normalized for evaluation)
    test_env = DiscreteTradingEnvironment(test_df)
    
    # Storage for results
    episode_rewards = []
    episode_lengths = []
    portfolio_histories = []
    
    for i in range(episodes):
        print(f"Episode {i+1}/{episodes}")
        
        # Reset environment
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # Recurrent policy requires initial states
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        while not done:
            # Get model's action
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            
            # Step environment
            obs, reward, terminated, truncated, info = test_env.step(action.item())
            
            # Done if either terminated or truncated
            done = terminated or truncated
            
            # After first step, episode_starts is always False
            episode_starts = False
            
            total_reward += reward
            step += 1
            
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"  Step {step}, Total Reward: {total_reward:.2f}, "
                      f"Total Assets: ${info['total_assets']:.2f}")
        
        # Save episode metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        portfolio_histories.append({
            'total_assets': test_env.history['total_assets'],
            'cash': test_env.history['cash'],
            'shares': test_env.history['shares']
        })
        
        # Plot portfolio performance for this episode
        test_env.plot_portfolio_performance()
        plt.savefig(f"evaluation_episode_{i+1}.png")
        
        print(f"  Episode {i+1} finished: Steps={step}, Total Reward={total_reward:.2f}, "
              f"Final Portfolio=${test_env.total_assets:.2f}")
    
    # Calculate average performance
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"Evaluation complete. Average reward: {avg_reward:.2f}, Average episode length: {avg_length:.1f}")
    
    # Plot comparison of episodes
    plt.figure(figsize=(12, 6))
    for i, hist in enumerate(portfolio_histories):
        plt.plot(hist['total_assets'], label=f"Episode {i+1}")
    
    plt.title('Portfolio Value Across Evaluation Episodes')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluation_comparison.png")
    plt.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'portfolio_histories': portfolio_histories
    }


if __name__ == "__main__":
    print("Starting RecurrentPPO with LSTM trading agent training and evaluation...")
    
    # Paths to data
    train_data_path = "train_data.csv"
    test_data_path = "test_data.csv"
    
    # Create and wrap training environment
    print("Creating training environment...")
    env = create_env(train_data_path, log_dir="./logs/monitor")
    
    # Train model
    print("Training model...")
    model = train_model(env, total_timesteps=10000)
    
    # Evaluate on test data
    print("Evaluating model on test data...")
    eval_results = evaluate_model(model, test_data_path, episodes=3)
    
    print("Training and evaluation complete!")
    print(f"Average reward on test set: {eval_results['avg_reward']:.2f}")
    print(f"Average episode length on test set: {eval_results['avg_length']:.1f}") 