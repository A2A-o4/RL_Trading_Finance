import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

# Import stable baselines components
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# Import our custom environment
from trading_environments import ContinuousTradingEnvironment

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
        env = ContinuousTradingEnvironment(df)
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


def train_model(env, total_timesteps=100000, save_path="models/sac"):
    """
    Train a SAC model
    
    Args:
        env: Training environment
        total_timesteps: Total timesteps for training
        save_path: Path to save the trained model
        
    Returns:
        Trained model
    """
    print("Creating SAC model with MLP policy...")
    
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    
    # Create model with MLP policy
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=0.0003,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        use_sde=False,
        policy_kwargs={"net_arch": [256, 256]}
    )
    
    # Set up evaluation callback
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
        tb_log_name="sac"
    )
    
    # Save the final model and normalization stats
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/final_model")
    env.save("vec_normalize.pkl")
    print(f"Model saved to {save_path}/final_model")
    print("Normalization statistics saved to vec_normalize.pkl")
    
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
    
    # Create test environment with normalization
    test_env = DummyVecEnv([lambda: ContinuousTradingEnvironment(test_df)])
    test_env = VecNormalize.load("vec_normalize.pkl", test_env)
    test_env.training = False  # Disable training mode
    test_env.norm_reward = False  # Disable reward normalization
    
    # Storage for results
    episode_rewards = []
    episode_lengths = []
    portfolio_histories = []
    
    for i in range(episodes):
        print(f"Episode {i+1}/{episodes}")
        
        # Reset environment
        obs = test_env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # Store history for this episode
        episode_history = {
            'total_assets': [10000],  # Initial portfolio value
            'cash': [10000],  # Initial cash
            'shares': [0],  # Initial shares
            'rewards': [0]  # Initial reward
        }
        
        while not done:
            # Get model's action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, rewards, dones, infos = test_env.step(action)
            
            # Extract values from vectorized environment
            reward = rewards[0]
            done = dones[0]
            info = infos[0]
            
            # Store step history
            episode_history['total_assets'].append(info['total_assets'])
            episode_history['cash'].append(info['cash_balance'])
            episode_history['shares'].append(info['shares_held'])
            episode_history['rewards'].append(reward)
            
            total_reward += reward
            step += 1
            
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"  Step {step}, Total Reward: {total_reward:.2f}, "
                      f"Total Assets: ${info['total_assets']:.2f}")
        
        # Save episode metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        portfolio_histories.append(episode_history)
        
        print(f"  Episode {i+1} finished: Steps={step}, Total Reward={total_reward:.2f}, "
              f"Final Portfolio=${info['total_assets']:.2f}")
    
    # Calculate average performance
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"Evaluation complete. Average reward: {avg_reward:.2f}, Average episode length: {avg_length:.1f}")
    
    # Plot comparison of episodes
    plt.figure(figsize=(15, 10))
    
    # Plot total assets
    plt.subplot(3, 1, 1)
    for i, hist in enumerate(portfolio_histories):
        plt.plot(hist['total_assets'], label=f"Episode {i+1}")
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot cash balance
    plt.subplot(3, 1, 2)
    for i, hist in enumerate(portfolio_histories):
        plt.plot(hist['cash'], label=f"Episode {i+1}")
    plt.title('Cash Balance Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Cash ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot shares held
    plt.subplot(3, 1, 3)
    for i, hist in enumerate(portfolio_histories):
        plt.plot(hist['shares'], label=f"Episode {i+1}")
    plt.title('Shares Held Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Number of Shares')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("evaluation_comparison.png")
    plt.close()
    
    # Save detailed evaluation results
    eval_results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'portfolio_histories': portfolio_histories
    }
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'episode': range(1, episodes + 1),
        'reward': episode_rewards,
        'length': episode_lengths,
        'final_portfolio_value': [hist['total_assets'][-1] for hist in portfolio_histories],
        'max_portfolio_value': [max(hist['total_assets']) for hist in portfolio_histories],
        'min_portfolio_value': [min(hist['total_assets']) for hist in portfolio_histories],
        'final_shares': [hist['shares'][-1] for hist in portfolio_histories],
        'final_cash': [hist['cash'][-1] for hist in portfolio_histories]
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    
    # Save detailed history for each episode
    for i, hist in enumerate(portfolio_histories):
        episode_df = pd.DataFrame({
            'step': range(len(hist['total_assets'])),
            'total_assets': hist['total_assets'],
            'cash': hist['cash'],
            'shares': hist['shares'],
            'reward': hist['rewards']
        })
        episode_df.to_csv(f'episode_{i+1}_history.csv', index=False)
    
    return eval_results


if __name__ == "__main__":
    print("Starting SAC trading agent training and evaluation...")
    
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