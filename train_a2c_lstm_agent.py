import os
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from trading_environments import DiscreteTradingEnvironment, ContinuousTradingEnvironment

class LSTMExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using LSTM layers
    """
    def __init__(self, observation_space, features_dim=128, lstm_hidden_dim=64, n_lstm_layers=2):
        super(LSTMExtractor, self).__init__(observation_space, features_dim)
        
        # Get input dimensions from observation space
        n_input = observation_space.shape[-1]  # Number of features
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=lstm_hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_dim, features_dim)
        self.fc2 = nn.Linear(features_dim, features_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(features_dim)
        
    def forward(self, observations):
        # Reshape observations for LSTM (batch_size, sequence_length, features)
        batch_size = observations.shape[0]
        sequence_length = observations.shape[1]
        features = observations.shape[2]
        
        # LSTM expects (batch_size, sequence_length, features)
        lstm_out, _ = self.lstm(observations)
        
        # Take the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(last_output))
        x = F.relu(self.fc2(x))
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x

def create_env(df, env_type='discrete', window_size=30, initial_balance=10000):
    """
    Create and wrap the trading environment
    """
    if env_type.lower() == 'discrete':
        env = DiscreteTradingEnvironment(
            df=df,
            window_size=window_size,
            initial_balance=initial_balance
        )
    else:
        env = ContinuousTradingEnvironment(
            df=df,
            window_size=window_size,
            initial_balance=initial_balance
        )
    
    # Wrap the environment with Monitor
    env = Monitor(env)
    return env

def train_a2c_lstm(
    train_data_path,
    test_data_path,
    env_type='discrete',
    window_size=30,
    initial_balance=10000,
    total_timesteps=100000,
    save_path='models/a2c_lstm_model'
):
    """
    Train A2C agent with LSTM architecture
    """
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(train_data_path, index_col=0)
    test_df = pd.read_csv(test_data_path, index_col=0)
    
    # Create environments
    train_env = create_env(train_df, env_type, window_size, initial_balance)
    eval_env = create_env(test_df, env_type, window_size, initial_balance)
    
    # Wrap environments
    train_env = DummyVecEnv([lambda: train_env])
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path='logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Create policy with LSTM feature extractor
    policy_kwargs = dict(
        features_extractor_class=LSTMExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            lstm_hidden_dim=64,
            n_lstm_layers=2
        )
    )
    
    # Initialize A2C model
    model = A2C(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log='logs/'
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    
    # Save the final model
    model.save(f"{save_path}_final")
    
    return model

def evaluate_model(model, test_data_path, env_type='discrete', window_size=30, initial_balance=10000):
    """
    Evaluate the trained model on test data
    """
    # Load test data
    test_df = pd.read_csv(test_data_path, index_col=0)
    
    # Create test environment
    test_env = create_env(test_df, env_type, window_size, initial_balance)
    test_env = DummyVecEnv([lambda: test_env])
    
    # Evaluate
    obs = test_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
    
    # Plot portfolio performance - access the underlying environment
    test_env.envs[0].env.plot_portfolio_performance()
    
    # Export trading history - access the underlying environment
    test_env.envs[0].env.export_history(f'{env_type}_trading_history.csv')
    
    return total_reward

if __name__ == "__main__":
    # Train and evaluate the model
    model = train_a2c_lstm(
        train_data_path='train_data.csv',
        test_data_path='test_data.csv',
        env_type='discrete',  # or 'continuous'
        window_size=30,
        initial_balance=10000,
        total_timesteps=100000
    )
    
    # Evaluate the model
    total_reward = evaluate_model(
        model,
        test_data_path='test_data.csv',
        env_type='discrete'  # or 'continuous'
    )
    
    print(f"Total reward on test data: {total_reward}") 