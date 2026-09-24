import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class CustomTradingEnvironment(gym.Env, ABC):
    """
    Abstract Base Class for custom trading environments.
    Inherits from gym.Env and ABC (Abstract Base Class).
    """
    
    def __init__(self, df, window_size=30, initial_balance=10000, max_shares=1000):
        """
        Initialize the trading environment.
        
        Args:
            df (pd.DataFrame): DataFrame with price and indicator data
            window_size (int): Size of the observation window
            initial_balance (float): Initial cash balance
            max_shares (int): Maximum number of shares that can be held
        """
        super(CustomTradingEnvironment, self).__init__()
        
        # Store the dataframe
        self.df = df
        
        # Trading parameters
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_shares = max_shares
        
        # Current state
        self.cash_balance = initial_balance
        self.shares_held = 0
        self.current_step = window_size  # Start after the window_size to have enough history
        self.total_assets = initial_balance
        
        # History
        self.history = {
            'cash': [initial_balance],
            'shares': [0],
            'total_assets': [initial_balance],
            'rewards': []
        }
        
        # Transaction costs (optional)
        self.transaction_cost_pct = 0.001  # 0.1% per trade
        
        # For normalization
        self.features = list(self.df.columns)
        
        # Define observation space
        # Features include OHLCV, indicators, and portfolio state (cash, shares)
        self.num_features = len(self.features) + 2  # +2 for cash balance and shares held
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )
    
    @abstractmethod
    def _get_observation(self):
        """
        Return the current observation.
        This should be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment based on the action.
        This should be implemented by subclasses.
        
        Args:
            action: The action to take
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        This should be implemented by subclasses.
        
        Args:
            seed: Random seed
            options: Additional options for reset
        
        Returns:
            observation: The initial observation
        """
        pass
    
    @abstractmethod
    def render(self, mode='human'):
        """
        Render the environment.
        This should be implemented by subclasses.
        
        Args:
            mode (str): The mode to render with
            
        Returns:
            None
        """
        pass
    
    def _normalize_observation(self, observation):
        """
        Normalize the observation to be between 0 and 1.
        
        Args:
            observation: The observation to normalize
            
        Returns:
            np.array: The normalized observation
        """
        # Use a simple min-max normalization for the price data
        # We would normally use statistics from the training set
        # But for simplicity, we'll just use local normalization
        obs_min = np.min(observation, axis=0)
        obs_max = np.max(observation, axis=0)
        
        # Avoid division by zero
        obs_max = np.where(obs_max - obs_min > 0, obs_max, obs_min + 1)
        
        return (observation - obs_min) / (obs_max - obs_min + 1e-8)
    
    def plot_portfolio_performance(self):
        """
        Plot the portfolio performance over time.
        
        Returns:
            None
        """
        plt.figure(figsize=(12, 8))
        
        # Plot total assets
        plt.subplot(3, 1, 1)
        plt.plot(self.history['total_assets'])
        plt.title('Total Portfolio Value Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Value ($)')
        
        # Plot cash balance
        plt.subplot(3, 1, 2)
        plt.plot(self.history['cash'])
        plt.title('Cash Balance Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Cash ($)')
        
        # Plot shares held
        plt.subplot(3, 1, 3)
        plt.plot(self.history['shares'])
        plt.title('Shares Held Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Shares')
        
        plt.tight_layout()
        plt.savefig('portfolio_performance.png')
        plt.close()
    
    def export_history(self, filename='trading_history.csv'):
        """
        Export the trading history to a CSV file.
        
        Args:
            filename (str): The name of the file to save to
            
        Returns:
            None
        """
        history_df = pd.DataFrame({
            'cash': self.history['cash'],
            'shares': self.history['shares'],
            'total_assets': self.history['total_assets'],
            'rewards': self.history['rewards'] + [0]  # Add a 0 to match the length
        })
        history_df.to_csv(filename, index_label='step')
        print(f"Trading history exported to {filename}")


class DiscreteTradingEnvironment(CustomTradingEnvironment):
    """
    Trading environment with discrete actions (Hold, Buy, Sell).
    """
    
    def __init__(self, df, window_size=30, initial_balance=10000, max_shares=1000):
        super(DiscreteTradingEnvironment, self).__init__(
            df=df,
            window_size=window_size,
            initial_balance=initial_balance,
            max_shares=max_shares
        )
        
        # Define action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Fixed trade size
        self.trade_size = 10  # Buy/sell 10 shares at a time
    
    def _get_observation(self):
        """
        Get current observation of window_size steps and portfolio state.
        
        Returns:
            np.array: The current observation
        """
        # Get the window of data
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # Get the window of price and indicator data
        window_data = self.df.iloc[start_idx:end_idx].values
        
        # Get the current price
        current_price = self.df.iloc[self.current_step - 1]['Close']
        
        # Add portfolio state (cash balance and shares held) to each timestep
        portfolio_state = np.array([
            [self.cash_balance / self.initial_balance, self.shares_held / self.max_shares] 
            for _ in range(self.window_size)
        ])
        
        # Combine price/indicator data with portfolio state
        observation = np.column_stack((window_data, portfolio_state))
        
        # Normalize the observation
        return self._normalize_observation(observation).astype(np.float32)
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action (int): 0=Hold, 1=Buy, 2=Sell
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Get current price
        current_price = self.df.iloc[self.current_step - 1]['Close']
        
        # Get previous total assets for reward calculation
        previous_total_assets = self.total_assets
        
        # Execute trade based on action
        if action == 1:  # Buy
            # Calculate maximum number of shares we can buy
            max_shares_possible = min(
                self.trade_size,  # Buy trade_size shares
                np.floor(self.cash_balance / (current_price * (1 + self.transaction_cost_pct)))
            )
            
            # Make the purchase (if we can)
            if max_shares_possible > 0:
                # Update shares and cash
                shares_bought = max_shares_possible
                cost = shares_bought * current_price * (1 + self.transaction_cost_pct)
                
                self.shares_held += shares_bought
                self.cash_balance -= cost
                
        elif action == 2:  # Sell
            # Calculate shares to sell
            shares_to_sell = min(self.shares_held, self.trade_size)
            
            # Make the sale (if we can)
            if shares_to_sell > 0:
                # Update shares and cash
                proceeds = shares_to_sell * current_price * (1 - self.transaction_cost_pct)
                
                self.shares_held -= shares_to_sell
                self.cash_balance += proceeds
        
        # Else action == 0 (Hold): Do nothing
        
        # Move to next timestep
        self.current_step += 1
        
        # Calculate new total assets and update history
        self.total_assets = self.cash_balance + self.shares_held * current_price
        
        self.history['cash'].append(self.cash_balance)
        self.history['shares'].append(self.shares_held)
        self.history['total_assets'].append(self.total_assets)
        
        # Calculate reward as change in total assets
        reward = self.total_assets - previous_total_assets
        self.history['rewards'].append(reward)
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        # Terminated if balance is too low (failed episode)
        if self.total_assets <= 0.1 * self.initial_balance:
            terminated = True
        
        # Truncated if we've reached the end of data
        if self.current_step >= len(self.df):
            truncated = True
        
        # Get new observation
        obs = self._get_observation()
        
        # Info dictionary for debugging
        info = {
            'step': self.current_step,
            'cash_balance': self.cash_balance,
            'shares_held': self.shares_held,
            'total_assets': self.total_assets,
            'current_price': current_price,
            'action': ['Hold', 'Buy', 'Sell'][action]
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            observation: The initial observation
            info: Empty dictionary
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state variables
        self.cash_balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size  # Start after window_size to have enough history
        self.total_assets = self.initial_balance
        
        # Reset history
        self.history = {
            'cash': [self.initial_balance],
            'shares': [0],
            'total_assets': [self.initial_balance],
            'rewards': []
        }
        
        return self._get_observation(), {}
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode to render with
            
        Returns:
            None
        """
        if mode == 'human':
            # Get current price
            current_price = self.df.iloc[self.current_step - 1]['Close']
            
            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Cash Balance: ${self.cash_balance:.2f}")
            print(f"Shares Held: {self.shares_held}")
            print(f"Total Assets: ${self.total_assets:.2f}")
            print("-" * 50)
        else:
            raise NotImplementedError(f"Render mode {mode} is not implemented")


class ContinuousTradingEnvironment(CustomTradingEnvironment):
    """
    Trading environment with continuous actions (-1 to 1).
    -1: Sell 100% of shares
    +1: Use 100% of cash to buy shares
    Values in between correspond to using that percentage of cash/shares.
    """
    
    def __init__(self, df, window_size=30, initial_balance=10000, max_shares=1000):
        super(ContinuousTradingEnvironment, self).__init__(
            df=df,
            window_size=window_size,
            initial_balance=initial_balance,
            max_shares=max_shares
        )
        
        # Define action space: -1 to 1 for sell and buy percentages
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    
    def _get_observation(self):
        """
        Get current observation of window_size steps and portfolio state.
        
        Returns:
            np.array: The current observation
        """
        # Get the window of data
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # Get the window of price and indicator data
        window_data = self.df.iloc[start_idx:end_idx].values
        
        # Get the current price
        current_price = self.df.iloc[self.current_step - 1]['Close']
        
        # Add portfolio state (cash balance and shares held) to each timestep
        portfolio_state = np.array([
            [self.cash_balance / self.initial_balance, self.shares_held / self.max_shares] 
            for _ in range(self.window_size)
        ])
        
        # Combine price/indicator data with portfolio state
        observation = np.column_stack((window_data, portfolio_state))
        
        # Normalize the observation
        return self._normalize_observation(observation).astype(np.float32)
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action (float): Value between -1 and 1
                -1: Sell 100% of shares
                +1: Use 100% of cash to buy shares
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Ensure action is in the correct range
        action = np.clip(action, -1, 1)[0]  # Extract the scalar value from the action array
        
        # Get current price
        current_price = self.df.iloc[self.current_step - 1]['Close']
        
        # Get previous total assets for reward calculation
        previous_total_assets = self.total_assets
        
        # Execute trade based on action
        if action > 0:  # Buy
            # Calculate percentage of cash to use (action is between 0 and 1)
            cash_to_use = self.cash_balance * action
            
            # Calculate number of shares to buy
            shares_to_buy = np.floor(cash_to_use / (current_price * (1 + self.transaction_cost_pct)))
            
            if shares_to_buy > 0:
                # Update shares and cash
                cost = shares_to_buy * current_price * (1 + self.transaction_cost_pct)
                self.shares_held += shares_to_buy
                self.cash_balance -= cost
        
        elif action < 0:  # Sell
            # Calculate percentage of shares to sell (action is between -1 and 0)
            shares_to_sell = np.floor(self.shares_held * abs(action))
            
            if shares_to_sell > 0:
                # Update shares and cash
                proceeds = shares_to_sell * current_price * (1 - self.transaction_cost_pct)
                self.shares_held -= shares_to_sell
                self.cash_balance += proceeds
        
        # Else action == 0 (Hold): Do nothing
        
        # Move to next timestep
        self.current_step += 1
        
        # Calculate new total assets and update history
        self.total_assets = self.cash_balance + self.shares_held * current_price
        
        self.history['cash'].append(self.cash_balance)
        self.history['shares'].append(self.shares_held)
        self.history['total_assets'].append(self.total_assets)
        
        # Calculate reward as change in total assets
        reward = self.total_assets - previous_total_assets
        self.history['rewards'].append(reward)
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        # Terminated if balance is too low (failed episode)
        if self.total_assets <= 0.1 * self.initial_balance:
            terminated = True
        
        # Truncated if we've reached the end of data
        if self.current_step >= len(self.df):
            truncated = True
        
        # Get new observation
        obs = self._get_observation()
        
        # Info dictionary for debugging
        info = {
            'step': self.current_step,
            'cash_balance': self.cash_balance,
            'shares_held': self.shares_held,
            'total_assets': self.total_assets,
            'current_price': current_price,
            'action': action
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            observation: The initial observation
            info: Empty dictionary
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state variables
        self.cash_balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size  # Start after window_size to have enough history
        self.total_assets = self.initial_balance
        
        # Reset history
        self.history = {
            'cash': [self.initial_balance],
            'shares': [0],
            'total_assets': [self.initial_balance],
            'rewards': []
        }
        
        return self._get_observation(), {}
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode to render with
            
        Returns:
            None
        """
        if mode == 'human':
            # Get current price
            current_price = self.df.iloc[self.current_step - 1]['Close']
            
            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Cash Balance: ${self.cash_balance:.2f}")
            print(f"Shares Held: {self.shares_held}")
            print(f"Total Assets: ${self.total_assets:.2f}")
            print("-" * 50)
        else:
            raise NotImplementedError(f"Render mode {mode} is not implemented")

# Example usage (as a comment so it's not executed when imported)
"""
# Load data from CSV file
df = pd.read_csv('train_data.csv', index_col=0, parse_dates=True)

# Create discrete environment
discrete_env = DiscreteTradingEnvironment(df)

# Reset env and get initial observation
obs, _ = discrete_env.reset()

# Take a few steps
for _ in range(10):
    action = np.random.randint(0, 3)  # Random action
    obs, reward, terminated, truncated, info = discrete_env.step(action)
    discrete_env.render()
    if terminated or truncated:
        break

# Plot results
discrete_env.plot_portfolio_performance()
discrete_env.export_history()

# Create continuous environment
continuous_env = ContinuousTradingEnvironment(df)

# Reset env and get initial observation
obs, _ = continuous_env.reset()

# Take a few steps
for _ in range(10):
    action = np.array([np.random.uniform(-1, 1)])  # Random action
    obs, reward, terminated, truncated, info = continuous_env.step(action)
    continuous_env.render()
    if terminated or truncated:
        break

# Plot results
continuous_env.plot_portfolio_performance()
continuous_env.export_history('continuous_history.csv')
""" 