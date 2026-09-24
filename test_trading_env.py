import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_environments import DiscreteTradingEnvironment, ContinuousTradingEnvironment

def test_discrete_environment():
    """
    Test the discrete trading environment with random actions.
    """
    print("Testing Discrete Trading Environment...")
    
    # Load data from CSV file
    df = pd.read_csv('train_data.csv', index_col=0, parse_dates=True)
    
    # Create discrete environment
    env = DiscreteTradingEnvironment(df)
    
    # Reset env and get initial observation
    obs, _ = env.reset()
    
    # Print initial state
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial cash balance: ${env.cash_balance:.2f}")
    print(f"Initial shares held: {env.shares_held}")
    print(f"Initial total assets: ${env.total_assets:.2f}")
    print("-" * 50)
    
    # Take 100 steps with random actions
    total_steps = 100
    for step in range(total_steps):
        # Random action (0=Hold, 1=Buy, 2=Sell)
        action = np.random.randint(0, 3)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print every 10th step
        if step % 10 == 0:
            print(f"Step {step}:")
            print(f"  Action: {info['action']}")
            print(f"  Price: ${info['current_price']:.2f}")
            print(f"  Reward: ${reward:.2f}")
            print(f"  Cash: ${info['cash_balance']:.2f}")
            print(f"  Shares: {info['shares_held']}")
            print(f"  Total Assets: ${info['total_assets']:.2f}")
            print("-" * 50)
        
        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            break
    
    # Plot results
    env.plot_portfolio_performance()
    env.export_history('discrete_trading_history.csv')
    
    return env

def test_continuous_environment():
    """
    Test the continuous trading environment with random actions.
    """
    print("\nTesting Continuous Trading Environment...")
    
    # Load data from CSV file
    df = pd.read_csv('train_data.csv', index_col=0, parse_dates=True)
    
    # Create continuous environment
    env = ContinuousTradingEnvironment(df)
    
    # Reset env and get initial observation
    obs, _ = env.reset()
    
    # Print initial state
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial cash balance: ${env.cash_balance:.2f}")
    print(f"Initial shares held: {env.shares_held}")
    print(f"Initial total assets: ${env.total_assets:.2f}")
    print("-" * 50)
    
    # Take 100 steps with random actions
    total_steps = 100
    for step in range(total_steps):
        # Random action between -1 and 1
        action = np.array([np.random.uniform(-1, 1)])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print every 10th step
        if step % 10 == 0:
            print(f"Step {step}:")
            print(f"  Action: {info['action']:.4f}")
            print(f"  Price: ${info['current_price']:.2f}")
            print(f"  Reward: ${reward:.2f}")
            print(f"  Cash: ${info['cash_balance']:.2f}")
            print(f"  Shares: {info['shares_held']}")
            print(f"  Total Assets: ${info['total_assets']:.2f}")
            print("-" * 50)
        
        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            break
    
    # Plot results
    env.plot_portfolio_performance()
    env.export_history('continuous_trading_history.csv')
    
    return env

if __name__ == "__main__":
    # Test both environments
    discrete_env = test_discrete_environment()
    continuous_env = test_continuous_environment()
    
    # Compare performance
    plt.figure(figsize=(12, 6))
    plt.plot(discrete_env.history['total_assets'], label='Discrete Actions')
    plt.plot(continuous_env.history['total_assets'], label='Continuous Actions')
    plt.title('Portfolio Value: Discrete vs. Continuous Actions')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison.png')
    plt.close()
    
    print("\nComparison saved to comparison.png")
    print("Test complete!") 