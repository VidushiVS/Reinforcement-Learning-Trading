import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001, 
                 lookback_window=30, max_position=1.0):
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.max_position = max_position
        
        # state variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.total_value = initial_balance
        self.trades = []
        
        # prepare data
        self._prepare_data()
        
        # action space: 0=hold, 1=buy, 2=sell
        self.action_space = 3
        
        # state space: price features + portfolio state
        self.state_space = self.lookback_window * 7 + 4  # 7 features per timestep + 4 portfolio features
        
    def _prepare_data(self):
        # calculate technical indicators
        close = self.data['Close']
        
        # returns
        self.data['returns'] = close.pct_change()
        
        # moving averages
        self.data['sma_5'] = close.rolling(5).mean()
        self.data['sma_20'] = close.rolling(20).mean()
        self.data['sma_50'] = close.rolling(50).mean()
        
        # rsi
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # bollinger bands
        bb_period = 20
        bb_std = 2
        sma = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        self.data['bb_upper'] = sma + (std * bb_std)
        self.data['bb_lower'] = sma - (std * bb_std)
        self.data['bb_position'] = (close - self.data['bb_lower']) / (self.data['bb_upper'] - self.data['bb_lower'])
        
        # macd
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        self.data['macd'] = exp1 - exp2
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
        
        # volume indicators
        if 'Volume' in self.data.columns:
            self.data['volume_sma'] = self.data['Volume'].rolling(20).mean()
            self.data['volume_ratio'] = self.data['Volume'] / self.data['volume_sma']
        else:
            self.data['volume_ratio'] = 1.0
        
        # volatility
        self.data['volatility'] = self.data['returns'].rolling(20).std()
        
        # normalize features
        feature_cols = ['returns', 'rsi', 'bb_position', 'macd', 'macd_signal', 'volume_ratio', 'volatility']
        
        for col in feature_cols:
            if col in self.data.columns:
                # z-score normalization with rolling window
                rolling_mean = self.data[col].rolling(252, min_periods=50).mean()
                rolling_std = self.data[col].rolling(252, min_periods=50).std()
                self.data[f'{col}_norm'] = (self.data[col] - rolling_mean) / (rolling_std + 1e-8)
        
        # drop nan rows
        self.data = self.data.dropna()
        
        # reset index
        self.data = self.data.reset_index(drop=True)
        
    def reset(self):
        # reset environment to initial state
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_value = self.initial_balance
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self):
        # construct state vector
        if self.current_step < self.lookback_window:
            # pad with zeros if not enough history
            price_features = np.zeros(self.lookback_window * 7)
        else:
            # get recent price features
            start_idx = self.current_step - self.lookback_window
            end_idx = self.current_step
            
            recent_data = self.data.iloc[start_idx:end_idx]
            
            price_features = []
            feature_cols = ['returns_norm', 'rsi_norm', 'bb_position_norm', 
                           'macd_norm', 'macd_signal_norm', 'volume_ratio_norm', 'volatility_norm']
            
            for _, row in recent_data.iterrows():
                for col in feature_cols:
                    if col in row:
                        val = row[col]
                        if pd.isna(val):
                            val = 0.0
                        price_features.append(np.clip(val, -3, 3))  # clip outliers
                    else:
                        price_features.append(0.0)
            
            price_features = np.array(price_features)
        
        # portfolio state
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_features = np.array([
            self.position,  # current position
            self.balance / self.initial_balance,  # normalized balance
            self.total_value / self.initial_balance,  # normalized total value
            len(self.trades) / 100.0  # normalized trade count
        ])
        
        # combine features
        state = np.concatenate([price_features, portfolio_features])
        
        # ensure fixed size
        if len(state) != self.state_space:
            # pad or truncate to exact size
            if len(state) < self.state_space:
                state = np.pad(state, (0, self.state_space - len(state)))
            else:
                state = state[:self.state_space]
        
        return state.astype(np.float32)
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        
        # execute action
        reward = 0.0
        new_position = self.position
        
        if action == 1:  # buy
            if self.position < self.max_position:
                # calculate how much to buy
                trade_size = min(0.1, self.max_position - self.position)  # 10% increments
                cost = trade_size * current_price * abs(trade_size) * self.initial_balance
                transaction_costs = cost * self.transaction_cost
                
                if self.balance >= cost + transaction_costs:
                    new_position = self.position + trade_size
                    self.balance -= (cost + transaction_costs)
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'price': current_price,
                        'size': trade_size,
                        'cost': transaction_costs
                    })
        
        elif action == 2:  # sell
            if self.position > -self.max_position:
                # calculate how much to sell
                trade_size = min(0.1, self.position + self.max_position)  # 10% increments
                proceeds = trade_size * current_price * abs(trade_size) * self.initial_balance
                transaction_costs = proceeds * self.transaction_cost
                
                new_position = self.position - trade_size
                self.balance += (proceeds - transaction_costs)
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'size': trade_size,
                    'cost': transaction_costs
                })
        
        # update position
        self.position = np.clip(new_position, -self.max_position, self.max_position)
        
        # calculate total portfolio value
        position_value = self.position * current_price * self.initial_balance
        self.total_value = self.balance + position_value
        
        # calculate reward
        reward = self._calculate_reward()
        
        # move to next step
        self.current_step += 1
        
        # check if episode is done
        done = (self.current_step >= len(self.data) - 1) or (self.total_value <= 0.1 * self.initial_balance)
        
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done
    
    def _calculate_reward(self):
        # multi-component reward function
        if len(self.trades) < 2:
            return 0.0
        
        # 1. portfolio return
        portfolio_return = (self.total_value - self.initial_balance) / self.initial_balance
        
        # 2. recent performance (last 10 steps)
        if self.current_step >= 10:
            recent_prices = self.data.iloc[self.current_step-10:self.current_step+1]['Close']
            recent_return = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            position_pnl = self.position * recent_return
        else:
            position_pnl = 0.0
        
        # 3. risk penalty (large positions in high volatility)
        current_vol = self.data.iloc[self.current_step]['volatility_norm']
        risk_penalty = abs(self.position) * max(0, current_vol) * 0.1
        
        # 4. transaction cost penalty
        recent_costs = sum([trade['cost'] for trade in self.trades[-5:]])  # last 5 trades
        cost_penalty = recent_costs / self.initial_balance
        
        # 5. sharpe-like ratio
        if len(self.trades) > 10:
            recent_returns = []
            for i in range(max(0, len(self.trades)-10), len(self.trades)):
                trade = self.trades[i]
                ret = self.position * self.data.iloc[trade['step']]['returns']
                recent_returns.append(ret)
            
            if len(recent_returns) > 0 and np.std(recent_returns) > 0:
                sharpe_component = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
            else:
                sharpe_component = 0.0
        else:
            sharpe_component = 0.0
        
        # combine reward components
        reward = (
            portfolio_return * 100 +           # main return component
            position_pnl * 50 +                # recent position performance
            sharpe_component * 10 +            # risk-adjusted return
            -risk_penalty * 20 +               # volatility penalty
            -cost_penalty * 100                # transaction cost penalty
        )
        
        return reward

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        # build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # dueling dqn components
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        self.advantage_head = nn.Linear(hidden_sizes[-1], action_size)
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # standard dqn
        features = self.network[:-1](x)  # all layers except last
        q_values = self.network[-1](features)  # final layer
        
        return q_values

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNTrader:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # replay buffer
        self.memory = ReplayBuffer()
        self.batch_size = 64
        
        # training history
        self.losses = []
        self.rewards = []
        self.epsilons = []
        
        # update target network
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # get q values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # sample batch
        experiences = self.memory.sample(self.batch_size)
        
        # convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences if e.next_state is not None]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # current q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # next q values from target network
        next_q_values = torch.zeros(self.batch_size).to(self.device)
        if len(next_states) > 0:
            with torch.no_grad():
                next_q_values[~dones] = self.target_network(next_states).max(1)[0]
        
        # target q values
        target_q_values = rewards + (self.gamma * next_q_values)
        
        # calculate loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # gradient clipping
        self.optimizer.step()
        
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # store metrics
        self.losses.append(loss.item())
        self.epsilons.append(self.epsilon)
    
    def train(self, env, episodes=1000, target_update_freq=100):
        print(f"Training DQN agent for {episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # choose action
                action = self.act(state)
                
                # take step
                next_state, reward, done = env.step(action)
                
                # store experience
                self.remember(state, action, reward, next_state, done)
                
                # update metrics
                total_reward += reward
                steps += 1
                
                # move to next state
                state = next_state
                
                # train network
                if len(self.memory) > self.batch_size:
                    self.replay()
                
                if done:
                    break
            
            # update target network
            if episode % target_update_freq == 0:
                self.update_target_network()
            
            # store episode metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            self.rewards.append(total_reward)
            
            # print progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.0f}, Epsilon: {self.epsilon:.3f}")
        
        print("Training completed!")
        return episode_rewards
    
    def test(self, env, episodes=10):
        print(f"Testing agent for {episodes} episodes...")
        
        test_rewards = []
        test_portfolios = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            portfolio_values = [env.initial_balance]
            
            while True:
                # choose action (no exploration)
                action = self.act(state, training=False)
                
                # take step
                next_state, reward, done = env.step(action)
                
                total_reward += reward
                portfolio_values.append(env.total_value)
                
                state = next_state
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            test_portfolios.append(portfolio_values)
            
            print(f"Test Episode {episode + 1}: Reward = {total_reward:.2f}, "
                  f"Final Value = ${env.total_value:.2f}")
        
        return test_rewards, test_portfolios

def load_and_prepare_data(symbol, start_date, end_date):
    # load market data
    print(f"Loading data for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if len(data) < 500:
        raise ValueError(f"Insufficient data for {symbol}")
    
    print(f"Loaded {len(data)} days of data")
    return data

def plot_training_results(agent, test_portfolios, symbol):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # training rewards
    if len(agent.rewards) > 0:
        axes[0, 0].plot(agent.rewards)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # training loss
    if len(agent.losses) > 0:
        axes[0, 1].plot(agent.losses)
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # epsilon decay
    if len(agent.epsilons) > 0:
        axes[1, 0].plot(agent.epsilons)
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
    
    # portfolio performance
    if test_portfolios:
        for i, portfolio in enumerate(test_portfolios):
            axes[1, 1].plot(portfolio, label=f'Test {i+1}', alpha=0.7)
        
        # buy and hold benchmark
        axes[1, 1].axhline(y=test_portfolios[0][0], color='red', 
                          linestyle='--', label='Initial Value')
        axes[1, 1].set_title(f'Portfolio Performance - {symbol}')
        axes[1, 1].set_xlabel('Trading Days')
        axes[1, 1].set_ylabel('Portfolio Value ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Deep Reinforcement Learning Trading Agent")
    print("=========================================\n")
    
    # parameters
    symbol = "SPY"  # s&p 500 etf
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    # load data
    data = load_and_prepare_data(symbol, start_date, end_date)
    
    # split data for training and testing
    split_point = int(len(data) * 0.8)
    train_data = data.iloc[:split_point].copy()
    test_data = data.iloc[split_point:].copy().reset_index(drop=True)
    
    print(f"Training data: {len(train_data)} days")
    print(f"Testing data: {len(test_data)} days")
    
    # create environment
    train_env = TradingEnvironment(train_data, initial_balance=10000)
    test_env = TradingEnvironment(test_data, initial_balance=10000)
    
    print(f"State space size: {train_env.state_space}")
    print(f"Action space size: {train_env.action_space}")
    
    # create agent
    agent = DQNTrader(
        state_size=train_env.state_space,
        action_size=train_env.action_space,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )
    
    # train agent
    print("\n1. Training Phase:")
    training_rewards = agent.train(train_env, episodes=500, target_update_freq=50)
    
    # test agent
    print("\n2. Testing Phase:")
    test_rewards, test_portfolios = agent.test(test_env, episodes=5)
    
    # calculate performance metrics
    print("\n3. Performance Analysis:")
    avg_test_reward = np.mean(test_rewards)
    avg_final_value = np.mean([portfolio[-1] for portfolio in test_portfolios])
    total_return = (avg_final_value - 10000) / 10000
    
    print(f"Average test reward: {avg_test_reward:.2f}")
    print(f"Average final portfolio value: ${avg_final_value:.2f}")
    print(f"Average total return: {total_return:.2%}")
    
    # benchmark comparison
    test_data_return = (test_data['Close'].iloc[-1] - test_data['Close'].iloc[0]) / test_data['Close'].iloc[0]
    print(f"Buy-and-hold return: {test_data_return:.2%}")
    print(f"Agent outperformance: {total_return - test_data_return:.2%}")
    
    # plot results
    print("\n4. Plotting results...")
    plot_training_results(agent, test_portfolios, symbol)
    
    print("\nTraining and testing completed!")

if __name__ == "__main__":
    main()
