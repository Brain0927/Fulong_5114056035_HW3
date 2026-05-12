"""
經驗回放緩衝區
包含：ReplayBuffer 和 PrioritizedReplayBuffer
"""
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """標準經驗回放緩衝區
    
    存儲轉移 (state, action, reward, next_state, done) 並支持隨機採樣
    """
    
    def __init__(self, max_size: int = 100000):
        """初始化回放緩衝區
        
        Args:
            max_size: 最大存儲容量
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        """添加轉移到緩衝區
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一個狀態
            done: 是否終止
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """隨機採樣一批轉移
        
        Args:
            batch_size: 批次大小
        
        Returns:
            states, actions, rewards, next_states, dones (均為 numpy 數組)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size {len(self.buffer)} < batch_size {batch_size}")
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """優先經驗回放（PER）緩衝區
    
    根據 TD 誤差優先級採樣 (priority = |TD_error| + epsilon)
    """
    
    def __init__(self, max_size: int = 100000, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6):
        """初始化優先回放緩衝區
        
        Args:
            max_size: 最大存儲容量
            alpha: 優先級指數 (0=uniform, 1=full priority)
            beta: 重要性採樣權重指數
            epsilon: 最小優先級（避免 0）
        """
        super().__init__(max_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.priorities = deque(maxlen=max_size)
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """添加轉移到緩衝區（最大優先級）
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一個狀態
            done: 是否終止
        """
        super().add(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """根據 TD 誤差更新優先級
        
        Args:
            indices: 轉移索引
            td_errors: TD 誤差值
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int):
        """根據優先級採樣一批轉移
        
        Args:
            batch_size: 批次大小
        
        Returns:
            states, actions, rewards, next_states, dones, weights, indices
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size {len(self.buffer)} < batch_size {batch_size}")
        
        # 計算採樣概率
        priorities = np.array(list(self.priorities))
        probabilities = priorities / np.sum(priorities)
        
        # 根據概率採樣索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # 計算重要性採樣權重
        weights = (1 / (len(self.buffer) * probabilities[indices])) ** self.beta
        weights /= np.max(weights)  # 正規化權重
        
        # 提取數據
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), weights, indices)
    
    def __len__(self):
        return len(self.buffer)
