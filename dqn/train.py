"""
訓練循環實現
支持自動機制選擇和多種訓練模式
"""
import numpy as np
from .agent import DQNAgent, DoubleDQNAgent
from .replay import ReplayBuffer, PrioritizedReplayBuffer


class TrainingLoop:
    """訓練循環管理器
    
    自動機制選擇：
    - Stage 1 (static): DQNAgent + ReplayBuffer
    - Stage 2 (player): DoubleDQNAgent + ReplayBuffer
    - Stage 3 (random): DoubleDQNAgent + PrioritizedReplayBuffer
    """
    
    def __init__(self, env, agent, replay_buffer, logger=None):
        """初始化訓練循環
        
        Args:
            env: 環境
            agent: DQN 代理
            replay_buffer: 回放緩衝區
            logger: 日誌記錄器
        """
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.episode_rewards = []
        self.episode_losses = []
        self.update_steps = 0
    
    @staticmethod
    def create(env, stage: int = 1, model_type: str = "standard", 
               lr: float = 0.0001, gamma: float = 0.99, logger=None):
        """工廠方法：根據階段自動創建訓練循環
        
        Args:
            env: 環境
            stage: 訓練階段 (1, 2, 或 3)
            model_type: 模型類型 ("standard" 或 "dueling")
            lr: 學習率
            gamma: 折扣因子
            logger: 日誌記錄器
        
        Returns:
            TrainingLoop 實例
        """
        # 獲取環境信息
        if hasattr(env, 'observation_space'):
            if hasattr(env.observation_space, 'shape'):
                state_size = env.observation_space.shape[0]
            else:
                state_size = env.observation_space.n
        else:
            state_size = 4  # 默認 CartPole
        
        action_size = env.action_space.n
        
        # 根據階段選擇代理和回放緩衝區
        if stage >= 2:
            # Stage 2+: 使用 Double DQN
            agent = DoubleDQNAgent(state_size, action_size, lr=lr, gamma=gamma, 
                                  model_type=model_type)
        else:
            # Stage 1: 使用標準 DQN
            agent = DQNAgent(state_size, action_size, lr=lr, gamma=gamma, 
                           model_type=model_type)
        
        if stage == 3:
            # Stage 3: 使用優先回放
            replay_buffer = PrioritizedReplayBuffer(max_size=100000, alpha=0.6, beta=0.4)
        else:
            # Stage 1-2: 使用標準回放
            replay_buffer = ReplayBuffer(max_size=100000)
        
        return TrainingLoop(env, agent, replay_buffer, logger)
    
    def train_episode(self) -> float:
        """訓練單個回合
        
        Returns:
            回合獎勵
        """
        state, info = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 選擇動作
            action = self.agent.select_action(state, training=True)
            
            # 執行動作
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 存儲轉移
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            # 如果緩衝區中有足夠的數據，進行更新
            if len(self.replay_buffer) >= 64:
                self._update_networks(64)
        
        # 衰減探索率
        self.agent.update_epsilon()
        
        self.episode_rewards.append(total_reward)
        if self.logger:
            self.logger.log_episode(len(self.episode_rewards), total_reward)
        
        return total_reward
    
    def train(self, num_episodes: int, update_target_freq: int = 1000, batch_size: int = 64):
        """訓練多個回合
        
        Args:
            num_episodes: 回合數
            update_target_freq: 目標網絡更新頻率（每 N 步）
            batch_size: 批次大小
        """
        for episode in range(num_episodes):
            reward = self.train_episode()
            
            # 定期更新目標網絡
            if (episode + 1) % update_target_freq == 0:
                self.agent.update_target_network()
                if self.logger:
                    self.logger.log_target_update(episode + 1)
    
    def _update_networks(self, batch_size: int):
        """更新網絡（處理優先回放）
        
        Args:
            batch_size: 批次大小
        """
        # 採樣批次
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, weights, indices = \
                self.replay_buffer.sample(batch_size)
            
            # 更新代理
            loss = self.agent.update((states, actions, rewards, next_states, dones))
            
            # 計算 TD 誤差並更新優先級
            batch = (states, actions, rewards, next_states, dones)
            td_errors = self.agent.get_td_errors(batch)
            self.replay_buffer.update_priorities(indices, td_errors)
            
            # 應用重要性採樣權重到損失
            weighted_loss = loss * np.mean(weights)
        else:
            # 標準回放緩衝區
            batch = self.replay_buffer.sample(batch_size)
            loss = self.agent.update(batch)
            weighted_loss = loss
        
        self.episode_losses.append(weighted_loss)
        self.update_steps += 1
    
    def evaluate(self, num_episodes: int = 100) -> float:
        """評估代理性能
        
        Args:
            num_episodes: 評估回合數
        
        Returns:
            平均獎勵
        """
        total_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 不使用探索，只進行貪心動作
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        
        if self.logger:
            self.logger.log_evaluation(avg_reward, total_rewards)
        
        return avg_reward
    
    def get_statistics(self):
        """獲取訓練統計信息
        
        Returns:
            包含統計信息的字典
        """
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0,
            'max_reward': np.max(self.episode_rewards) if len(self.episode_rewards) > 0 else 0,
            'min_reward': np.min(self.episode_rewards) if len(self.episode_rewards) > 0 else 0,
            'avg_loss': np.mean(self.episode_losses[-100:]) if len(self.episode_losses) > 0 else 0,
            'update_steps': self.update_steps,
        }
