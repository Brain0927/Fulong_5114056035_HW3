"""
DQN 代理實現
包含：DQNAgent 和 DoubleDQNAgent
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from .models import create_model, copy_model
from .replay import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """標準 DQN 代理
    
    機制：
    - Replay Buffer (S1)
    - Target Network (S2)
    """
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.0001, 
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, model_type: str = "standard"):
        """初始化 DQN 代理
        
        Args:
            state_size: 狀態空間維度
            action_size: 動作空間大小
            lr: 學習率
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_decay: 探索率衰減
            epsilon_min: 最小探索率
            model_type: 模型類型 ("standard" 或 "dueling")
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model_type = model_type
        
        # 神經網絡
        self.q_network = create_model(action_size, state_size, model_type)
        self.target_network = create_model(action_size, state_size, model_type)
        self.update_target_network()
        
        # 優化器
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """選擇動作（ε-貪心策略）
        
        Args:
            state: 當前狀態
            training: 是否在訓練模式（False 時不使用 epsilon）
        
        Returns:
            選擇的動作
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # 貪心動作
        state = np.expand_dims(state, axis=0).astype(np.float32)
        q_values = self.q_network(state, training=False)
        return np.argmax(q_values[0].numpy())
    
    def update(self, batch):
        """使用一批數據更新 Q 網絡
        
        Args:
            batch: (states, actions, rewards, next_states, dones)
        
        Returns:
            損失值
        """
        states, actions, rewards, next_states, dones = batch
        
        # 轉換為 TensorFlow 張量
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # 當前 Q 值
            q_values = self.q_network(states, training=True)
            q_values = tf.gather_nd(q_values, 
                                    tf.stack([tf.range(len(actions)), actions], axis=1))
            
            # 目標 Q 值（使用目標網絡）
            next_q_values = tf.reduce_max(self.target_network(next_states, training=False), axis=1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            # 損失（MSE）
            loss = tf.reduce_mean(tf.square(q_values - target_q_values))
        
        # 反向傳播
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss.numpy()
    
    def update_target_network(self):
        """將目標網絡的權重更新為主網絡的權重"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def update_epsilon(self):
        """衰減探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_td_errors(self, batch):
        """計算 TD 誤差（用於優先級更新）
        
        Args:
            batch: (states, actions, rewards, next_states, dones)
        
        Returns:
            TD 誤差數組
        """
        states, actions, rewards, next_states, dones = batch
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # 當前 Q 值
        q_values = self.q_network(states, training=False)
        q_values = tf.gather_nd(q_values, 
                                tf.stack([tf.range(len(actions)), actions], axis=1))
        
        # 目標 Q 值
        next_q_values = tf.reduce_max(self.target_network(next_states, training=False), axis=1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # TD 誤差
        td_errors = (q_values - target_q_values).numpy()
        return td_errors


class DoubleDQNAgent(DQNAgent):
    """雙 DQN 代理
    
    機制：
    - Replay Buffer (S1)
    - Target Network (S2)
    - Double DQN (S3): 使用主網絡選擇，目標網絡評估
    - Dueling DQN (S4): 可選模型
    """
    
    def update(self, batch):
        """使用雙 DQN 更新 Q 網絡
        
        雙 DQN 的核心：選擇和評估解耦
        - 主網絡：選擇最優動作
        - 目標網絡：評估該動作的 Q 值
        
        Args:
            batch: (states, actions, rewards, next_states, dones)
        
        Returns:
            損失值
        """
        states, actions, rewards, next_states, dones = batch
        
        # 轉換為 TensorFlow 張量
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # 當前 Q 值
            q_values = self.q_network(states, training=True)
            q_values = tf.gather_nd(q_values, 
                                    tf.stack([tf.range(len(actions)), actions], axis=1))
            
            # 雙 DQN：使用主網絡選擇動作，目標網絡評估
            # 1. 主網絡選擇最優動作
            next_actions = tf.argmax(self.q_network(next_states, training=False), axis=1)
            next_actions = tf.cast(next_actions, tf.int32)  # 轉換為 int32
            
            # 2. 目標網絡評估該動作的 Q 值
            next_q_values_all = self.target_network(next_states, training=False)
            next_q_values = tf.gather_nd(next_q_values_all,
                                        tf.stack([tf.range(len(next_actions)), next_actions], axis=1))
            
            # 目標 Q 值
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            # 損失（MSE）
            loss = tf.reduce_mean(tf.square(q_values - target_q_values))
        
        # 反向傳播
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss.numpy()
    
    def get_td_errors(self, batch):
        """計算 TD 誤差（雙 DQN 版本）
        
        Args:
            batch: (states, actions, rewards, next_states, dones)
        
        Returns:
            TD 誤差數組
        """
        states, actions, rewards, next_states, dones = batch
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # 當前 Q 值
        q_values = self.q_network(states, training=False)
        q_values = tf.gather_nd(q_values, 
                                tf.stack([tf.range(len(actions)), actions], axis=1))
        
        # 雙 DQN：使用主網絡選擇，目標網絡評估
        next_actions = tf.argmax(self.q_network(next_states, training=False), axis=1)
        next_q_values_all = self.target_network(next_states, training=False)
        next_q_values = tf.gather_nd(next_q_values_all,
                                    tf.stack([tf.range(len(next_actions)), next_actions], axis=1))
        
        # 目標 Q 值
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # TD 誤差
        td_errors = (q_values - target_q_values).numpy()
        return td_errors
