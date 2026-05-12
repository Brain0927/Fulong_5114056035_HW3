"""
DQN 神經網絡模型定義
包含：QNetwork, DuelingQNetwork 和工廠函數
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class QNetwork(keras.Model):
    """標準 DQN 神經網絡
    
    架構：Input -> Dense(128) -> ReLU -> Dense(128) -> ReLU -> Dense(action_size)
    """
    
    def __init__(self, action_size, state_size=None, name="q_network"):
        super(QNetwork, self).__init__(name=name)
        self.action_size = int(action_size)
        self.state_size = state_size
        
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(self.action_size, activation=None)
    
    def call(self, state, training=False):
        """前向傳播
        
        Args:
            state: 狀態輸入 (batch_size, state_size)
            training: 是否在訓練模式
        
        Returns:
            Q 值 (batch_size, action_size)
        """
        x = self.dense1(state, training=training)
        x = self.dense2(x, training=training)
        q_values = self.output_layer(x, training=training)
        return q_values
    
    def get_config(self):
        return {"action_size": self.action_size, "state_size": self.state_size}


class DuelingQNetwork(keras.Model):
    """決鬥 DQN 神經網絡
    
    架構：共享層 -> V值流 + A值流 -> Q = V + (A - mean(A))
    """
    
    def __init__(self, action_size, state_size=None, name="dueling_q_network"):
        super(DuelingQNetwork, self).__init__(name=name)
        self.action_size = int(action_size)
        self.state_size = state_size
        
        # 共享層
        self.shared_dense1 = layers.Dense(128, activation='relu')
        self.shared_dense2 = layers.Dense(128, activation='relu')
        
        # V值流（狀態價值）
        self.v_dense = layers.Dense(64, activation='relu')
        self.v_output = layers.Dense(1, activation=None)
        
        # A值流（動作優勢）
        self.a_dense = layers.Dense(64, activation='relu')
        self.a_output = layers.Dense(self.action_size, activation=None)
    
    def call(self, state, training=False):
        """前向傳播
        
        Args:
            state: 狀態輸入 (batch_size, state_size)
            training: 是否在訓練模式
        
        Returns:
            Q 值 (batch_size, action_size)
        """
        # 共享層
        x = self.shared_dense1(state, training=training)
        x = self.shared_dense2(x, training=training)
        
        # V值流
        v = self.v_dense(x, training=training)
        v = self.v_output(v, training=training)  # (batch, 1)
        
        # A值流
        a = self.a_dense(x, training=training)
        a = self.a_output(a, training=training)  # (batch, action_size)
        
        # 決鬥層：Q = V + (A - mean(A))
        q_values = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        
        return q_values
    
    def get_config(self):
        return {"action_size": self.action_size, "state_size": self.state_size}


def create_model(action_size, state_size=None, model_type="standard"):
    """工廠函數：創建 DQN 模型
    
    Args:
        action_size: 動作空間大小
        state_size: 狀態空間大小
        model_type: 模型類型 ("standard" 或 "dueling")
    
    Returns:
        DQN 模型實例
    """
    action_size = int(action_size)
    
    if model_type == "standard":
        return QNetwork(action_size, state_size)
    elif model_type == "dueling":
        return DuelingQNetwork(action_size, state_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'standard' or 'dueling'")


def copy_model(model):
    """複製模型的權重
    
    Args:
        model: 源模型
    
    Returns:
        具有相同權重的新模型
    """
    new_model = keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())
    return new_model

