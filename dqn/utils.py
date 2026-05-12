"""
工具函數
包含：Logger, set_seed, plot_training_stats 等
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime


class Logger:
    """訓練日誌記錄器"""
    
    def __init__(self, log_file: str = None):
        """初始化日誌記錄器
        
        Args:
            log_file: 日誌文件路徑（可選）
        """
        self.log_file = log_file
        self.logs = []
        
        if log_file:
            with open(log_file, 'w') as f:
                f.write(f"Training Log - {datetime.now()}\n")
                f.write("=" * 80 + "\n\n")
    
    def log_episode(self, episode: int, reward: float):
        """記錄回合信息
        
        Args:
            episode: 回合編號
            reward: 回合獎勵
        """
        msg = f"Episode {episode}: Reward = {reward:.2f}"
        self.logs.append(msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + "\n")
    
    def log_target_update(self, step: int):
        """記錄目標網絡更新
        
        Args:
            step: 更新步數
        """
        msg = f"[{step}] Target Network Updated"
        self.logs.append(msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + "\n")
    
    def log_evaluation(self, avg_reward: float, rewards: list):
        """記錄評估結果
        
        Args:
            avg_reward: 平均獎勵
            rewards: 獎勵列表
        """
        msg = f"Evaluation: Avg Reward = {avg_reward:.2f}, Min = {min(rewards):.2f}, Max = {max(rewards):.2f}"
        self.logs.append(msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + "\n")
    
    def log_info(self, message: str):
        """記錄自定義信息
        
        Args:
            message: 信息內容
        """
        self.logs.append(message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + "\n")


def set_seed(seed: int):
    """設置隨機種子以實現可重現性
    
    Args:
        seed: 種子值
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def plot_training_stats(episode_rewards: list, episode_losses: list = None, 
                       save_path: str = None, window_size: int = 100):
    """繪製訓練統計圖表
    
    Args:
        episode_rewards: 回合獎勵列表
        episode_losses: 損失列表（可選）
        save_path: 保存路徑（可選）
        window_size: 移動平均窗口大小
    """
    fig, axes = plt.subplots(1, 2 if episode_losses is not None else 1, figsize=(14, 5))
    
    if episode_losses is None:
        axes = [axes]
    
    # 繪製獎勵
    axes[0].plot(episode_rewards, alpha=0.5, label='Episode Reward')
    
    # 移動平均
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0].plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                    'r-', linewidth=2, label=f'{window_size}-episode Moving Average')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 繪製損失（如果提供）
    if episode_losses is not None:
        axes[1].plot(episode_losses, alpha=0.5, label='Training Loss')
        
        if len(episode_losses) >= window_size:
            moving_avg_loss = np.convolve(episode_losses, np.ones(window_size)/window_size, mode='valid')
            axes[1].plot(range(window_size-1, len(episode_losses)), moving_avg_loss,
                        'r-', linewidth=2, label=f'{window_size}-step Moving Average')
        
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    
    plt.show()


def smooth_curve(data: np.ndarray, window_size: int = 10) -> np.ndarray:
    """使用移動平均平滑曲線
    
    Args:
        data: 數據數組
        window_size: 窗口大小
    
    Returns:
        平滑後的數據
    """
    if len(data) < window_size:
        return data
    
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def calculate_moving_average(data: list, window_size: int = 100) -> list:
    """計算移動平均
    
    Args:
        data: 數據列表
        window_size: 窗口大小
    
    Returns:
        移動平均列表
    """
    if len(data) < window_size:
        return data
    
    moving_avg = []
    for i in range(len(data) - window_size + 1):
        avg = np.mean(data[i:i+window_size])
        moving_avg.append(avg)
    
    return moving_avg


def print_statistics(episode_rewards: list, episode_losses: list = None):
    """打印訓練統計信息
    
    Args:
        episode_rewards: 回合獎勵列表
        episode_losses: 損失列表（可選）
    """
    print("\n" + "=" * 60)
    print("Training Statistics")
    print("=" * 60)
    
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"Best Reward: {max(episode_rewards):.2f}")
    print(f"Worst Reward: {min(episode_rewards):.2f}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Last 100 Avg: {np.mean(episode_rewards[-100:]):.2f}")
    
    if episode_losses is not None:
        print(f"\nTotal Training Steps: {len(episode_losses)}")
        print(f"Best Loss: {min(episode_losses):.6f}")
        print(f"Worst Loss: {max(episode_losses):.6f}")
        print(f"Average Loss: {np.mean(episode_losses):.6f}")
        print(f"Last 100 Avg Loss: {np.mean(episode_losses[-100:]):.6f}")
    
    print("=" * 60 + "\n")
