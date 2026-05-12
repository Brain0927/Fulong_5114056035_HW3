"""
Stage 1 訓練腳本 (靜態環境)
使用機制: S1-Replay Buffer

DQN 機制:
- S1: Experience Replay
- S2: Target Network
"""
import numpy as np
import gymnasium as gym
from dqn import TrainingLoop, Logger, set_seed, plot_training_stats, print_statistics
from env_adapter import create_env, print_env_info


def main():
    """Stage 1 訓練主程序"""
    
    # 配置
    CONFIG = {
        'env_name': 'CartPole-v1',
        'seed': 42,
        'num_episodes': 500,
        'batch_size': 64,
        'update_target_freq': 1000,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'model_type': 'standard',  # 標準 DQN
        'stage': 1,
    }
    
    # 設置隨機種子
    set_seed(CONFIG['seed'])
    
    print("\n" + "=" * 80)
    print("STAGE 1: DQN with Experience Replay")
    print("=" * 80)
    print(f"Configuration: {CONFIG}\n")
    
    # 創建環境
    env = create_env(CONFIG['env_name'])
    env_info = print_env_info(env)
    
    # 創建日誌記錄器
    logger = Logger(log_file='stage1_training.log')
    logger.log_info(f"Stage 1 Training Configuration: {CONFIG}")
    logger.log_info(f"Environment: {env_info['name']}")
    logger.log_info(f"State Size: {env_info['state_size']}, Action Size: {env_info['action_size']}\n")
    
    # 創建訓練循環
    print("Creating training loop...")
    training_loop = TrainingLoop.create(
        env=env,
        stage=CONFIG['stage'],
        model_type=CONFIG['model_type'],
        lr=CONFIG['learning_rate'],
        gamma=CONFIG['gamma'],
        logger=logger
    )
    
    print(f"✓ Training loop created")
    print(f"✓ Agent Type: DQNAgent")
    print(f"✓ Replay Buffer: Standard (Experience Replay)")
    print(f"✓ Model Type: {CONFIG['model_type']}\n")
    
    # 訓練
    print(f"Training for {CONFIG['num_episodes']} episodes...")
    print("-" * 80)
    
    training_loop.train(
        num_episodes=CONFIG['num_episodes'],
        update_target_freq=CONFIG['update_target_freq'],
        batch_size=CONFIG['batch_size']
    )
    
    print("-" * 80)
    print("Training completed!\n")
    
    # 評估
    print("Evaluating trained agent...")
    avg_reward = training_loop.evaluate(num_episodes=100)
    print(f"Average Evaluation Reward: {avg_reward:.2f}\n")
    
    logger.log_info(f"\nTraining completed!")
    logger.log_info(f"Average Evaluation Reward: {avg_reward:.2f}")
    
    # 統計
    stats = training_loop.get_statistics()
    print_statistics(training_loop.episode_rewards, training_loop.episode_losses)
    
    logger.log_info(f"\nFinal Statistics: {stats}")
    
    # 保存圖表
    print("Saving training graphs...")
    plot_training_stats(
        training_loop.episode_rewards,
        training_loop.episode_losses,
        save_path='stage1_training_stats.png'
    )
    
    # 清理
    env.close()
    
    print("\n" + "=" * 80)
    print("STAGE 1 COMPLETED")
    print("=" * 80)
    print("\nKey Mechanisms Enabled:")
    print("  ✓ S1: Experience Replay Buffer")
    print("  ✓ S2: Target Network")
    print("\nGenerated Files:")
    print("  • stage1_training.log")
    print("  • stage1_training_stats.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
