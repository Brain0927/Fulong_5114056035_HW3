"""
Stage 2 訓練腳本 (玩家環境)
使用機制: S1-Replay Buffer, S2-Target Network, S3-Double DQN, S4-Dueling DQN

DQN 機制:
- S1: Experience Replay
- S2: Target Network
- S3: Double DQN (解耦選擇和評估)
- S4: Dueling DQN (V+A 分離)
"""
import numpy as np
import gymnasium as gym
from dqn import TrainingLoop, Logger, set_seed, plot_training_stats, print_statistics
from env_adapter import create_env, print_env_info


def main():
    """Stage 2 訓練主程序"""
    
    # 配置
    CONFIG = {
        'env_name': 'CartPole-v1',
        'seed': 42,
        'num_episodes': 500,
        'batch_size': 64,
        'update_target_freq': 500,  # 更頻繁的更新
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'model_type': 'dueling',  # 決鬥 DQN
        'stage': 2,
    }
    
    # 設置隨機種子
    set_seed(CONFIG['seed'])
    
    print("\n" + "=" * 80)
    print("STAGE 2: Double DQN + Dueling DQN (Player Environment)")
    print("=" * 80)
    print(f"Configuration: {CONFIG}\n")
    
    # 創建環境
    env = create_env(CONFIG['env_name'])
    env_info = print_env_info(env)
    
    # 創建日誌記錄器
    logger = Logger(log_file='stage2_training.log')
    logger.log_info(f"Stage 2 Training Configuration: {CONFIG}")
    logger.log_info(f"Environment: {env_info['name']}")
    logger.log_info(f"State Size: {env_info['state_size']}, Action Size: {env_info['action_size']}\n")
    
    # 創建訓練循環
    print("Creating training loop...")
    training_loop = TrainingLoop.create(
        env=env,
        stage=CONFIG['stage'],  # Stage >= 2 自動使用 DoubleDQNAgent
        model_type=CONFIG['model_type'],
        lr=CONFIG['learning_rate'],
        gamma=CONFIG['gamma'],
        logger=logger
    )
    
    print(f"✓ Training loop created")
    print(f"✓ Agent Type: DoubleDQNAgent (S3 enabled)")
    print(f"✓ Replay Buffer: Standard (Experience Replay)")
    print(f"✓ Model Type: {CONFIG['model_type']} (S4 enabled)\n")
    
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
        save_path='stage2_training_stats.png'
    )
    
    # 清理
    env.close()
    
    print("\n" + "=" * 80)
    print("STAGE 2 COMPLETED")
    print("=" * 80)
    print("\nKey Mechanisms Enabled:")
    print("  ✓ S1: Experience Replay Buffer")
    print("  ✓ S2: Target Network")
    print("  ✓ S3: Double DQN (Decoupled Selection & Evaluation)")
    print("  ✓ S4: Dueling DQN (Separate V and A streams)")
    print("\nGenerated Files:")
    print("  • stage2_training.log")
    print("  • stage2_training_stats.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
