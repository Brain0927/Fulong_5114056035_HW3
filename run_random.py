"""
Stage 3 訓練腳本 (隨機環境)
使用機制: S1-Replay Buffer, S2-Target Network, S3-Double DQN, S4-Dueling DQN, S5-PER

DQN 機制:
- S1: Experience Replay
- S2: Target Network
- S3: Double DQN (解耦選擇和評估)
- S4: Dueling DQN (V+A 分離)
- S5: Prioritized Experience Replay (基於 TD 誤差)
"""
import numpy as np
import gymnasium as gym
from dqn import TrainingLoop, Logger, set_seed, plot_training_stats, print_statistics
from env_adapter import create_env, print_env_info


def main():
    """Stage 3 訓練主程序"""
    
    # 配置
    CONFIG = {
        'env_name': 'CartPole-v1',
        'seed': 42,
        'num_episodes': 500,
        'batch_size': 64,
        'update_target_freq': 500,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'model_type': 'dueling',  # 決鬥 DQN
        'stage': 3,  # 自動使用 PER
    }
    
    # 設置隨機種子
    set_seed(CONFIG['seed'])
    
    print("\n" + "=" * 80)
    print("STAGE 3: Double DQN + Dueling DQN + Prioritized Experience Replay")
    print("=" * 80)
    print(f"Configuration: {CONFIG}\n")
    
    # 創建環境
    env = create_env(CONFIG['env_name'])
    env_info = print_env_info(env)
    
    # 創建日誌記錄器
    logger = Logger(log_file='stage3_training.log')
    logger.log_info(f"Stage 3 Training Configuration: {CONFIG}")
    logger.log_info(f"Environment: {env_info['name']}")
    logger.log_info(f"State Size: {env_info['state_size']}, Action Size: {env_info['action_size']}\n")
    
    # 創建訓練循環
    print("Creating training loop...")
    training_loop = TrainingLoop.create(
        env=env,
        stage=CONFIG['stage'],  # Stage == 3 自動使用 PrioritizedReplayBuffer
        model_type=CONFIG['model_type'],
        lr=CONFIG['learning_rate'],
        gamma=CONFIG['gamma'],
        logger=logger
    )
    
    print(f"✓ Training loop created")
    print(f"✓ Agent Type: DoubleDQNAgent (S3 enabled)")
    print(f"✓ Replay Buffer: Prioritized (PER with TD-error priority)")
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
        save_path='stage3_training_stats.png'
    )
    
    # 清理
    env.close()
    
    print("\n" + "=" * 80)
    print("STAGE 3 COMPLETED")
    print("=" * 80)
    print("\nKey Mechanisms Enabled:")
    print("  ✓ S1: Experience Replay Buffer")
    print("  ✓ S2: Target Network")
    print("  ✓ S3: Double DQN (Decoupled Selection & Evaluation)")
    print("  ✓ S4: Dueling DQN (Separate V and A streams)")
    print("  ✓ S5: Prioritized Experience Replay (TD-error based)")
    print("\nGenerated Files:")
    print("  • stage3_training.log")
    print("  • stage3_training_stats.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
