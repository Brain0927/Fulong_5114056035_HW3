"""
完整的測試套件
測試：環境、模型、代理、回放緩衝區和訓練循環
"""
import sys
import numpy as np
import gymnasium as gym
from dqn import (
    QNetwork, DuelingQNetwork, create_model,
    ReplayBuffer, PrioritizedReplayBuffer,
    DQNAgent, DoubleDQNAgent,
    TrainingLoop, Logger, set_seed
)
from env_adapter import create_env, print_env_info, test_env


def test_models():
    """測試模型"""
    print("\n" + "=" * 80)
    print("TEST 1: Models")
    print("=" * 80)
    
    try:
        # 測試標準 DQN
        print("\n[Test 1.1] QNetwork (Standard DQN)")
        q_net = QNetwork(action_size=4, state_size=4)
        state = np.random.randn(1, 4).astype(np.float32)
        q_values = q_net(state)
        assert q_values.shape == (1, 4), f"Expected shape (1, 4), got {q_values.shape}"
        print(f"✓ QNetwork output shape: {q_values.shape}")
        
        # 測試決鬥 DQN
        print("\n[Test 1.2] DuelingQNetwork (Dueling DQN)")
        dueling_net = DuelingQNetwork(action_size=4, state_size=4)
        q_values = dueling_net(state)
        assert q_values.shape == (1, 4), f"Expected shape (1, 4), got {q_values.shape}"
        print(f"✓ DuelingQNetwork output shape: {q_values.shape}")
        
        # 測試工廠函數
        print("\n[Test 1.3] create_model factory function")
        model1 = create_model(4, 4, "standard")
        model2 = create_model(4, 4, "dueling")
        print(f"✓ Created standard model: {type(model1).__name__}")
        print(f"✓ Created dueling model: {type(model2).__name__}")
        
        print("\n✓ All model tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Model test failed: {e}")
        return False


def test_replay_buffers():
    """測試回放緩衝區"""
    print("\n" + "=" * 80)
    print("TEST 2: Replay Buffers")
    print("=" * 80)
    
    try:
        # 測試標準回放緩衝區
        print("\n[Test 2.1] Standard Replay Buffer")
        buffer = ReplayBuffer(max_size=100)
        
        # 添加數據
        for i in range(50):
            state = np.random.randn(4)
            action = 0
            reward = 1.0
            next_state = np.random.randn(4)
            done = False
            buffer.add(state, action, reward, next_state, done)
        
        print(f"✓ Added 50 transitions. Buffer size: {len(buffer)}")
        
        # 採樣
        batch = buffer.sample(32)
        assert len(batch) == 5, f"Expected 5 elements in batch, got {len(batch)}"
        states, actions, rewards, next_states, dones = batch
        print(f"✓ Sampled batch shapes: {states.shape}, {actions.shape}, {rewards.shape}")
        
        # 測試優先回放緩衝區
        print("\n[Test 2.2] Prioritized Replay Buffer")
        per_buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6, beta=0.4)
        
        # 添加數據
        for i in range(50):
            state = np.random.randn(4)
            action = 0
            reward = 1.0
            next_state = np.random.randn(4)
            done = False
            per_buffer.add(state, action, reward, next_state, done)
        
        print(f"✓ Added 50 transitions. Buffer size: {len(per_buffer)}")
        
        # 採樣（包含權重）
        batch = per_buffer.sample(32)
        assert len(batch) == 7, f"Expected 7 elements in batch, got {len(batch)}"
        states, actions, rewards, next_states, dones, weights, indices = batch
        print(f"✓ Sampled batch with weights. Shape: {states.shape}, Weights: {weights.shape}")
        
        # 更新優先級
        td_errors = np.random.randn(32)
        per_buffer.update_priorities(indices, td_errors)
        print(f"✓ Updated priorities for {len(indices)} transitions")
        
        print("\n✓ All replay buffer tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Replay buffer test failed: {e}")
        return False


def test_agents():
    """測試代理"""
    print("\n" + "=" * 80)
    print("TEST 3: Agents")
    print("=" * 80)
    
    try:
        # 測試 DQN 代理
        print("\n[Test 3.1] DQNAgent")
        agent = DQNAgent(state_size=4, action_size=4)
        state = np.random.randn(4).astype(np.float32)
        
        # 選擇動作
        action = agent.select_action(state, training=True)
        assert 0 <= action < 4, f"Invalid action: {action}"
        print(f"✓ Selected action: {action}")
        
        # 更新代理
        batch = (
            np.random.randn(32, 4).astype(np.float32),
            np.random.randint(0, 4, 32),
            np.random.randn(32).astype(np.float32),
            np.random.randn(32, 4).astype(np.float32),
            np.random.randint(0, 2, 32).astype(np.float32)
        )
        loss = agent.update(batch)
        print(f"✓ Updated agent. Loss: {loss:.6f}")
        
        # 更新目標網絡
        agent.update_target_network()
        print(f"✓ Updated target network")
        
        # 測試雙 DQN 代理
        print("\n[Test 3.2] DoubleDQNAgent")
        double_agent = DoubleDQNAgent(state_size=4, action_size=4, model_type='dueling')
        
        # 選擇動作
        action = double_agent.select_action(state, training=True)
        assert 0 <= action < 4, f"Invalid action: {action}"
        print(f"✓ Selected action: {action}")
        
        # 更新代理（雙 DQN）
        loss = double_agent.update(batch)
        print(f"✓ Updated double agent. Loss: {loss:.6f}")
        
        # 獲取 TD 誤差
        td_errors = double_agent.get_td_errors(batch)
        print(f"✓ Calculated TD errors. Shape: {td_errors.shape}")
        
        print("\n✓ All agent tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """測試環境"""
    print("\n" + "=" * 80)
    print("TEST 4: Environment")
    print("=" * 80)
    
    try:
        print("\n[Test 4.1] Creating CartPole environment")
        env = create_env('CartPole-v1')
        
        print("\n[Test 4.2] Environment information")
        info = print_env_info(env)
        
        print("\n[Test 4.3] Running test steps")
        state, _ = env.reset()
        print(f"✓ Initial state shape: {state.shape}")
        
        for _ in range(10):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        print(f"✓ Environment test completed")
        
        print("\n✓ All environment tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Environment test failed: {e}")
        return False


def test_training_loop():
    """測試訓練循環"""
    print("\n" + "=" * 80)
    print("TEST 5: Training Loop (Quick Smoke Test)")
    print("=" * 80)
    
    try:
        print("\n[Test 5.1] Creating environment")
        env = create_env('CartPole-v1')
        
        print("\n[Test 5.2] Creating Stage 1 training loop")
        logger = Logger()
        training_loop = TrainingLoop.create(env, stage=1, model_type='standard', logger=logger)
        print(f"✓ Created training loop with DQNAgent")
        
        print("\n[Test 5.3] Training for 10 episodes (smoke test)")
        training_loop.train(num_episodes=10, update_target_freq=5, batch_size=32)
        print(f"✓ Training completed. Total episodes: {len(training_loop.episode_rewards)}")
        
        print("\n[Test 5.4] Evaluating agent")
        avg_reward = training_loop.evaluate(num_episodes=5)
        print(f"✓ Evaluation completed. Average reward: {avg_reward:.2f}")
        
        print("\n[Test 5.5] Getting statistics")
        stats = training_loop.get_statistics()
        print(f"✓ Statistics: Episodes={stats['episodes']}, Avg Reward={stats['avg_reward']:.2f}")
        
        env.close()
        
        print("\n✓ All training loop tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_automatic_stage_selection():
    """測試自動階段選擇"""
    print("\n" + "=" * 80)
    print("TEST 6: Automatic Stage Selection")
    print("=" * 80)
    
    try:
        env = create_env('CartPole-v1')
        
        print("\n[Test 6.1] Stage 1 (Standard DQN + Replay)")
        loop1 = TrainingLoop.create(env, stage=1)
        assert hasattr(loop1.agent, 'q_network'), "Stage 1 should have q_network"
        assert isinstance(loop1.replay_buffer, ReplayBuffer), "Stage 1 should use ReplayBuffer"
        assert not isinstance(loop1.replay_buffer, PrioritizedReplayBuffer), "Stage 1 should not use PER"
        print(f"✓ Agent: {type(loop1.agent).__name__}")
        print(f"✓ Buffer: {type(loop1.replay_buffer).__name__}")
        
        print("\n[Test 6.2] Stage 2 (Double DQN + Replay)")
        loop2 = TrainingLoop.create(env, stage=2)
        assert isinstance(loop2.agent, DoubleDQNAgent), "Stage 2 should use DoubleDQNAgent"
        assert isinstance(loop2.replay_buffer, ReplayBuffer), "Stage 2 should use ReplayBuffer"
        assert not isinstance(loop2.replay_buffer, PrioritizedReplayBuffer), "Stage 2 should not use PER"
        print(f"✓ Agent: {type(loop2.agent).__name__}")
        print(f"✓ Buffer: {type(loop2.replay_buffer).__name__}")
        
        print("\n[Test 6.3] Stage 3 (Double DQN + PER)")
        loop3 = TrainingLoop.create(env, stage=3)
        assert isinstance(loop3.agent, DoubleDQNAgent), "Stage 3 should use DoubleDQNAgent"
        assert isinstance(loop3.replay_buffer, PrioritizedReplayBuffer), "Stage 3 should use PrioritizedReplayBuffer"
        print(f"✓ Agent: {type(loop3.agent).__name__}")
        print(f"✓ Buffer: {type(loop3.replay_buffer).__name__}")
        
        env.close()
        
        print("\n✓ All automatic selection tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Automatic selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """運行所有測試"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  COMPLETE DQN PROJECT TEST SUITE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 設置隨機種子
    set_seed(42)
    
    results = []
    
    # 運行所有測試
    results.append(("Models", test_models()))
    results.append(("Replay Buffers", test_replay_buffers()))
    results.append(("Agents", test_agents()))
    results.append(("Environment", test_environment()))
    results.append(("Training Loop", test_training_loop()))
    results.append(("Automatic Stage Selection", test_automatic_stage_selection()))
    
    # 總結
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("\n" + "-" * 80)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓".center(80))
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("SOME TESTS FAILED ✗".center(80))
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
