"""
環境適配器
支持 Gymnasium 和 OpenAI Gym
"""
import gymnasium as gym


def create_env(env_name: str = "CartPole-v1"):
    """創建環境
    
    優先使用 Gymnasium，回退到 Gym（如果已安裝）
    
    Args:
        env_name: 環境名稱
    
    Returns:
        環境實例
    """
    try:
        # 優先嘗試 Gymnasium
        env = gym.make(env_name)
        return env
    except Exception as e:
        print(f"Failed to create environment with Gymnasium: {e}")
        print("Please install gymnasium: pip install gymnasium")
        raise


def get_env_info(env):
    """獲取環境信息
    
    Args:
        env: 環境實例
    
    Returns:
        包含環境信息的字典
    """
    info = {
        'name': env.spec.id if hasattr(env.spec, 'id') else 'Unknown',
        'observation_space': env.observation_space,
        'action_space': env.action_space,
    }
    
    # 獲取狀態大小
    if hasattr(env.observation_space, 'shape'):
        info['state_size'] = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else 1
    else:
        info['state_size'] = env.observation_space.n
    
    # 獲取動作大小
    if hasattr(env.action_space, 'n'):
        info['action_size'] = env.action_space.n
    else:
        info['action_size'] = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 1
    
    return info


def print_env_info(env):
    """打印環境信息
    
    Args:
        env: 環境實例
    """
    info = get_env_info(env)
    
    print("\n" + "=" * 60)
    print("Environment Information")
    print("=" * 60)
    print(f"Environment: {info['name']}")
    print(f"Observation Space: {info['observation_space']}")
    print(f"Action Space: {info['action_space']}")
    print(f"State Size: {info['state_size']}")
    print(f"Action Size: {info['action_size']}")
    print("=" * 60 + "\n")
    
    return info


def test_env(env, num_steps: int = 10):
    """測試環境
    
    Args:
        env: 環境實例
        num_steps: 測試步數
    """
    print("\nTesting environment...")
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    
    for step in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: action={action}, reward={reward:.2f}, done={terminated or truncated}")
        
        if terminated or truncated:
            print("Episode terminated!")
            break
        
        state = next_state
    
    env.close()
    print("Environment test completed!\n")
