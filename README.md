# 強化學習 DQN 項目

完整的 DQN (Deep Q-Network) 實現，包含 5 個核心機制和 3 個訓練階段。

## 項目特性

### 5 個 DQN 核心機制

| 機制 | 代碼 | 描述 | 引入版本 |
|------|------|------|---------|
| 經驗重放 | S1 | ReplayBuffer 隨機採樣批次 | DQN (2015) |
| 目標網絡 | S2 | 分離的目標網絡評估 Q 值 | DQN (2015) |
| 雙 DQN | S3 | 解耦選擇和評估動作 | Double DQN (2015) |
| 決鬥 DQN | S4 | V 和 A 流分離 | Dueling DQN (2015) |
| 優先重放 | S5 | 基於 TD 誤差的採樣 | PER (2016) |

### 3 個訓練階段

| 階段 | 文件 | 代理 | 機制 | 描述 |
|------|------|------|------|------|
| Stage 1 | run_static.py | DQNAgent | S1, S2 | 基礎 DQN 與經驗重放 |
| Stage 2 | run_player.py | DoubleDQNAgent | S1, S2, S3, S4 | Double DQN + Dueling 網絡 |
| Stage 3 | run_random.py | DoubleDQNAgent | S1, S2, S3, S4, S5 | 完整實現含優先重放 |

## 項目結構

```
H3/
├── dqn/
│   ├── __init__.py          # 模塊初始化
│   ├── models.py             # QNetwork, DuelingQNetwork
│   ├── replay.py             # ReplayBuffer, PrioritizedReplayBuffer
│   ├── agent.py              # DQNAgent, DoubleDQNAgent
│   ├── train.py              # TrainingLoop 類
│   └── utils.py              # Logger, 工具函數
├── env_adapter.py            # Gymnasium 環境適配器
├── run_static.py             # Stage 1 訓練
├── run_player.py             # Stage 2 訓練
├── run_random.py             # Stage 3 訓練
├── test_setup.py             # 完整測試套件
├── README.md                 # 本文件
├── STRUCTURE.md              # 詳細文件結構說明
├── COMPLETION.md             # 項目完成度說明
└── FINAL_SUMMARY.md          # 最終總結
```

## 快速開始

### 前置條件

```bash
pip install tensorflow>=2.0 gymnasium numpy matplotlib
```

### 運行測試

```bash
python test_setup.py
```

### 運行訓練

#### Stage 1: 基礎 DQN
```bash
python run_static.py
```

生成文件：
- `stage1_training.log` - 訓練日誌
- `stage1_training_stats.png` - 訓練統計圖表

#### Stage 2: Double DQN + Dueling DQN
```bash
python run_player.py
```

生成文件：
- `stage2_training.log` - 訓練日誌
- `stage2_training_stats.png` - 訓練統計圖表

#### Stage 3: Double DQN + Dueling + PER
```bash
python run_random.py
```

生成文件：
- `stage3_training.log` - 訓練日誌
- `stage3_training_stats.png` - 訓練統計圖表

## 核心類說明

### DQN 模型

#### QNetwork（標準 DQN）
```python
model = QNetwork(action_size=4, state_size=4)
q_values = model(state)  # 輸出 Q 值
```

特點：
- 簡單的多層感知機架構
- Input -> Dense(128) -> ReLU -> Dense(128) -> ReLU -> Dense(action_size)

#### DuelingQNetwork（決鬥 DQN）
```python
model = DuelingQNetwork(action_size=4, state_size=4)
q_values = model(state)  # Q = V + (A - mean(A))
```

特點：
- 分離的 V（狀態價值）和 A（動作優勢）流
- 自動計算 Q 值：Q(s,a) = V(s) + A(s,a) - mean(A)

### 代理

#### DQNAgent（標準 DQN 代理）
```python
agent = DQNAgent(state_size=4, action_size=4)
action = agent.select_action(state, training=True)  # ε-貪心
loss = agent.update(batch)  # 更新網絡
agent.update_target_network()  # 更新目標網絡
```

機制：
- S1: Experience Replay
- S2: Target Network

#### DoubleDQNAgent（雙 DQN 代理）
```python
agent = DoubleDQNAgent(state_size=4, action_size=4, model_type='dueling')
action = agent.select_action(state, training=True)
loss = agent.update(batch)  # 雙 DQN 更新
```

機制：
- S1: Experience Replay
- S2: Target Network
- S3: Double DQN (選擇和評估解耦)
- S4: Dueling 網絡（可選）

### 回放緩衝區

#### ReplayBuffer（標準回放）
```python
buffer = ReplayBuffer(max_size=100000)
buffer.add(state, action, reward, next_state, done)
batch = buffer.sample(batch_size=32)
```

#### PrioritizedReplayBuffer（優先重放）
```python
buffer = PrioritizedReplayBuffer(max_size=100000, alpha=0.6, beta=0.4)
buffer.add(state, action, reward, next_state, done)
states, actions, rewards, next_states, dones, weights, indices = buffer.sample(32)
buffer.update_priorities(indices, td_errors)  # 更新優先級
```

優點：
- 優先採樣重要的轉移（高 TD 誤差）
- 重要性採樣權重自動計算

### 訓練循環

#### 自動機制選擇
```python
# Stage 1: DQNAgent + ReplayBuffer
training_loop = TrainingLoop.create(env, stage=1)

# Stage 2: DoubleDQNAgent + ReplayBuffer
training_loop = TrainingLoop.create(env, stage=2, model_type='dueling')

# Stage 3: DoubleDQNAgent + PrioritizedReplayBuffer
training_loop = TrainingLoop.create(env, stage=3, model_type='dueling')
```

## 主要實現細節

### Double DQN 核心
```python
# 雙 DQN：選擇和評估解耦
next_actions = tf.argmax(self.q_network(next_states), axis=1)  # 主網絡選擇
next_q_values = tf.gather_nd(self.target_network(next_states),  # 目標網絡評估
                             tf.stack([tf.range(len(next_actions)), next_actions], axis=1))
target_q_values = rewards + gamma * next_q_values * (1 - dones)
```

### Dueling DQN 核心
```python
# 分離流
v = self.v_output(v_dense)      # 狀態價值 (batch, 1)
a = self.a_output(a_dense)      # 動作優勢 (batch, action_size)

# Q 值計算
q_values = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
```

### PER 核心
```python
# 優先級 = |TD error| + epsilon
priorities = (np.abs(td_errors) + epsilon) ** alpha

# 採樣概率
probabilities = priorities / np.sum(priorities)

# 重要性採樣權重
weights = (1 / (N * probabilities)) ** beta
```

## 依賴項

- **TensorFlow 2.0+** - 深度學習框架
- **Gymnasium** - 環境接口
- **NumPy** - 數值計算
- **Matplotlib** - 可視化

## 配置參數

每個訓練腳本都可以通過修改 `CONFIG` 字典自定義：

```python
CONFIG = {
    'env_name': 'CartPole-v1',      # 環境名稱
    'seed': 42,                      # 隨機種子
    'num_episodes': 500,             # 訓練回合數
    'batch_size': 64,                # 批次大小
    'update_target_freq': 1000,      # 目標網絡更新頻率
    'learning_rate': 0.0001,         # 學習率
    'gamma': 0.99,                   # 折扣因子
    'model_type': 'standard',        # 'standard' 或 'dueling'
    'stage': 1,                      # 訓練階段
}
```

## 文件生成

訓練完成後，會生成以下文件：

- `stage{N}_training.log` - 訓練日誌（回合奬勵、目標網絡更新、評估結果）
- `stage{N}_training_stats.png` - 訓練統計圖表（奬勵曲線和損失曲線）

## 故障排除

### 1. 模塊導入失敗
確保 dqn 包在同一目錄中，或者：
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/H3"
```

### 2. 環境創建失敗
安裝 gymnasium：
```bash
pip install gymnasium
```

### 3. TensorFlow 警告
這是正常的警告，不影響運行。可以忽略。

## 項目完成度

✓ 所有 15 個文件已創建
✓ 5 個 DQN 機制已實現
✓ 3 個訓練階段已實現
✓ 自動機制選擇已實現
✓ 完整測試套件已創建
✓ 文檔已完成

詳見 [COMPLETION.md](COMPLETION.md)

## 許可證

MIT License

## 作者

強化學習課程項目
