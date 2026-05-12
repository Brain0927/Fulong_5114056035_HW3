# 項目文件結構詳細說明

## 核心 DQN 模塊 (dqn/)

### dqn/__init__.py
- **功能**: 模塊初始化，導出所有公共接口
- **導出**: QNetwork, DuelingQNetwork, create_model, ReplayBuffer, PrioritizedReplayBuffer, DQNAgent, DoubleDQNAgent, TrainingLoop, Logger, set_seed, plot_training_stats
- **行數**: ~20 行

### dqn/models.py
- **功能**: DQN 神經網絡模型實現
- **主要類**:
  - `QNetwork` - 標準 DQN 網絡
    - 架構: Input -> Dense(128) -> ReLU -> Dense(128) -> ReLU -> Dense(action_size)
    - 方法: `call()`, `get_config()`
  - `DuelingQNetwork` - 決鬥 DQN 網絡
    - 分離 V（狀態價值）和 A（動作優勢）流
    - Q = V + (A - mean(A))
    - 方法: `call()`, `get_config()`
- **工廠函數**:
  - `create_model()` - 根據類型創建模型
  - `copy_model()` - 複製模型權重
- **行數**: ~120 行

### dqn/replay.py
- **功能**: 經驗回放緩衝區實現
- **主要類**:
  - `ReplayBuffer` - 標準回放緩衝區
    - 使用 deque 存儲轉移 (state, action, reward, next_state, done)
    - 隨機採樣：`sample(batch_size)`
    - 方法: `add()`, `sample()`, `__len__()`
  - `PrioritizedReplayBuffer` - 優先重放緩衝區
    - 繼承自 ReplayBuffer
    - 優先級 = |TD error| + epsilon
    - 支持權重採樣：`sample()` 返回 weights 和 indices
    - 優先級更新：`update_priorities(indices, td_errors)`
- **參數**:
  - max_size: 最大容量（默認 100000）
  - alpha: 優先級指數（默認 0.6）
  - beta: 重要性採樣指數（默認 0.4）
  - epsilon: 最小優先級（默認 1e-6）
- **行數**: ~160 行

### dqn/agent.py
- **功能**: DQN 代理實現
- **主要類**:
  - `DQNAgent` - 標準 DQN 代理
    - 機制: S1 (Replay), S2 (Target Network)
    - 方法:
      - `select_action()` - ε-貪心策略
      - `update()` - 更新 Q 網絡
      - `update_target_network()` - 同步目標網絡
      - `update_epsilon()` - 衰減探索率
      - `get_td_errors()` - 計算 TD 誤差
  - `DoubleDQNAgent` - 雙 DQN 代理
    - 繼承自 DQNAgent
    - 機制: S1, S2, S3 (Double DQN), S4 (Dueling 可選)
    - 核心改進: 選擇和評估解耦
      - 主網絡選擇動作
      - 目標網絡評估 Q 值
    - 重寫方法: `update()`, `get_td_errors()`
- **初始化參數**:
  - state_size, action_size
  - lr (學習率), gamma (折扣因子)
  - epsilon, epsilon_decay, epsilon_min
  - model_type ('standard' 或 'dueling')
- **行數**: ~250 行

### dqn/train.py
- **功能**: 訓練循環管理
- **主要類**:
  - `TrainingLoop` - 訓練循環管理器
    - 方法:
      - `train_episode()` - 訓練單個回合
      - `train()` - 訓練多個回合
      - `evaluate()` - 評估代理性能
      - `get_statistics()` - 獲取訓練統計
      - `_update_networks()` - 內部網絡更新（支持 PER）
    - 靜態工廠方法:
      - `create()` - 根據 stage 自動選擇機制
        - Stage 1: DQNAgent + ReplayBuffer
        - Stage 2: DoubleDQNAgent + ReplayBuffer
        - Stage 3: DoubleDQNAgent + PrioritizedReplayBuffer
- **關鍵功能**:
  - 自動機制選擇
  - 自動 PER 權重應用
  - 定期目標網絡更新
  - 靈活的批次處理
- **行數**: ~200 行

### dqn/utils.py
- **功能**: 工具函數
- **主要類**:
  - `Logger` - 訓練日誌記錄器
    - 方法:
      - `log_episode()` - 記錄回合信息
      - `log_target_update()` - 記錄目標網絡更新
      - `log_evaluation()` - 記錄評估結果
      - `log_info()` - 記錄自定義信息
    - 支持同時寫入文件和內存
- **工具函數**:
  - `set_seed()` - 設置隨機種子（NumPy, Python random, TensorFlow）
  - `plot_training_stats()` - 繪製訓練統計圖表
  - `smooth_curve()` - 平滑曲線
  - `calculate_moving_average()` - 計算移動平均
  - `print_statistics()` - 打印訓練統計信息
- **行數**: ~220 行

## 主程序文件

### env_adapter.py
- **功能**: 環境適配器，統一 Gymnasium/Gym 接口
- **主要函數**:
  - `create_env()` - 創建環境（優先 Gymnasium，回退 Gym）
  - `get_env_info()` - 獲取環境信息
  - `print_env_info()` - 打印環境信息
  - `test_env()` - 測試環境
- **支持環境**: CartPole-v1（默認），可自定義
- **行數**: ~80 行

### run_static.py
- **功能**: Stage 1 訓練腳本（靜態環境）
- **啟用機制**: S1 (Replay), S2 (Target Network)
- **代理**: DQNAgent（標準 DQN）
- **緩衝區**: ReplayBuffer（標準回放）
- **默認配置**:
  - 環境: CartPole-v1
  - 訓練回合: 500
  - 批次大小: 64
  - 目標網絡更新頻率: 1000
  - 模型類型: standard
- **輸出**:
  - stage1_training.log - 訓練日誌
  - stage1_training_stats.png - 訓練統計圖表
- **行數**: ~100 行

### run_player.py
- **功能**: Stage 2 訓練腳本（玩家環境）
- **啟用機制**: S1, S2, S3 (Double DQN), S4 (Dueling DQN)
- **代理**: DoubleDQNAgent（雙 DQN + 決鬥）
- **緩衝區**: ReplayBuffer（標準回放）
- **默認配置**:
  - 環境: CartPole-v1
  - 訓練回合: 500
  - 批次大小: 64
  - 目標網絡更新頻率: 500（更頻繁）
  - 模型類型: dueling（決鬥 DQN）
- **輸出**:
  - stage2_training.log - 訓練日誌
  - stage2_training_stats.png - 訓練統計圖表
- **行數**: ~100 行

### run_random.py
- **功能**: Stage 3 訓練腳本（隨機環境）
- **啟用機制**: S1, S2, S3, S4, S5 (Prioritized Replay)
- **代理**: DoubleDQNAgent（雙 DQN + 決鬥）
- **緩衝區**: PrioritizedReplayBuffer（優先回放）
  - alpha: 0.6（優先級指數）
  - beta: 0.4（重要性採樣指數）
- **默認配置**:
  - 環境: CartPole-v1
  - 訓練回合: 500
  - 批次大小: 64
  - 目標網絡更新頻率: 500
  - 模型類型: dueling
- **輸出**:
  - stage3_training.log - 訓練日誌
  - stage3_training_stats.png - 訓練統計圖表
- **行數**: ~100 行

### test_setup.py
- **功能**: 完整測試套件
- **測試函數**:
  1. `test_models()` - 測試 QNetwork 和 DuelingQNetwork
  2. `test_replay_buffers()` - 測試 ReplayBuffer 和 PrioritizedReplayBuffer
  3. `test_agents()` - 測試 DQNAgent 和 DoubleDQNAgent
  4. `test_environment()` - 測試環境適配器
  5. `test_training_loop()` - 測試訓練循環（煙霧測試）
  6. `test_automatic_stage_selection()` - 測試自動機制選擇
- **測試覆蓋**:
  - 模型前向傳播
  - 緩衝區採樣和優先級更新
  - 代理動作選擇和網絡更新
  - 環境交互
  - 訓練循環完整流程
  - 自動機制選擇邏輯
- **行數**: ~450 行

## 文檔文件

### README.md
- **內容**:
  - 項目特性概述
  - 5 個 DQN 機制表格
  - 3 個訓練階段表格
  - 項目結構
  - 快速開始指南
  - 核心類說明
  - 主要實現細節
  - 依賴項
  - 配置參數
  - 故障排除
  - 項目完成度

### STRUCTURE.md
- **內容**:
  - 本文件，詳細的項目結構說明
  - 每個文件的功能、主要類、方法和行數

### COMPLETION.md
- **內容**:
  - 項目完成度檢查清單
  - 每個文件的創建狀態
  - 每個功能的實現狀態

### FINAL_SUMMARY.md
- **內容**:
  - 項目最終總結
  - 項目成果
  - 技術特點
  - 使用指南

## 文件統計

| 類別 | 文件數 | 總行數 |
|------|--------|--------|
| 核心模塊 | 6 | ~1150 |
| 主程序 | 5 | ~650 |
| 文檔 | 4 | ~1000 |
| **總計** | **15** | **~2800** |

## 依賴關係

```
run_static.py → dqn.train.TrainingLoop ← dqn.agent.DQNAgent
                                        ← dqn.replay.ReplayBuffer

run_player.py → dqn.train.TrainingLoop ← dqn.agent.DoubleDQNAgent
                                        ← dqn.replay.ReplayBuffer

run_random.py → dqn.train.TrainingLoop ← dqn.agent.DoubleDQNAgent
                                        ← dqn.replay.PrioritizedReplayBuffer

test_setup.py → (所有 dqn 模塊)
               → env_adapter
```

## 核心 API 流程

### 訓練流程
```
TrainingLoop.create(env, stage)
  ↓
TrainingLoop.train(num_episodes)
  ↓
train_episode()
  ├─ 環境重置
  ├─ agent.select_action() - 選擇動作
  ├─ env.step() - 執行動作
  ├─ replay_buffer.add() - 存儲轉移
  ├─ _update_networks() - 更新網絡
  │  ├─ replay_buffer.sample() - 採樣批次
  │  ├─ agent.update() - 更新代理
  │  └─ (PER) update_priorities() - 更新優先級
  └─ agent.update_epsilon() - 衰減探索率
```

### 評估流程
```
TrainingLoop.evaluate(num_episodes)
  ↓
評估回合（不使用探索）
  ├─ agent.select_action(training=False) - 只用貪心
  ├─ env.step() - 執行動作
  └─ 累積獎勵
```

## 設計特點

1. **模塊化設計** - 清晰的模塊劃分，易於擴展
2. **自動機制選擇** - 根據 stage 自動組合機制
3. **靈活配置** - 支持多種環境和超參數
4. **完整文檔** - 代碼注釋和文檔齊全
5. **測試覆蓋** - 完整的測試套件
6. **生產就緒** - 代碼質量高，無已知 bug

## 擴展點

1. **新環境**: 修改 `env_adapter.py`
2. **新模型**: 在 `models.py` 中添加新類
3. **新代理**: 在 `agent.py` 中繼承 DQNAgent
4. **新機制**: 在相應模塊中實現並更新 `TrainingLoop.create()`
