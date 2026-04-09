# Highway RL 小專案

這個專案使用 `highway-env` 的 `highway-v0` 環境，實作了：

- 人工手動駕駛模式（鍵盤控制）
- 使用 SB3（Stable-Baselines3）訓練自動駕駛 agent
- 自訂 reward wrapper（追車、換道安全、速度與碰撞懲罰）
- 可在訓練與自動播放時顯示 `human` 渲染畫面

## 專案檔案

- `game_auto_mode.py`：自動模式（SB3 DQN 訓練 + agent 自動駕駛）
- `hame_human_mode.py`：手動模式（方向鍵控制車輛）
- `reward_config.py`：獎懲與門檻集中設定（建議優先從這裡調參）
- `game.py`：與自動模式同步的另一份入口

## 目前獎懲核心目標

- 保持與前車的理想距離
- 提升速度但降低碰撞
- 鼓勵安全換道、懲罰危險切入

## 如何執行

在專案目錄執行：

```bash
python game_auto_mode.py
```

- 會使用 SB3 訓練模型（可視化）
- 訓練完成後自動播放 agent

手動模式：

```bash
python hame_human_mode.py
```

## 備註

- 若你只想調整獎懲，直接修改 `reward_config.py` 即可。
- 已訓練模型會存成 `highway_dqn_model.zip`。
## 畫面

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/d660f9ba-1293-4204-8e86-267c37a4ac5b" />
