REWARD_CONFIG = {
    "distance": {
        "ideal": 30.0,
        "tolerance": 5.0,
        "far_threshold": 45.0,
        "close_threshold": 30.0,
        "ideal_reward": 0.8,
        "far_penalty": -0.7,
        "close_penalty": -0.7,
        "missing_reward": 0.2,
    },
    "speed": {
        "threshold": 28.0,
        "reward": 0.1,
    },
    "collision": {
        "penalty": -1.0,
    },
    "lane_change": {
        "trigger_distance": 30.0,
        "gain_threshold": 8.0,
        "success_reward": 0.5,
        "failure_penalty": -0.2,
    },
    "lane_safety": {
        "side_block_distance": 12.0,
        "safe_change_reward": 0.6,
        "blocked_left_penalty": -0.9,
        "blocked_right_penalty": -0.9,
    },
    "env": {
        "duration": 60,
    },
}