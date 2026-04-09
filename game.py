from pathlib import Path
import sys
import time

import gymnasium as gym
import highway_env  # noqa: F401  # 讓 highway-v0 註冊到 Gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from reward_config import REWARD_CONFIG


MODEL_PATH = Path(__file__).with_name("highway_dqn_model")
TRAIN_TIMESTEPS = 50_000
RETRAIN_EVERY_RUN = True
TRAIN_RENDER_MODE = "human"
TRAIN_RENDER_EVERY_STEPS = 1
PLAY_EPISODES = 10
PLAY_FPS = 15
POST_PLAY_HOLD_SECONDS = 5


class ChaseLaneRewardWrapper(gym.Wrapper):
	"""Reward wrapper for chasing the front vehicle and switching lanes wisely."""

	def __init__(
		self,
		env: gym.Env,
		ideal_distance: float = REWARD_CONFIG["distance"]["ideal"],
		distance_tolerance: float = REWARD_CONFIG["distance"]["tolerance"],
		far_distance: float = REWARD_CONFIG["distance"]["far_threshold"],
		close_distance: float = REWARD_CONFIG["distance"]["close_threshold"],
		speed_threshold: float = REWARD_CONFIG["speed"]["threshold"],
		lane_change_trigger: float = REWARD_CONFIG["lane_change"]["trigger_distance"],
		lane_change_gain: float = REWARD_CONFIG["lane_change"]["gain_threshold"],
		side_block_distance: float = REWARD_CONFIG["lane_safety"]["side_block_distance"],
	) -> None:
		super().__init__(env)
		self.ideal_distance = ideal_distance
		self.distance_tolerance = distance_tolerance
		self.far_distance = far_distance
		self.close_distance = close_distance
		self.speed_threshold = speed_threshold
		self.lane_change_trigger = lane_change_trigger
		self.lane_change_gain = lane_change_gain
		self.side_block_distance = side_block_distance
		self.collision_penalty = REWARD_CONFIG["collision"]["penalty"]
		self.distance_missing_reward = REWARD_CONFIG["distance"]["missing_reward"]
		self.distance_ideal_reward = REWARD_CONFIG["distance"]["ideal_reward"]
		self.distance_far_penalty = REWARD_CONFIG["distance"]["far_penalty"]
		self.distance_close_penalty = REWARD_CONFIG["distance"]["close_penalty"]
		self.speed_reward_value = REWARD_CONFIG["speed"]["reward"]
		self.lane_change_success_reward = REWARD_CONFIG["lane_change"]["success_reward"]
		self.lane_change_failure_penalty = REWARD_CONFIG["lane_change"]["failure_penalty"]
		self.lane_safety_safe_reward = REWARD_CONFIG["lane_safety"]["safe_change_reward"]
		self.lane_safety_left_penalty = REWARD_CONFIG["lane_safety"]["blocked_left_penalty"]
		self.lane_safety_right_penalty = REWARD_CONFIG["lane_safety"]["blocked_right_penalty"]

		self._prev_lane_index = None
		self._prev_front_distance = None

	def reset(self, **kwargs):
		observation, info = self.env.reset(**kwargs)
		ego = self._ego_vehicle()
		self._prev_lane_index = ego.lane_index if ego is not None else None
		self._prev_front_distance = self._front_distance(self._prev_lane_index)
		return observation, info

	def step(self, action):
		action = int(action)
		prev_lane_index = self._prev_lane_index
		prev_front_distance = self._prev_front_distance
		left_occupied_before_change = self._is_target_lane_occupied(prev_lane_index, 0)
		right_occupied_before_change = self._is_target_lane_occupied(prev_lane_index, 2)

		observation, base_reward, terminated, truncated, info = self.env.step(action)

		ego = self._ego_vehicle()
		current_lane_index = ego.lane_index if ego is not None else None
		current_front_distance = self._front_distance(current_lane_index)
		speed = float(info.get("speed", getattr(ego, "speed", 0.0)))
		crashed = bool(info.get("crashed", False)) or bool(getattr(ego, "crashed", False))

		reward = 0.0
		reward_terms: dict[str, float] = {}

		if crashed:
			reward += self.collision_penalty
			reward_terms["collision"] = self.collision_penalty

		distance_reward = self._distance_reward(current_front_distance)
		reward += distance_reward
		reward_terms["distance"] = distance_reward

		speed_reward = self.speed_reward_value if speed >= self.speed_threshold else 0.0
		reward += speed_reward
		reward_terms["speed"] = speed_reward

		lane_change_reward = self._lane_change_reward(
			action=action,
			prev_lane_index=prev_lane_index,
			current_lane_index=current_lane_index,
			prev_front_distance=prev_front_distance,
			current_front_distance=current_front_distance,
		)
		if lane_change_reward != 0.0:
			reward += lane_change_reward
			reward_terms["lane_change"] = lane_change_reward

		lane_safety_reward = self._lane_safety_reward(
			action=action,
			prev_lane_index=prev_lane_index,
			current_lane_index=current_lane_index,
			left_occupied=left_occupied_before_change,
			right_occupied=right_occupied_before_change,
		)
		if lane_safety_reward != 0.0:
			reward += lane_safety_reward
			reward_terms["lane_safety"] = lane_safety_reward

		info["base_reward"] = base_reward
		info["reward_terms"] = reward_terms
		info["shaped_reward"] = reward
		info["lane_index"] = current_lane_index
		info["front_distance"] = current_front_distance
		info["prev_front_distance"] = prev_front_distance
		info["lane_changed"] = prev_lane_index != current_lane_index
		info["left_occupied_before_change"] = left_occupied_before_change
		info["right_occupied_before_change"] = right_occupied_before_change

		self._prev_lane_index = current_lane_index
		self._prev_front_distance = current_front_distance

		return observation, reward, terminated, truncated, info

	def _ego_vehicle(self):
		return getattr(self.env.unwrapped, "vehicle", None)

	def _front_vehicle(self, lane_index):
		ego = self._ego_vehicle()
		road = getattr(self.env.unwrapped, "road", None)
		if ego is None or road is None or lane_index is None:
			return None
		front_vehicle, _ = road.neighbour_vehicles(ego, lane_index)
		return front_vehicle

	def _front_distance(self, lane_index):
		ego = self._ego_vehicle()
		front_vehicle = self._front_vehicle(lane_index)
		if ego is None or front_vehicle is None:
			return None
		return max(0.0, float(ego.front_distance_to(front_vehicle)))

	def _distance_reward(self, front_distance):
		if front_distance is None:
			return self.distance_missing_reward
		if abs(front_distance - self.ideal_distance) <= self.distance_tolerance:
			return self.distance_ideal_reward
		if front_distance > self.far_distance:
			return self.distance_far_penalty
		if front_distance < self.close_distance:
			return self.distance_close_penalty
		return 0.0

	def _lane_change_reward(
		self,
		action,
		prev_lane_index,
		current_lane_index,
		prev_front_distance,
		current_front_distance,
	):
		lane_change_actions = {0, 2}
		if action not in lane_change_actions:
			return 0.0

		if prev_lane_index is None or current_lane_index is None:
			return self.lane_change_failure_penalty

		if prev_lane_index == current_lane_index:
			return self.lane_change_failure_penalty

		if prev_front_distance is None or current_front_distance is None:
			return self.lane_change_failure_penalty

		if (
			prev_front_distance <= self.lane_change_trigger
			and current_front_distance >= prev_front_distance + self.lane_change_gain
		):
			return self.lane_change_success_reward

		return self.lane_change_failure_penalty

	def _target_lane_index(self, lane_index, action):
		if lane_index is None:
			return None
		road = getattr(self.env.unwrapped, "road", None)
		if road is None:
			return None
		side_lanes = road.network.side_lanes(lane_index)
		if action == 0:
			for candidate in side_lanes:
				if candidate[2] < lane_index[2]:
					return candidate
		elif action == 2:
			for candidate in side_lanes:
				if candidate[2] > lane_index[2]:
					return candidate
		return None

	def _is_lane_occupied(self, lane_index):
		ego = self._ego_vehicle()
		road = getattr(self.env.unwrapped, "road", None)
		if ego is None or road is None or lane_index is None:
			return False
		front_vehicle, rear_vehicle = road.neighbour_vehicles(ego, lane_index)
		if front_vehicle is not None:
			if abs(float(ego.front_distance_to(front_vehicle))) <= self.side_block_distance:
				return True
		if rear_vehicle is not None:
			if abs(float(ego.front_distance_to(rear_vehicle))) <= self.side_block_distance:
				return True
		return False

	def _is_target_lane_occupied(self, lane_index, action):
		if action not in {0, 2}:
			return False
		target_lane_index = self._target_lane_index(lane_index, action)
		return self._is_lane_occupied(target_lane_index)

	def _lane_safety_reward(
		self,
		action,
		prev_lane_index,
		current_lane_index,
		left_occupied,
		right_occupied,
	):
		if action == 0 and left_occupied:
			return self.lane_safety_left_penalty
		if action == 2 and right_occupied:
			return self.lane_safety_right_penalty
		if action in {0, 2} and prev_lane_index is not None and current_lane_index is not None:
			if prev_lane_index != current_lane_index:
				return self.lane_safety_safe_reward
		return 0.0


class TrainingRenderCallback(BaseCallback):
	def __init__(self, render_every_steps: int = 1, verbose: int = 0):
		super().__init__(verbose)
		self.render_every_steps = max(1, int(render_every_steps))

	def _on_step(self) -> bool:
		if self.n_calls % self.render_every_steps == 0:
			self.training_env.render()
		return True


def make_env(render_mode=None):
	env = gym.make("highway-v0", render_mode=render_mode)
	env.unwrapped.config["duration"] = REWARD_CONFIG["env"]["duration"]
	if render_mode == "human":
		env.unwrapped.config["offscreen_rendering"] = False
		env.unwrapped.config["real_time_rendering"] = True
	return Monitor(ChaseLaneRewardWrapper(env))


def train_agent(total_timesteps: int = TRAIN_TIMESTEPS, render_mode=None) -> DQN:
	env = make_env(render_mode=render_mode)
	try:
		model = DQN(
			"MlpPolicy",
			env,
			learning_rate=1e-4,
			buffer_size=50_000,
			learning_starts=1_000,
			batch_size=64,
			gamma=0.99,
			target_update_interval=1_000,
			train_freq=4,
			gradient_steps=1,
			exploration_fraction=0.3,
			exploration_final_eps=0.05,
			verbose=1,
		)
		callback = None
		if render_mode == "human":
			callback = TrainingRenderCallback(render_every_steps=TRAIN_RENDER_EVERY_STEPS)
		model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
		model.save(str(MODEL_PATH))
		return model
	finally:
		env.close()


def load_or_train_model(total_timesteps: int = TRAIN_TIMESTEPS) -> DQN:
	if MODEL_PATH.with_suffix(".zip").exists():
		return DQN.load(str(MODEL_PATH))
	return train_agent(total_timesteps=total_timesteps)


def play_agent(model: DQN, episodes: int = PLAY_EPISODES) -> None:
	env = make_env(render_mode="human")
	try:
		for _ in range(episodes):
			observation, info = env.reset()
			env.unwrapped.render()
			done = False
			while not done:
				action, _ = model.predict(observation, deterministic=True)
				observation, reward, terminated, truncated, info = env.step(action)
				env.unwrapped.render()
				time.sleep(1 / PLAY_FPS)
				done = terminated or truncated

		hold_until = time.time() + POST_PLAY_HOLD_SECONDS
		while time.time() < hold_until:
			env.unwrapped.render()
			time.sleep(1 / PLAY_FPS)
	finally:
		env.close()


def main() -> None:
	if RETRAIN_EVERY_RUN:
		model = train_agent(total_timesteps=TRAIN_TIMESTEPS, render_mode=TRAIN_RENDER_MODE)
	else:
		model = load_or_train_model(total_timesteps=TRAIN_TIMESTEPS)
	play_agent(model, episodes=PLAY_EPISODES)
	return


if __name__ == "__main__":
	main()
