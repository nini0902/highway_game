import sys

import gymnasium as gym
import highway_env  # noqa: F401  # 讓 highway-v0 註冊到 Gymnasium
import pygame


def main() -> None:
    env = gym.make("highway-v0", render_mode="human")
    env.unwrapped.config["duration"] = 60
    observation, info = env.reset()

    pygame.init()
    clock = pygame.time.Clock()

    # highway-v0 的預設離散動作：
    # 0 = 變換到左車道
    # 1 = 保持
    # 2 = 變換到右車道
    # 3 = 加速
    # 4 = 減速
    action = 1

    try:
        running = True
        while running:
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_DOWN]:
                action = 2
            elif keys[pygame.K_UP]:
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 3
            elif keys[pygame.K_LEFT]:
                action = 4
            else:
                action = 1

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

            clock.tick(15)
    finally:
        env.close()
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
