
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
import time


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.probabilities = []

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, -1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        x, y = self._agent_location
        tx, ty = self._target_location

        # Distances to walls (normalized)
        dist_up = y / (self.size - 1)
        dist_down = (self.size - 1 - y) / (self.size - 1)
        dist_left = x / (self.size - 1)
        dist_right = (self.size - 1 - x) / (self.size - 1)

        # Target visibility flags (1 if target is in that direction)
        target_up = (y - ty) / (self.size - 1) if tx == x and ty < y else 0.0
        target_down = (ty - y) / (self.size - 1) if tx == x and ty > y else 0.0
        target_left = (x - tx) / (self.size - 1) if ty == y and tx < x else 0.0
        target_right = (tx - x) / (self.size - 1) if ty == y and tx > x else 0.0
        # dx/dy = normalized vector to the target (always included, even if target isn't visible)
        dx = (tx - x) / (self.size - 1)
        dy = (ty - y) / (self.size - 1)
        return np.array([
            dist_up, dist_down, dist_left, dist_right,
            target_up, target_down, target_left, target_right,
            dx, dy
        ], dtype=np.float32)

        # return np.array([player_pos_normalized, target_pos_normalized, probability, normalized_time], dtype=np.float32)

        # return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # Seed the environment
        super().reset(seed=seed)

        # Randomize agent and target locations
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        self.steps = 0

        # Record the start time
        self.start_time = time.time()

        # Observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Render if required
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _get_distance(self):
        return np.abs(self._agent_location[0] - self._target_location[0]) + \
            np.abs(self._agent_location[1] - self._target_location[1])

    def step(self, action):
        # Apply the action
        direction = self._action_to_direction[action]
        new_location = self._agent_location + direction

        # Check if the new location is out of bounds
        if not (0 <= new_location[0] < self.size) or not (0 <= new_location[1] < self.size):
            observation, info = self.reset()
            reward = -1.0  # Penalty for going out of bounds (scaled down)
            terminated = True
            return observation, reward, terminated, False, info

        # Get distance before moving
        prev_distance = np.abs(self._agent_location[0] - self._target_location[0]) + \
                        np.abs(self._agent_location[1] - self._target_location[1])

        # Move the agent
        self._agent_location = new_location
        self.steps += 1

        # Get distance after moving
        new_distance = np.abs(self._agent_location[0] - self._target_location[0]) + \
                    np.abs(self._agent_location[1] - self._target_location[1])

        # --- Reward Scaling ---
        goal_reward = 1.0
        step_closer = 0.1
        step_farther = -0.1
        no_progress_penalty = -0.2
        alignment_bonus_scale = 0.05
        time_penalty = -0.01

        # --- Initialize reward ---
        reward = time_penalty  # always apply time penalty
        terminated = False

        if np.array_equal(self._agent_location, self._target_location):
            reward += goal_reward
            terminated = True
        else:
            if new_distance < prev_distance:
                reward += step_closer
            elif new_distance > prev_distance:
                reward += step_farther
            else:
                reward += no_progress_penalty

            # --- Directional bonus ---
            dx = self._target_location[0] - self._agent_location[0]
            dy = self._target_location[1] - self._agent_location[1]
            norm = max(np.linalg.norm([dx, dy]), 1e-5)

            goal_vector = np.array([dx, dy]) / norm
            move_vector = direction.astype(np.float32)

            alignment = np.dot(goal_vector, move_vector)
            reward += alignment * alignment_bonus_scale  # gently guide movement direction

        # Optional: Clamp the reward to avoid extreme values
        reward = np.clip(reward, -1.0, 1.0)

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        
    def run(self, control_mode="random"):
        """
        Run the environment.

        Args:
            control_mode (str): "random" for random actions, "manual" for arrow-key control.
        """
        self.render_mode = "human"  # Ensure human rendering mode
        observation, info = self.reset()  # Initialize the environment
        running = True

        while running:
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif control_mode == "manual" and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        action = Actions.right.value
                    elif event.key == pygame.K_UP:
                        action = Actions.up.value
                    elif event.key == pygame.K_LEFT:
                        action = Actions.left.value
                    elif event.key == pygame.K_DOWN:
                        action = Actions.down.value

                # debug:
                # print(self.observation_space)
                # print(self._agent_location)

            # If control_mode is "random", select a random action
            if control_mode == "random" and action is None:
                action = random.choice([0, 1, 2, 3])  # Corresponds to Actions.right, etc.

            if action is not None:
                # Take a step in the environment
                observation, reward, terminated, truncated, info = self.step(action)

                # Check if the episode is done
                if terminated:
                    # print("Target reached! Resetting environment.")
                    observation, info = self.reset()  # Reset if the target is reached

            # Render the current frame
            self.render()

        self.close()  # Clean up after exiting the loop


# Instantiate and run the environment
if __name__ == "__main__":
    env = GridWorldEnv()
    env.run(control_mode="manual")