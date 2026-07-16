from enum import Enum
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
        module=r"pygame\.pkgdata",
    )
    import pygame


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, obstacles=None, obstacle_penalty=-1.0):
        self.size = size  # The size of the square grid
        self.obstacles = obstacles
        self.obstacle_penalty = float(obstacle_penalty)
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
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
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
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
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "success": False,
        }

    def _is_obstacle_cell(self, location):
        return self.obstacles is not None and tuple(location) in self.obstacles

    def _sample_free_location(self, excluded=()):
        """Sample a cell that is not an obstacle or in `excluded`."""

        excluded_cells = {tuple(np.asarray(location, dtype=int)) for location in excluded}
        free_cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if not self._is_obstacle_cell((x, y)) and (x, y) not in excluded_cells
        ]
        if not free_cells:
            raise ValueError("No free cells available for sampling.")

        index = self.np_random.integers(0, len(free_cells))
        return np.array(free_cells[index], dtype=int)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if options:
            agent_location = options.get("agent_location")
            target_location = options.get("target_location")

            if agent_location is not None:
                self._agent_location = np.asarray(agent_location, dtype=int)
                if self._is_obstacle_cell(self._agent_location):
                    raise ValueError("Agent location cannot be on an obstacle.")

            if target_location is not None:
                self._target_location = np.asarray(target_location, dtype=int)
                if self._is_obstacle_cell(self._target_location):
                    raise ValueError("Target location cannot be on an obstacle.")

            if agent_location is None:
                excluded = (target_location,) if target_location is not None else ()
                self._agent_location = self._sample_free_location(excluded=excluded)

            if target_location is None:
                self._target_location = self._sample_free_location(
                    excluded=(self._agent_location,)
                )
            elif np.array_equal(self._target_location, self._agent_location):
                raise ValueError("Agent and target locations must differ.")
        else:
            self._agent_location = self._sample_free_location()
            self._target_location = self._sample_free_location(
                excluded=(self._agent_location,)
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        on_obstacle = self._is_obstacle_cell(self._agent_location)
        on_target = np.array_equal(self._agent_location, self._target_location)
        terminated = on_obstacle or on_target
        if on_obstacle:
            reward = self.obstacle_penalty
        elif on_target:
            reward = 1.0
        else:
            reward = 0.0
        observation = self._get_obs()
        info = self._get_info()
        info["success"] = on_target

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

        for obstacle in self.obstacles or ():
            pygame.draw.rect(
                canvas,
                (64, 64, 64),
                pygame.Rect(
                    pix_square_size * np.array(obstacle),
                    (pix_square_size, pix_square_size),
                ),
            )

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
