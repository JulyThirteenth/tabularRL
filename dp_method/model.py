import gymnasium as gym
import numpy as np

import tabularRL.grid_env

WORLD_SIZE = 5


def make_grid_env(render_mode=None, obstacles=None, obstacle_penalty=-1.0):
    return gym.make(
        "grid_env/GridWorld-v0",
        size=WORLD_SIZE,
        render_mode=render_mode,
        obstacles=obstacles,
        obstacle_penalty=obstacle_penalty,
    )


class GridWorldModel:
    """Deterministic transition model for the GridWorld task."""

    @staticmethod
    def normalize_obstacles(obstacles, size):
        """Convert obstacle coordinates to a validated frozenset, or None if unset."""

        if obstacles is None:
            return None

        normalized = frozenset((int(x), int(y)) for x, y in obstacles)
        for x, y in normalized:
            if not (0 <= x < size and 0 <= y < size):
                raise ValueError(
                    f"Obstacle ({x}, {y}) is out of bounds for a {size}x{size} grid."
                )
        return normalized

    def __init__(
        self,
        size,
        action_space,
        obstacles=None,
        obstacle_penalty=-1.0,
    ):
        self.size = size
        self.action_space = action_space
        self.obstacles = self.normalize_obstacles(obstacles, size)
        self.obstacle_penalty = float(obstacle_penalty)
        self.state_space = self.size**4
        self.action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    @classmethod
    def from_env(cls, env):
        """Build a planning model from the environment metadata."""

        unwrapped = env.unwrapped
        return cls(
            size=unwrapped.size,
            action_space=env.action_space.n,
            obstacles=unwrapped.obstacles,
            obstacle_penalty=unwrapped.obstacle_penalty,
        )

    def encode_state(self, agent_location, target_location):
        """Map two grid coordinates to one integer state id."""

        ax, ay = np.asarray(agent_location, dtype=int)
        tx, ty = np.asarray(target_location, dtype=int)
        return ((ax * self.size + ay) * self.size + tx) * self.size + ty

    def decode_state(self, state):
        """Map one integer state id back to agent and target coordinates."""

        ty = state % self.size
        state //= self.size
        tx = state % self.size
        state //= self.size
        ay = state % self.size
        ax = state // self.size
        return np.array([ax, ay]), np.array([tx, ty])

    def is_terminal(self, state):
        agent_location, target_location = self.decode_state(state)
        if np.array_equal(agent_location, target_location):
            return True
        return self.obstacles is not None and tuple(agent_location) in self.obstacles

    def transition(self, state, action):
        """Deterministic model: return next_state, reward, done."""

        agent_location, target_location = self.decode_state(state)
        if self.is_terminal(state):
            return state, 0.0, True

        next_agent_location = np.clip(
            agent_location + self.action_to_direction[action],
            0,
            self.size - 1,
        )
        if self.obstacles is not None and tuple(next_agent_location) in self.obstacles:
            next_state = self.encode_state(next_agent_location, target_location)
            return next_state, self.obstacle_penalty, True

        done = np.array_equal(next_agent_location, target_location)
        reward = 1.0 if done else 0.0
        next_state = self.encode_state(next_agent_location, target_location)
        return next_state, reward, done
