from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import tabularRL.grid_env

WORLD_SIZE = 10
DEFAULT_NUM_EPISODES = 10_000
DEFAULT_MAX_EPISODE_STEPS = 200
DEFAULT_EVAL_EPISODES = 500
DEFAULT_MAX_EVAL_STEPS = 100
DEFAULT_SEED = 0
EVALUATION_SEED = 100_000


def make_grid_env(render_mode=None, obstacles=None, obstacle_penalty=-1.0):
    return gym.make(
        "grid_env/GridWorld-v0",
        size=WORLD_SIZE,
        render_mode=render_mode,
        obstacles=obstacles,
        obstacle_penalty=obstacle_penalty,
    )


class TabularTDAgent:
    """Shared state representation, value table, and behavior policy."""

    def __init__(
        self,
        env,
        discount_factor=0.99,
        learning_rate=0.1,
        epsilon=0.1,
        seed=None,
    ):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Tabular TD control requires a discrete action space.")

        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.num_states = env.observation_space.n
            self._observation_type = "discrete"
        elif isinstance(env.observation_space, gym.spaces.Dict) and {
            "agent",
            "target",
        }.issubset(env.observation_space.spaces):
            self.world_size = env.unwrapped.size
            self.num_states = self.world_size**4
            self._observation_type = "grid_world"
        else:
            raise TypeError(
                "Tabular TD control requires either a Discrete observation space "
                "or a GridWorld Dict observation with 'agent' and 'target'."
            )

        self.q_table = np.zeros((self.num_states, env.action_space.n), dtype=np.float64)

    def encode_state(self, observation):
        """Encode an observation as an integer Q-table index."""

        if self._observation_type == "discrete":
            return int(observation)

        agent_x, agent_y = np.asarray(observation["agent"], dtype=int)
        target_x, target_y = np.asarray(observation["target"], dtype=int)
        return int(
            ((agent_x * self.world_size + agent_y) * self.world_size + target_x)
            * self.world_size
            + target_y
        )

    def update_toward(self, state, action, td_target):
        """Move one Q-value toward an algorithm-specific TD target."""

        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def greedy_action(self, state):
        """Choose randomly among actions tied for the largest Q-value."""

        action_values = self.q_table[state]
        greedy_actions = np.flatnonzero(action_values == np.max(action_values))
        return int(self.rng.choice(greedy_actions))

    def select_action(self, state):
        """Select an action using an epsilon-greedy behavior policy."""

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.env.action_space.n))
        return self.greedy_action(state)

    def extract_policy(self):
        """Extract one greedy action for every encoded state."""

        return np.array(
            [self.greedy_action(state) for state in range(self.num_states)],
            dtype=int,
        )


@dataclass(frozen=True)
class EvaluationResult:
    success_rate: float
    mean_episode_length: float


def evaluate_policy(
    env,
    policy,
    state_encoder,
    num_episodes=DEFAULT_EVAL_EPISODES,
    max_steps=DEFAULT_MAX_EVAL_STEPS,
    seed=EVALUATION_SEED,
):
    """Evaluate a deterministic policy on a reproducible set of episodes."""

    successes = 0
    episode_lengths = []

    for episode in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode)
        state = state_encoder.encode_state(observation)

        for step in range(1, max_steps + 1):
            observation, _, terminated, truncated, info = env.step(int(policy[state]))
            state = state_encoder.encode_state(observation)

            if terminated or truncated:
                successes += int(info.get("success", False))
                episode_lengths.append(step)
                break
        else:
            episode_lengths.append(max_steps)

    return EvaluationResult(
        success_rate=successes / num_episodes,
        mean_episode_length=float(np.mean(episode_lengths)),
    )


def run_demo(agent_class, train_function, seed=DEFAULT_SEED):
    """Train and evaluate one tabular TD-control algorithm on GridWorld."""

    env = make_grid_env()
    try:
        policy = train_function(env, seed=seed)
        state_encoder = agent_class(env, epsilon=0.0, seed=seed)
        result = evaluate_policy(env, policy, state_encoder)
        print(f"Policy shape: {policy.shape}")
        print(f"Evaluation success rate: {result.success_rate:.1%}")
        print(f"Mean episode length: {result.mean_episode_length:.2f} steps")
    finally:
        env.close()
