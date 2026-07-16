from tabularRL.td_learning.common import (
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_NUM_EPISODES,
    WORLD_SIZE,
    TabularTDAgent,
    make_grid_env,
    run_demo,
)


class QLearning(TabularTDAgent):
    """Off-policy tabular TD control using a greedy bootstrap target."""

    def update(self, state, action, reward, next_state, done):
        """Update the Q-table using the Q-learning update rule."""

        td_target = reward + (
            0 if done else self.discount_factor * self.q_table[next_state].max()
        )
        self.update_toward(state, action, td_target)


def run_q_learning(
    env,
    num_episodes=DEFAULT_NUM_EPISODES,
    discount_factor=0.99,
    learning_rate=0.1,
    epsilon=0.1,
    max_steps_per_episode=DEFAULT_MAX_EPISODE_STEPS,
    seed=None,
):
    """Run Q-learning on the given environment."""

    q_learning_agent = QLearning(
        env,
        discount_factor=discount_factor,
        learning_rate=learning_rate,
        epsilon=epsilon,
        seed=seed,
    )

    for episode in range(num_episodes):
        episode_seed = None if seed is None else seed + episode
        observation, _ = env.reset(seed=episode_seed)
        state = q_learning_agent.encode_state(observation)

        for _ in range(max_steps_per_episode):
            action = q_learning_agent.select_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = q_learning_agent.encode_state(next_observation)
            done = terminated or truncated
            q_learning_agent.update(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

    return q_learning_agent.extract_policy()


def main():
    """Train and evaluate Q-learning on GridWorld."""

    run_demo(QLearning, run_q_learning)


if __name__ == "__main__":
    main()
