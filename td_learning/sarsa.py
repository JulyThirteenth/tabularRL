from tabularRL.td_learning.common import (
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_NUM_EPISODES,
    WORLD_SIZE,
    TabularTDAgent,
    make_grid_env,
    run_demo,
)


class Sarsa(TabularTDAgent):
    """Tabular on-policy SARSA agent."""

    def update(self, state, action, reward, next_state, next_action, done):
        """Apply the on-policy SARSA update."""

        td_target = reward + (
            0 if done else self.discount_factor * self.q_table[next_state, next_action]
        )
        self.update_toward(state, action, td_target)


def run_sarsa(
    env,
    num_episodes=DEFAULT_NUM_EPISODES,
    discount_factor=0.99,
    learning_rate=0.1,
    epsilon=0.1,
    max_steps_per_episode=DEFAULT_MAX_EPISODE_STEPS,
    seed=None,
):
    """Train an on-policy SARSA agent and return its greedy policy."""

    agent = Sarsa(
        env,
        discount_factor=discount_factor,
        learning_rate=learning_rate,
        epsilon=epsilon,
        seed=seed,
    )

    for episode in range(num_episodes):
        episode_seed = None if seed is None else seed + episode
        observation, _ = env.reset(seed=episode_seed)
        state = agent.encode_state(observation)
        action = agent.select_action(state)

        for _ in range(max_steps_per_episode):
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.encode_state(next_observation)
            done = terminated or truncated
            next_action = None if done else agent.select_action(next_state)

            agent.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                done,
            )

            if done:
                break

            state = next_state
            action = next_action

    return agent.extract_policy()


def main():
    """Train and evaluate SARSA on GridWorld."""

    run_demo(Sarsa, run_sarsa)


if __name__ == "__main__":
    main()
