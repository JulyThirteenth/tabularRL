import numpy as np

from tabularRL.dp_method.model import GridWorldModel, WORLD_SIZE, make_grid_env


class DynamicProgramming:
    """Shared Bellman operations for model-based tabular planning."""

    def __init__(self, model, discount_factor=0.99, theta=1e-10):
        self.model = model
        self.discount_factor = discount_factor
        self.theta = theta
        self.value_function = np.zeros(model.state_space, dtype=np.float64)
        self.policy = np.zeros(model.state_space, dtype=np.int64)

    def action_value(self, state, action):
        """Compute a one-step Bellman backup for one state-action pair."""

        next_state, reward, done = self.model.transition(state, action)
        bootstrap = 0.0 if done else self.value_function[next_state]
        return reward + self.discount_factor * bootstrap

    def action_values(self, state):
        """Compute Bellman backup values for every action in one state."""

        return np.array(
            [
                self.action_value(state, action)
                for action in range(self.model.action_space)
            ],
            dtype=np.float64,
        )

    def bellman_sweep(self, backup):
        """Apply one in-place value-function sweep and return its max change."""

        delta = 0.0
        for state in range(self.model.state_space):
            old_value = self.value_function[state]
            self.value_function[state] = (
                0.0 if self.model.is_terminal(state) else backup(state)
            )
            delta = max(delta, abs(old_value - self.value_function[state]))
        return delta

    def extract_policy(self):
        """Make the stored policy greedy with respect to the value function."""

        for state in range(self.model.state_space):
            if not self.model.is_terminal(state):
                self.policy[state] = int(np.argmax(self.action_values(state)))
        return self.policy

    def policy_grid_for_target(self, target_location):
        """Return the policy as action symbols for one fixed target."""

        action_symbols = np.array([">", "^", "<", "v"])
        grid = []
        for y in reversed(range(self.model.size)):
            row = []
            for x in range(self.model.size):
                if self.model.obstacles is not None and (x, y) in self.model.obstacles:
                    row.append("#")
                    continue

                state = self.model.encode_state([x, y], target_location)
                symbol = (
                    "T"
                    if self.model.is_terminal(state)
                    else action_symbols[self.policy[state]]
                )
                row.append(symbol)
            grid.append(row)
        return grid


def run_demo(planner_class, algorithm_name):
    """Run and display one planning algorithm on the shared GridWorld model."""

    env = make_grid_env()
    try:
        planner = planner_class(GridWorldModel.from_env(env))
        iterations = planner.run()
        target = np.array([(WORLD_SIZE - 1) // 2] * 2)

        print(f"{algorithm_name} converged in {iterations} iterations")
        print("Greedy policy for target", target.tolist())
        for row in planner.policy_grid_for_target(target):
            print(" ".join(row))
    finally:
        env.close()
