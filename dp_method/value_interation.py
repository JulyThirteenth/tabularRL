from tabularRL.dp_method.common import DynamicProgramming, run_demo


class ValueIteration(DynamicProgramming):
    """Dynamic-programming value iteration with an explicit GridWorld model."""

    def value_update(self):
        """Apply one full sweep of Bellman optimality updates."""

        return self.bellman_sweep(lambda state: self.action_values(state).max())

    def run_value_iteration(self, max_iterations=1000):
        """Iterate Bellman optimality backups until convergence."""

        for iteration in range(max_iterations):
            delta = self.value_update()
            if delta < self.theta:
                self.extract_policy()
                return iteration + 1
        raise RuntimeError("Value iteration did not converge.")

    def run(self, max_iterations=1000):
        return self.run_value_iteration(max_iterations)


if __name__ == "__main__":
    run_demo(ValueIteration, "Value iteration")
