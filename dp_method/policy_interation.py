from tabularRL.dp_method.common import DynamicProgramming, run_demo


class PolicyIteration(DynamicProgramming):
    """Dynamic-programming policy iteration with an explicit model.

    Gymnasium's env.step(action) is an interaction API: it advances the current
    environment state. Policy iteration needs a model p(s', r | s, a), so this
    class plans with GridWorldModel instead of sampling env.step().
    """

    def policy_evaluation(self):
        """Evaluate the current deterministic policy to convergence."""

        while self.bellman_sweep(
            lambda state: self.action_value(state, self.policy[state])
        ) >= self.theta:
            pass

    def policy_improvement(self):
        """Greedify the policy and report whether it was already stable."""

        old_policy = self.policy.copy()
        self.extract_policy()
        return bool((old_policy == self.policy).all())

    def run_policy_iteration(self, max_iterations=1000):
        for iteration in range(max_iterations):
            self.policy_evaluation()
            if self.policy_improvement():
                return iteration + 1
        raise RuntimeError("Policy iteration did not converge.")

    def run(self, max_iterations=1000):
        return self.run_policy_iteration(max_iterations)


if __name__ == "__main__":
    run_demo(PolicyIteration, "Policy iteration")
