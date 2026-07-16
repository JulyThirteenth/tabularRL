"""Visual comparison of policy iteration and value iteration on a cluttered grid."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from tabularRL.dp_method.model import GridWorldModel
from tabularRL.dp_method.policy_interation import PolicyIteration
from tabularRL.dp_method.value_interation import ValueIteration

GRID_SIZE = 10
TARGET = np.array([9, 9])
OUTPUT_PATH = Path(__file__).resolve().parent / "dp_policy_comparison.png"

ACTION_VECTORS = {
    0: (0.35, 0.0),   # right
    1: (0.0, 0.35),   # up
    2: (-0.35, 0.0),  # left
    3: (0.0, -0.35),  # down
}


def build_complex_obstacles(size: int = GRID_SIZE) -> list[tuple[int, int]]:
    """Build a maze-like obstacle field with multiple corridors and dead ends."""

    obstacles: set[tuple[int, int]] = set()

    for y in range(1, size - 1):
        obstacles.add((2, y))
    obstacles.discard((2, 4))
    obstacles.discard((2, 7))

    for y in range(2, size - 1):
        obstacles.add((5, y))
    obstacles.discard((5, 2))
    obstacles.discard((5, 6))

    for x in range(3, 7):
        obstacles.add((x, 3))
    obstacles.discard((4, 3))
    obstacles.discard((6, 3))

    obstacles.update((x, 6) for x in range(4, 8))
    obstacles.discard((6, 6))

    obstacles.update((7, y) for y in range(1, 5))
    obstacles.update((x, 1) for x in range(7, size - 1))

    obstacles.discard(tuple(TARGET))
    return sorted(obstacles)


def policy_actions_for_target(planner, target_location: np.ndarray) -> np.ndarray:
    """Return a (size, size) array of greedy actions for one fixed target."""

    model = planner.model
    actions = np.full((model.size, model.size), -1, dtype=int)

    for x in range(model.size):
        for y in range(model.size):
            if model.obstacles is not None and (x, y) in model.obstacles:
                continue

            state = model.encode_state([x, y], target_location)
            if model.is_terminal(state):
                actions[x, y] = -1
            else:
                actions[x, y] = int(planner.policy[state])

    return actions


def plot_policy(
    ax,
    planner,
    target_location: np.ndarray,
    title: str,
) -> None:
    """Draw obstacles, target, and greedy action arrows on one axes."""

    model = planner.model
    size = model.size
    actions = policy_actions_for_target(planner, target_location)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(True, color="#cccccc", linewidth=0.8)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for x in range(size):
        for y in range(size):
            if model.obstacles is not None and (x, y) in model.obstacles:
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="#4d4d4d",
                        edgecolor="#333333",
                        linewidth=0.5,
                        zorder=1,
                    )
                )

    target_x, target_y = target_location
    ax.add_patch(
        Rectangle(
            (target_x - 0.5, target_y - 0.5),
            1,
            1,
            facecolor="#ff6666",
            edgecolor="#cc0000",
            linewidth=1.5,
            zorder=2,
        )
    )
    ax.text(
        target_x,
        target_y,
        "G",
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        fontweight="bold",
        zorder=3,
    )

    arrow_x: list[float] = []
    arrow_y: list[float] = []
    arrow_u: list[float] = []
    arrow_v: list[float] = []

    for x in range(size):
        for y in range(size):
            action = actions[x, y]
            if action < 0:
                continue

            dx, dy = ACTION_VECTORS[action]
            arrow_x.append(x)
            arrow_y.append(y)
            arrow_u.append(dx)
            arrow_v.append(dy)

    ax.quiver(
        arrow_x,
        arrow_y,
        arrow_u,
        arrow_v,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#1f4e79",
        width=0.012,
        zorder=4,
    )


def run_planners(model: GridWorldModel) -> tuple[PolicyIteration, ValueIteration, int, int]:
    """Run both DP algorithms on the same model."""

    policy_planner = PolicyIteration(model)
    value_planner = ValueIteration(model)

    policy_iterations = policy_planner.run()
    value_iterations = value_planner.run()

    return policy_planner, value_planner, policy_iterations, value_iterations


def policies_match_for_target(
    policy_planner: PolicyIteration,
    value_planner: ValueIteration,
    target_location: np.ndarray,
) -> bool:
    """Check whether both planners agree on every non-terminal, non-obstacle cell."""

    model = policy_planner.model

    for x in range(model.size):
        for y in range(model.size):
            if model.obstacles is not None and (x, y) in model.obstacles:
                continue

            state = model.encode_state([x, y], target_location)
            if model.is_terminal(state):
                continue

            if policy_planner.policy[state] != value_planner.policy[state]:
                return False

    return True


def main() -> None:
    obstacles = build_complex_obstacles(GRID_SIZE)
    model = GridWorldModel(size=GRID_SIZE, action_space=4, obstacles=obstacles)

    policy_planner, value_planner, pi_iters, vi_iters = run_planners(model)

    assert policies_match_for_target(policy_planner, value_planner, TARGET), (
        "Policy iteration and value iteration produced different policies."
    )

    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Obstacles: {len(obstacles)}")
    print(f"Target: {TARGET.tolist()}")
    print(f"Policy iteration converged in {pi_iters} outer iterations")
    print(f"Value iteration converged in {vi_iters} sweeps")
    print("Policies match for the fixed target.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    plot_policy(
        axes[0],
        policy_planner,
        TARGET,
        f"Policy Iteration ({pi_iters} iterations)\nTarget = {TARGET.tolist()}",
    )
    plot_policy(
        axes[1],
        value_planner,
        TARGET,
        f"Value Iteration ({vi_iters} sweeps)\nTarget = {TARGET.tolist()}",
    )

    fig.suptitle(
        "Greedy Policies on a Cluttered GridWorld",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
