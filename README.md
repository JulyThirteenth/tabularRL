# tabularRL

`tabularRL` is a compact educational implementation of tabular reinforcement-learning algorithms on a deterministic Gymnasium GridWorld. It places model-based dynamic programming and model-free temporal-difference control behind small shared abstractions while keeping their algorithm-specific update rules explicit.

## Implemented algorithms

### Dynamic programming

- Policy Iteration
- Value Iteration

Both algorithms use an explicit `GridWorldModel` that can query transitions for arbitrary state-action pairs. They share Bellman backups, value storage, greedy policy extraction, and result display through `DynamicProgramming`.

### Temporal-difference control

- Q-learning
- SARSA

Both algorithms learn directly from Gymnasium interactions. They share state encoding, a tabular action-value function, epsilon-greedy action selection, TD-error updates, and evaluation through `TabularTDAgent`.

## Repository structure

```text
tabularRL/
├── grid_env/                 # GridWorld environment and registration
├── dp_method/
│   ├── model.py              # Explicit transition model
│   ├── common.py             # Shared DynamicProgramming abstraction
│   ├── policy_interation.py  # Policy Iteration
│   └── value_interation.py   # Value Iteration
├── td_learning/
│   ├── common.py             # Shared TabularTDAgent and evaluation
│   ├── q_learning.py         # Q-learning
│   └── sarsa.py              # SARSA
└── pyproject.toml
```

The GridWorld state contains both the agent and target coordinates:

```text
(agent_x, agent_y, target_x, target_y)
```

For an `n x n` grid, this produces `n^4` encoded states and four discrete actions.

## Custom obstacles

You can add static obstacles to the grid when creating the environment. Obstacle cells are shown in gray during rendering. The agent and target are never sampled on an obstacle.

If the agent steps onto an obstacle, the episode terminates immediately with a configurable penalty (default `-1.0`). Reaching the target still gives reward `+1.0`. Evaluation uses `info["success"]` so obstacle failures are not counted as successes.

```python
import gymnasium as gym
import tabularRL.grid_env

obstacles = [(1, 1), (1, 2), (2, 1), (3, 3)]

env = gym.make(
    "grid_env/GridWorld-v0",
    size=5,
    obstacles=obstacles,
    obstacle_penalty=-1.0,
)

observation, info = env.reset(seed=0, options={
    "agent_location": [0, 0],
    "target_location": [4, 4],
})
```

The DP and TD helpers accept the same keyword arguments:

```python
from tabularRL.dp_method.model import make_grid_env as make_dp_env
from tabularRL.td_learning.common import make_grid_env as make_td_env

env = make_dp_env(obstacles=obstacles)
env = make_td_env(obstacles=obstacles, obstacle_penalty=-1.0)
```

Policy visualization marks obstacle cells with `#`.

## Environment setup

The project requires Python 3.10 or newer. A Conda environment is recommended.

```bash
git clone https://github.com/JulyThirteenth/tabularRL.git
cd tabularRL

conda create -n tabularrl python=3.10 -y
conda activate tabularrl

python -m pip install --upgrade pip
python -m pip install -e .
```

The editable installation installs NumPy, Gymnasium, and Pygame and makes the `tabularRL` package available from any working directory. The bundled GridWorld is registered automatically; it does not require a separate installation step.

To verify the installation:

```bash
python -c "import tabularRL; print(tabularRL.__file__)"
```

## Running the algorithms

Run all commands from the repository root or any directory after the editable installation.

### Policy Iteration

```bash
python -m tabularRL.dp_method.policy_interation
```

### Value Iteration

```bash
python -m tabularRL.dp_method.value_interation
```

### Q-learning

```bash
python -m tabularRL.td_learning.q_learning
```

### SARSA

```bash
python -m tabularRL.td_learning.sarsa
```

## Current reference results

The default configuration uses a `5 x 5` GridWorld for dynamic programming and a `10 x 10` GridWorld for TD control.

| Method | Default result |
| --- | ---: |
| Policy Iteration | converged in 9 outer iterations |
| Value Iteration | converged in 9 sweeps |
| Q-learning | 83.4% success over 500 evaluation episodes |
| SARSA | 83.8% success over 500 evaluation episodes |

Policy Iteration and Value Iteration produce identical policies and value functions over all 625 states in the default planning task, with zero Bellman residual. Q-learning and SARSA use the same training budget, hyperparameters, evaluation states, and random seed for a directly comparable smoke test.

These single-seed TD results validate the implementation but should not be treated as a general ranking of the algorithms. Reliable empirical comparison requires multiple random seeds, learning curves, and uncertainty estimates.

## Core update rules

Q-learning uses an off-policy greedy bootstrap target:

```text
R[t+1] + gamma * max_a Q(S[t+1], a)
```

SARSA uses the action selected by its current behavior policy:

```text
R[t+1] + gamma * Q(S[t+1], A[t+1])
```

This difference remains explicit in the two training loops even though the algorithms share the same tabular agent infrastructure.
