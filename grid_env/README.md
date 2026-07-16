# GridWorld environment

This directory contains the deterministic Gymnasium GridWorld bundled with `tabularRL`. Install the project from the repository root:

```bash
python -m pip install -e .
```

Importing `tabularRL.grid_env` registers `grid_env/GridWorld-v0`. The DP and TD entry points perform this import automatically.

Optional constructor kwargs:

- `obstacles`: list of `(x, y)` cells that end the episode with a penalty when entered
- `obstacle_penalty`: reward received on obstacle contact (default `-1.0`)
