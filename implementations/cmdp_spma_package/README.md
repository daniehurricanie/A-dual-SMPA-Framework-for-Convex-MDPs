
# cmdp_spma — SPMA Policy Oracle for CMDPs

This is a minimal, self-contained implementation of the **policy (primal) player** in a convex MDP saddle-point solver using **Softmax Policy Mirror Ascent (SPMA)**. It is designed to plug into an outer dual-ascent loop (not yet included), and comes with:

- Discrete and Gaussian actors
- Armijo backtracking for the SPMA mirror step
- GAE, critic training
- Shaped-reward wrappers for tabular and feature-based dual variables
- Occupancy / feature-expectation estimators

> For the theory and estimator formulas, see our project proposal (Appendix C).

## Installation

```bash
pip install torch gymnasium numpy
```

## Quick start (tabular)

```python
import gymnasium as gym
import numpy as np
from cmdp_spma import PolicyOracleSPMA, OracleConfig, TabularEstimator

def make_env():
    return gym.make("FrozenLake-v1", is_slippery=False)  # Discrete S,A

gamma = 0.99
env = make_env()
nS, nA = env.observation_space.n, env.action_space.n
y = np.zeros((nS, nA), dtype=np.float32)  # dual variable table
est = TabularEstimator(nS, nA, gamma)
cfg = OracleConfig(discrete=True, steps_per_rollout=512, K_inner=3, gamma=gamma)

oracle = PolicyOracleSPMA(make_env, est, cfg)
pol, d_hat, logs = oracle.improve(y, K=3, rollout_steps=1024, seed=0)
print("sum d_hat:", d_hat.sum())
```

## Quick start (feature-based)

```python
import gymnasium as gym
import numpy as np
from cmdp_spma import PolicyOracleSPMA, OracleConfig, FeatureEstimator

def make_env():
    return gym.make("Pendulum-v1")

def phi_fn(s, a):
    s = np.asarray(s, dtype=np.float32)
    a = np.atleast_1d(a).astype(np.float32)
    return np.concatenate([s, a, s**2, a**2])

d =  (3 + 1 + 3 + 1)
w = np.zeros(d, dtype=np.float32)    # dual parameter
y = (phi_fn, w)

gamma = 0.99
est = FeatureEstimator(phi_fn, d, gamma)
cfg = OracleConfig(discrete=False, steps_per_rollout=2048, K_inner=3, gamma=gamma)

oracle = PolicyOracleSPMA(make_env, est, cfg)
pol, ephi_hat, logs = oracle.improve(y, K=3, rollout_steps=4096, seed=0)
print("||E[phi]||:", (ephi_hat**2).sum()**0.5)
```

## Notes

- The SPMA actor loss is:  
  `E[ -Δlogπ * A + (1/η) * ((exp(Δlogπ)-1) - Δlogπ) ]`  
  where `Δlogπ = logπ_new - logπ_old`.
- We run Armijo backtracking on the full batch; if it fails, we take a tiny gradient step.
- For tabular CMDPs, the wrapper replaces the reward with `r_y(s,a) = -y[s,a]`.
- For function approximation CMDPs, the wrapper uses `r_y(s,a) = -phi(s,a)^T w`.

