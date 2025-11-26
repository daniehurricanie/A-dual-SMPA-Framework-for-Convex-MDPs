# cmdp_spma — Dual-SPMA for Constrained MDPs

A self-contained implementation of **Softmax Policy Mirror Ascent (SPMA)** for solving constrained MDPs via saddle-point optimization. Includes both the policy oracle (primal player) and outer dual-ascent loops, plus an **NPG-PD baseline** for comparison.

## Features

| Component | Description |
|-----------|-------------|
| **SPMA Policy Oracle** | Mirror-descent policy updates with Armijo line search |
| **NPG-PD Baseline** | Natural Policy Gradient with Primal-Dual updates |
| **Outer Dual Loops** | Generic dual ascent for entropy-regularized RL and constrained safety |
| **Objective Classes** | Pluggable objectives (entropy, quadratic, Lagrangian CMDP) |
| **Occupancy Estimators** | Tabular and feature-based discounted occupancy estimation |
| **Shaped Reward Wrappers** | Automatic reward shaping r_y(s,a) = -y[s,a] |

> For theory and estimator formulas, see our project proposal (Appendix C).

## Installation

```bash
pip install torch gymnasium numpy
```

## Project Structure

```
cmdp_spma_package/
├── cmdp_spma/
│   ├── __init__.py
│   ├── helpers.py           # build_reward_table, build_uniform_cost_table, estimate_Jr_Jc
│   ├── line_search.py       # Armijo backtracking
│   ├── nets.py              # ActorDiscrete, ActorGaussian, Critic
│   ├── npg_pd.py            # NPG-PD baseline (diagonal Fisher)
│   ├── objectives.py        # EntropyRegularizedObjective, QuadraticObjective, ConstrainedSafetyObjective
│   ├── occupancy.py         # TabularEstimator, FeatureEstimator
│   ├── policy_oracle.py     # PolicyOracleSPMA, OracleConfig
│   ├── rollout.py           # collect_rollouts, GAE
│   ├── shaped_reward_env.py # TabularShapedReward, FeatureShapedReward
│   └── spma_losses.py       # spma_actor_loss, mse_loss
├── scripts/
│   ├── outer_loop_demo.py
│   ├── run_constrained_safety_npg_pd.py  # NPG-PD baseline
│   ├── run_constrained_safety_spma.py    # SPMA for constrained safety
│   ├── run_dual_spma_tabular.py          # Generic dual-SPMA loop
│   ├── run_feature_example.py
│   └── run_tabular_example.py
├── tests/
│   └── test_estimator.py
└── README.md
```

---

## Quick Start

### 1. Basic SPMA Policy Oracle (Tabular)

```python
import gymnasium as gym
import numpy as np
from cmdp_spma import PolicyOracleSPMA, OracleConfig, TabularEstimator

def make_env():
    return gym.make("FrozenLake-v1", is_slippery=False)

gamma = 0.99
env = make_env()
nS, nA = env.observation_space.n, env.action_space.n
y = np.zeros((nS, nA), dtype=np.float32)  # dual variable

est = TabularEstimator(nS, nA, gamma)
cfg = OracleConfig(discrete=True, steps_per_rollout=512, K_inner=3, gamma=gamma)
oracle = PolicyOracleSPMA(make_env, est, cfg)

pol, d_hat, logs = oracle.improve(y, K=3, rollout_steps=1024, seed=0)
print("sum d_hat:", d_hat.sum())
```

### 2. Entropy-Regularized RL (Outer Dual Loop)

```python
import gymnasium as gym
from cmdp_spma import (
    EntropyRegularizedObjective,
    build_reward_table,
)
from scripts.run_dual_spma_tabular import run_dual_spma_tabular

def make_env():
    return gym.make("FrozenLake-v1", is_slippery=False)

env = make_env()
r_table = build_reward_table(env)
obj = EntropyRegularizedObjective(r_table, alpha=0.1)

y_final, history = run_dual_spma_tabular(
    make_env, obj,
    K_outer=20, gamma=0.99, alpha_y=0.5,
    cfg_kwargs=dict(steps_per_rollout=1024, K_inner=3),
)
```

### 3. Constrained Safety CMDP (SPMA)

```python
import gymnasium as gym
from cmdp_spma import build_reward_table, build_uniform_cost_table
from scripts.run_constrained_safety_spma import run_constrained_safety_spma

def make_env():
    return gym.make("FrozenLake-v1", is_slippery=False)

env = make_env()
r_table = build_reward_table(env)
c_table = build_uniform_cost_table(env, bad_states=[5, 7, 11, 12], cost_bad=1.0)

lam_final, history = run_constrained_safety_spma(
    make_env,
    reward_table=r_table,
    cost_table=c_table,
    tau=0.1,  # safety threshold
    K_outer=20,
    alpha_lambda=0.5,
)
```

### 4. Constrained Safety CMDP (NPG-PD Baseline)

```python
from scripts.run_constrained_safety_npg_pd import run_constrained_safety_npg_pd

actor, history = run_constrained_safety_npg_pd(
    make_env=make_env,
    reward_table=r_table,
    cost_table=c_table,
    tau=0.1,
    K_outer=20,
    npg_step=0.05,
    beta_lambda=0.5,
)
```

### 5. Feature-Based (Continuous Actions)

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

d = (3 + 1 + 3 + 1)
w = np.zeros(d, dtype=np.float32)
y = (phi_fn, w)

gamma = 0.99
est = FeatureEstimator(phi_fn, d, gamma)
cfg = OracleConfig(discrete=False, steps_per_rollout=2048, K_inner=3, gamma=gamma)
oracle = PolicyOracleSPMA(make_env, est, cfg)

pol, ephi_hat, logs = oracle.improve(y, K=3, rollout_steps=4096, seed=0)
print("||E[phi]||:", (ephi_hat**2).sum()**0.5)
```

---

## API Reference

### Objective Classes

| Class | Problem | Conjugate f*(y) |
|-------|---------|-----------------|
| `EntropyRegularizedObjective` | max J_r + α H(d) | α log Σ exp((y+r)/α) |
| `QuadraticObjective` | min ‖d‖² | (λ/2) ‖y‖² |
| `ConstrainedSafetyObjective` | max J_r s.t. J_c ≤ τ | Lagrangian with scalar λ |

### Estimators

| Class | Use Case | Output |
|-------|----------|--------|
| `TabularEstimator` | Discrete S×A | d_hat[s,a] occupancy table |
| `FeatureEstimator` | Function approx | E[φ(s,a)] feature expectations |

Both have a `reset()` method to clear data for per-iteration estimation.

### Policy Updates

| Method | Algorithm | Update Rule |
|--------|-----------|-------------|
| `PolicyOracleSPMA.improve()` | SPMA | Mirror descent + Armijo |
| `npg_actor_step_diag()` | NPG-PD | θ ← θ + α F⁻¹ ∇J |

---

## Algorithm Details

### SPMA Actor Loss

```
L = E[ -Δlogπ · A + (1/η) · ((exp(Δlogπ) - 1) - Δlogπ) ]
```

where `Δlogπ = log π_new - log π_old`. This is the mirror descent update with KL divergence Bregman regularization.

### NPG-PD Update

```
θ ← θ + α · F_diag⁻¹ · ∇_θ J_{r_λ}(π)
λ ← [λ + β (J_c(π) - τ)]_+
```

where `F_diag` is the diagonal Fisher information matrix and `r_λ = r - λc`.

### Outer Dual Loop

For entropy-regularized RL:
```
y_{k+1} = y_k + α_y (d̂_π_k - ∇f*(y_k))
```

For constrained safety (Lagrangian):
```
λ_{k+1} = [λ_k + α_λ (J_c(π_k) - τ)]_+
```

### Shaped Reward

- **Tabular:** `r_y(s,a) = -y[s,a]`
- **Feature-based:** `r_y(s,a) = -φ(s,a)ᵀ w`

---

## Configuration

### OracleConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `inv_eta` | 5.0 | 1/η for SPMA Bregman term |
| `K_inner` | 5 | Inner SPMA iterations per outer step |
| `steps_per_rollout` | 2048 | Environment steps per rollout |
| `persistent_policy` | True | Keep actor/critic across outer iterations |
| `hidden` | (64, 64) | MLP hidden layer sizes |

---

## Logging & Metrics

The outer loops track:

| Metric | Description |
|--------|-------------|
| `L` | Saddle value: y·d - f*(y) |
| `f_d` | Primal objective: f(d_hat) |
| `Jr` | Return under original reward |
| `Jc` | Return under cost (for CMDP) |
| `constraint_violation` | J_c - τ |
| `wall_time` | Cumulative runtime |

---

## Tests

Run all tests:

```bash
cd cmdp_spma_package
python -m pytest tests/
```

Or run directly:

```python
from tests.test_estimator import run_all_tests
run_all_tests()
```

Available tests:
- `test_simple_sum_to_one` — Occupancy sums to ~1
- `test_feature_estimator_indicator` — φ(s,a) = e_i recovers correct coordinate
- `test_shaped_reward_affects_returns` — Shaped reward r_y = -y works correctly
- `test_spma_bandit_convergence` — Policy converges to best arm in bandit

---

## Comparison: SPMA vs NPG-PD

| Aspect | SPMA | NPG-PD |
|--------|------|--------|
| **Primal update** | Mirror descent (KL Bregman) | Natural gradient (Fisher) |
| **Step size** | Armijo line search | Fixed α with F⁻¹ scaling |
| **Theory** | Convex optimization on d | Policy gradient on θ |
| **Convergence** | O(1/√K) for convex f | O(1/√K) for smooth J |

Use `run_constrained_safety_spma.py` and `run_constrained_safety_npg_pd.py` with the same reward/cost tables to compare empirically.

---

## References

- **SPMA:** Softmax Policy Mirror Ascent for convex MDPs
- **NPG-PD:** Ding et al., "Natural Policy Gradient Primal-Dual Method for Constrained Markov Decision Processes"
- **Occupancy estimation:** See project proposal Appendix C

---

## License

MIT