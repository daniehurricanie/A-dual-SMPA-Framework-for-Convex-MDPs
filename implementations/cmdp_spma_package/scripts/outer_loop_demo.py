
"""
Outer-loop demo:
min_d f(d) with f*(y) = (λ/2)||y||^2  ⇒  ∇_y L(π,y) = d_π - λ y

We run a few outer iterations on FrozenLake (tabular), using the PolicyOracleSPMA
to produce d̂_π, and update y by gradient ascent on L(π,y).
"""
import numpy as np
import gymnasium as gym
from cmdp_spma import PolicyOracleSPMA, OracleConfig, TabularEstimator

def main():
    def make_env():
        return gym.make("FrozenLake-v1", is_slippery=False)
    env = make_env()
    nS, nA = env.observation_space.n, env.action_space.n
    gamma = 0.99

    # dual init y_0
    y = np.zeros((nS, nA), dtype=np.float32)
    lam = 0.1   # f*(y) = (lam/2)||y||^2
    alpha = 0.5 # dual step size

    cfg = OracleConfig(discrete=True, steps_per_rollout=1024, K_inner=2, gamma=gamma)
    est = TabularEstimator(nS, nA, gamma)
    oracle = PolicyOracleSPMA(make_env, est, cfg)

    for k in range(5):
        # policy (primal) step under r_y = -y
        pol, d_hat, logs = oracle.improve(y, K=2, rollout_steps=2048, seed=42+k)
        # dual ascent
        grad_y = d_hat - lam * y
        y = y + alpha * grad_y
        # report
        L = (y * d_hat).sum() - 0.5 * lam * (y**2).sum()
        print(f"[iter {k}] sum d={float(d_hat.sum()):.3f}, y·d={float((y*d_hat).sum()):.4f}, L={float(L):.4f}")
    print("Done.")

if __name__ == "__main__":
    main()
