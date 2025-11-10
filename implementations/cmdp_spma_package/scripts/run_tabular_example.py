
import numpy as np
import gymnasium as gym
from cmdp_spma import PolicyOracleSPMA, OracleConfig, TabularEstimator

def main():
    def make_env():
        return gym.make("FrozenLake-v1", is_slippery=False)
    env = make_env()
    nS, nA = env.observation_space.n, env.action_space.n
    gamma = 0.99
    y = np.zeros((nS, nA), dtype=np.float32)
    est = TabularEstimator(nS, nA, gamma)
    cfg = OracleConfig(discrete=True, steps_per_rollout=1024, K_inner=3, gamma=gamma)
    oracle = PolicyOracleSPMA(make_env, est, cfg)
    pol, d_hat, logs = oracle.improve(y, K=3, rollout_steps=2048, seed=0)
    print("sum d_hat:", float(d_hat.sum()))
    print("y·d:", logs["y_dot_d"])

if __name__ == "__main__":
    main()
