
import numpy as np
import gymnasium as gym
from cmdp_spma import PolicyOracleSPMA, OracleConfig, FeatureEstimator

def phi_fn(s, a):
    s = np.asarray(s, dtype=np.float32)
    a = np.atleast_1d(a).astype(np.float32)
    return np.concatenate([s, a, s**2, a**2])

def main():
    def make_env():
        return gym.make("Pendulum-v1")
    gamma = 0.99
    d =  (3 + 1 + 3 + 1)
    w = np.zeros(d, dtype=np.float32)
    y = (phi_fn, w)
    est = FeatureEstimator(phi_fn, d, gamma)
    cfg = OracleConfig(discrete=False, steps_per_rollout=2048, K_inner=3, gamma=gamma)
    oracle = PolicyOracleSPMA(make_env, est, cfg)
    pol, ephi_hat, logs = oracle.improve(y, K=2, rollout_steps=4096, seed=0)
    print("||E[phi]||:", float((ephi_hat**2).sum()**0.5))
    print("y·d:", logs["y_dot_d"])

if __name__ == "__main__":
    main()
