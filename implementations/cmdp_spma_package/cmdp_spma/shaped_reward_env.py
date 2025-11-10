
import gymnasium as gym
import numpy as np

class TabularShapedReward(gym.Wrapper):
    """
    Wraps a discrete env and replaces reward with r_y(s,a) = -y_table[s,a].
    Assumes observation_space is Discrete and action_space is Discrete.
    """
    def __init__(self, env, y_table, gamma: float):
        assert isinstance(env.observation_space, gym.spaces.Discrete), "TabularShapedReward needs Discrete obs"
        assert isinstance(env.action_space, gym.spaces.Discrete), "TabularShapedReward needs Discrete actions"
        super().__init__(env)
        self.y = np.array(y_table, dtype=np.float32)
        self.gamma = gamma
        self._last_s = None
    
    def reset(self, **kwargs):
        s, info = self.env.reset(**kwargs)
        self._last_s = int(s)
        return s, info
    
    def step(self, a):
        s_next, _, terminated, truncated, info = self.env.step(a)
        s = self._last_s
        shaped_r = - float(self.y[s, a])
        self._last_s = int(s_next)
        return s_next, shaped_r, terminated, truncated, info

class FeatureShapedReward(gym.Wrapper):
    """
    Wraps an env and replaces reward with r_y(s,a) = - phi(s,a)^T w.
    phi_fn: callable (s, a) -> np.ndarray[d]
    w: np.ndarray[d]
    """
    def __init__(self, env, phi_fn, w, gamma: float):
        super().__init__(env)
        self.phi_fn = phi_fn
        self.w = np.array(w, dtype=np.float32)
        self.gamma = gamma
        self._last_obs = None
    
    def reset(self, **kwargs):
        s, info = self.env.reset(**kwargs)
        self._last_obs = s
        return s, info
    
    def step(self, a):
        s_next, _, terminated, truncated, info = self.env.step(a)
        s = self._last_obs
        phi = self.phi_fn(s, a)
        shaped_r = - float(np.dot(phi, self.w))
        self._last_obs = s_next
        return s_next, shaped_r, terminated, truncated, info
