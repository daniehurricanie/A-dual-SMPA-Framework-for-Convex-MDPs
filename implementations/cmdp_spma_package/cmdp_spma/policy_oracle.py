
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple, Union
import numpy as np
import torch
import gymnasium as gym

from .nets import ActorDiscrete, ActorGaussian, Critic
from .rollout import collect_rollouts
from .spma_losses import spma_actor_loss, mse_loss
from .line_search import _flatten_grads, _flatten_params, _set_params_from_vector, armijo_backtracking
from .shaped_reward_env import TabularShapedReward, FeatureShapedReward
from .occupancy import TabularEstimator, FeatureEstimator

@dataclass
class OracleConfig:
    device: str = "cpu"
    gamma: float = 0.99
    lam: float = 0.95
    inv_eta: float = 5.0            # 1/η in the SPMA term
    max_grad_norm: float = 1.0
    armijo_c: float = 1e-4
    armijo_beta: float = 0.5
    armijo_init: float = 1.0
    armijo_max_steps: int = 15
    critic_lr: float = 3e-4
    K_inner: int = 5
    steps_per_rollout: int = 2048
    discrete: bool = True
    hidden: Tuple[int,int] = (64,64)

class PolicyOracleSPMA:
    """
    Policy oracle that, given a fixed dual y, performs K_inner SPMA updates
    under the shaped reward r_y and returns occupancy estimates.
    """
    def __init__(self, 
                 env_maker: Callable[[], gym.Env],
                 estimator: Union[TabularEstimator, FeatureEstimator],
                 cfg: OracleConfig):
        self.env_maker = env_maker
        self.estimator = estimator
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._obs_dim = None
        self._act_dim = None
        self._n_actions = None

    def _build_models(self, env):
        obs_space = env.observation_space
        act_space = env.action_space
        if hasattr(act_space, "n"):
            n_actions = act_space.n
            self._n_actions = n_actions
            # infer obs_dim
            if hasattr(obs_space, "shape") and obs_space.shape is not None:
                obs_dim = int(np.prod(obs_space.shape))
            else:
                # Discrete observation: represent as one-hot of size obs_space.n
                obs_dim = obs_space.n
            self._obs_dim = obs_dim
            actor = ActorDiscrete(obs_dim, n_actions, hidden=self.cfg.hidden).to(self.device)
        else:
            act_dim = int(np.prod(act_space.shape))
            self._act_dim = act_dim
            obs_dim = int(np.prod(obs_space.shape))
            self._obs_dim = obs_dim
            actor = ActorGaussian(obs_dim, act_dim, hidden=self.cfg.hidden).to(self.device)
        critic = Critic(self._obs_dim, hidden=self.cfg.hidden).to(self.device)
        return actor, critic

    def _ensure_obs_flat(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            return obs
        return obs.reshape(-1)

    def _make_shaped_env(self, y):
        base = self.env_maker()
        if hasattr(base.action_space, "n") and hasattr(base.observation_space, "n"):
            # tabular path
            env = TabularShapedReward(base, y, self.cfg.gamma)
        else:
            # feature-based path
            phi_fn, w = y
            env = FeatureShapedReward(base, phi_fn, w, self.cfg.gamma)
        return env

    def _actor_logp(self, actor, obs, acts):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        if hasattr(acts, "shape") and len(np.shape(acts)) == 1:
            acts_t = torch.as_tensor(acts, dtype=torch.long if isinstance(actor, ActorDiscrete) else torch.float32, device=self.device)
        else:
            acts_t = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        return actor.log_prob(obs_t, acts_t)

    def _update_actor_spma(self, actor, batch):
        obs = batch["obs"]
        acts = batch["acts"]
        adv = batch["adv"]
        old_logp = batch["old_logp"]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # for discrete, ensure actions are long
        if isinstance(actor, ActorDiscrete):
            acts_t = torch.as_tensor(acts, dtype=torch.long, device=self.device)
        else:
            acts_t = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        old_logp_t = old_logp_t.detach()

        def loss_fn():
            logp_new = actor.log_prob(obs_t, acts_t)
            return spma_actor_loss(logp_new, old_logp_t, adv_t, self.cfg.inv_eta)

        # compute gradient at current params
        L0 = float(loss_fn().detach().cpu())
        for p in actor.parameters():
            if p.grad is not None:
                p.grad.zero_()
        L = loss_fn()
        L.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(actor.parameters(), self.cfg.max_grad_norm)
        g = _flatten_grads(actor)
        d = -g  # descent direction
        gTd = float((g * d).sum().detach().cpu())
        if gTd >= 0:
            # if not a descent direction (rare due to numerical issues), just do a tiny step
            gTd = -1.0

        accepted, alpha, Lnew = armijo_backtracking(
            loss_fn, actor, L0, gTd, d, 
            alpha0=self.cfg.armijo_init, beta=self.cfg.armijo_beta, c=self.cfg.armijo_c,
            max_steps=self.cfg.armijo_max_steps
        )
        if not accepted:
            # fallback: small GD step
            step = 1e-3
            with torch.no_grad():
                vec = _flatten_params(actor)
                vec = vec + step * d
                _set_params_from_vector(actor, vec)
            Lnew = float(loss_fn().detach().cpu())
        return L0, Lnew

    def _update_critic(self, critic, batch):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(critic.parameters(), lr=self.cfg.critic_lr)
        for _ in range(10):
            v = critic.value(obs)
            loss = mse_loss(v, ret)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach().cpu())

    def improve(self, y, K: Optional[int]=None, rollout_steps: Optional[int]=None, seed: Optional[int]=None):
        if K is None: K = self.cfg.K_inner
        if rollout_steps is None: rollout_steps = self.cfg.steps_per_rollout
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); 
        env = self._make_shaped_env(y)
        actor, critic = self._build_models(env)
        logs = {"actor_loss_before":[],"actor_loss_after":[],"critic_loss":[]}
        for _ in range(K):
            batch = collect_rollouts(env, actor, critic, rollout_steps, device=self.device, gamma=self.cfg.gamma, lam=self.cfg.lam)
            L0, Lnew = self._update_actor_spma(actor, batch)
            closs = self._update_critic(critic, batch)
            logs["actor_loss_before"].append(L0)
            logs["actor_loss_after"].append(Lnew)
            logs["critic_loss"].append(closs)
            # update occupancy estimator
            if isinstance(self.estimator, TabularEstimator):
                self.estimator.update_from_batch(batch["obs"], batch["acts"], batch["dones"])
            else:
                self.estimator.update_from_batch(batch["obs"], batch["acts"], batch["dones"])
        # return occupancy/feature estimates
        if isinstance(self.estimator, TabularEstimator):
            d_hat = self.estimator.value()
            ydotd = self.estimator.y_dot_d(y)
        else:
            ephi = self.estimator.value()
            if isinstance(y, tuple):
                # y = (phi_fn, w)
                _, w = y
                ydotd = self.estimator.y_dot_d(w)
            else:
                ydotd = float(np.nan)
            d_hat = ephi  # feature expectations
        # freeze policy snapshot (state_dict)
        policy_snapshot = {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        return policy_snapshot, d_hat, {"y_dot_d": ydotd, **logs}
