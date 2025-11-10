
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

def mlp(sizes, activation=nn.Tanh, out_act=None):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else (out_act if out_act is not None else nn.Identity)
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

class ActorDiscrete(nn.Module):
    """
    Softmax policy over discrete actions.
    """
    def __init__(self, obs_dim:int, n_actions:int, hidden=(64,64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions])
    
    def logits(self, obs:torch.Tensor)->torch.Tensor:
        return self.net(obs)
    
    def log_prob(self, obs:torch.Tensor, act:torch.Tensor)->torch.Tensor:
        logits = self.logits(obs)
        pi = Categorical(logits=logits)
        return pi.log_prob(act)
    
    def act(self, obs:torch.Tensor):
        logits = self.logits(obs)
        pi = Categorical(logits=logits)
        a = pi.sample()
        logp = pi.log_prob(a)
        return a, logp

class ActorGaussian(nn.Module):
    """
    Gaussian policy with state-dependent mean and state-independent log std.
    """
    def __init__(self, obs_dim:int, act_dim:int, hidden=(64,64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def dist(self, obs:torch.Tensor):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
    
    def log_prob(self, obs:torch.Tensor, act:torch.Tensor)->torch.Tensor:
        dist = self.dist(obs)
        return dist.log_prob(act).sum(axis=-1)
    
    def act(self, obs:torch.Tensor):
        dist = self.dist(obs)
        a = dist.sample()
        logp = dist.log_prob(a).sum(axis=-1)
        return a, logp

class Critic(nn.Module):
    def __init__(self, obs_dim:int, hidden=(64,64)):
        super().__init__()
        self.v = mlp([obs_dim, *hidden, 1])
    
    def value(self, obs:torch.Tensor)->torch.Tensor:
        return self.v(obs).squeeze(-1)
