
import torch
import numpy as np

@torch.no_grad()
def collect_rollouts(env, actor, critic, steps, device, gamma=0.99, lam=0.95):
    obs_list, act_list, rew_list, done_list, logp_list, val_list = [],[],[],[],[],[]
    obs, info = env.reset()
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if hasattr(env.action_space, "n"): # discrete
            a, logp = actor.act(obs_t)
            a_np = int(a.squeeze(0).cpu().item())
        else:
            a, logp = actor.act(obs_t)
            a_np = a.squeeze(0).cpu().numpy()
        v = critic.value(obs_t)
        obs_list.append(obs)
        act_list.append(a_np)
        logp_list.append(logp.squeeze(0).cpu().numpy())
        val_list.append(v.squeeze(0).cpu().numpy())
        obs, r, terminated, truncated, info = env.step(a_np)
        done = terminated or truncated
        rew_list.append(float(r))
        done_list.append(done)
        if done:
            obs, info = env.reset()
    # convert to arrays
    obs = np.array(obs_list, dtype=np.float32)
    logp = np.array(logp_list, dtype=np.float32)
    val = np.array(val_list, dtype=np.float32)
    acts = np.array(act_list, dtype=np.float32)
    rews = np.array(rew_list, dtype=np.float32)
    dones = np.array(done_list, dtype=np.float32)
    # compute advantages and returns
    adv, ret = gae_advantages(rews, val, dones, gamma, lam)
    batch = {
        "obs": obs,
        "acts": acts,
        "rews": rews,
        "dones": dones,
        "adv": adv,
        "ret": ret,
        "old_logp": logp,
    }
    return batch

def gae_advantages(rews, vals, dones, gamma, lam):
    T = len(rews)
    adv = np.zeros_like(rews, dtype=np.float32)
    lastgaelam = 0.0
    nextnonterm = 1.0
    nextv = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rews[t] + gamma * nextv * nextnonterm - vals[t]
        lastgaelam = delta + gamma * lam * nextnonterm * lastgaelam
        adv[t] = lastgaelam
        nextv = vals[t]
        nextnonterm = nonterm
    ret = adv + vals
    # normalize advantages for stability
    if adv.std() > 1e-8:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret
