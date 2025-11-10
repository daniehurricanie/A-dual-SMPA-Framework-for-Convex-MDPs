
import torch

def spma_actor_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, inv_eta: float):
    """
    SPMA mirror-descent style loss on a fixed batch:
        L = E[ - Δlogπ * A  +  (1/η) * ( (exp(Δlogπ) - 1) - Δlogπ ) ]
    where Δlogπ = logp_new - logp_old, inv_eta = 1/η.
    """
    dlog = logp_new - logp_old
    dlog_clamped = torch.clamp(dlog, -30.0, 30.0)  # numerical safety
    breg = torch.expm1(dlog_clamped) - dlog_clamped
    # stop grad through old logp and advantages
    loss = (-dlog * adv).mean() + (inv_eta * breg).mean()
    return loss

def mse_loss(pred: torch.Tensor, target: torch.Tensor):
    return ((pred - target)**2).mean()
