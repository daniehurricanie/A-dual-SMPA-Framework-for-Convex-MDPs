
import torch
from contextlib import contextmanager

def _flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])

def _flatten_grads(model):
    grads = []
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                grads.append(torch.zeros_like(p.data).view(-1))
            else:
                grads.append(p.grad.view(-1))
    return torch.cat(grads)

def _set_params_from_vector(model, vec):
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(vec[idx:idx+numel].view_as(p))
            idx += numel

@contextmanager
def preserve_params(model):
    vec = _flatten_params(model).clone()
    try:
        yield
    finally:
        _set_params_from_vector(model, vec)

def armijo_backtracking(loss_fn, model, init_loss, grad_dot_dir, direction_vec, 
                        alpha0=1.0, beta=0.5, c=1e-4, max_steps=20):
    """
    Classic Armijo backtracking line search on a single parameterized model.
    direction_vec is a flat vector with the *descent* direction (e.g., -grad).
    grad_dot_dir should be the directional derivative at current params: g^T d.
    Returns (accepted, alpha, new_loss).
    """
    assert grad_dot_dir < 0, "Direction must be a descent direction (g^T d < 0)."
    with preserve_params(model):
        params0 = _flatten_params(model)
        for i in range(max_steps):
            alpha = (alpha0 * (beta ** i))
            new_params = params0 + alpha * direction_vec
            _set_params_from_vector(model, new_params)
            Lnew = float(loss_fn().detach().cpu())
            if Lnew <= init_loss + c * alpha * grad_dot_dir:
                # keep new params
                return True, alpha, Lnew
        # if not accepted, restore params (preserve_params does this) and return failure
        return False, 0.0, init_loss
