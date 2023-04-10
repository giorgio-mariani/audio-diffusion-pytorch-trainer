import numpy as np
from scipy import integrate
import torch

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def log_likelihood_song(model, data, sigma_max, hutchinson_type="Gaussian"):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.
    Args:
      model: A score model.
      data: A PyTorch tensor.
    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
        shape = data.shape
        if hutchinson_type == 'Gaussian':
            epsilon = torch.randn_like(data)
        elif hutchinson_type == 'Rademacher':
            epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

        def ode_func(t, x):
            sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
            vec_t = torch.ones(sample.shape[0], device=sample.device) * t
            drift = to_flattened_numpy(drift_fn_karras(model, sample, vec_t))
            logp_grad = to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
            out = np.concatenate([drift, logp_grad], axis=0)
            return out

        init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
        
        solution = integrate.solve_ivp(ode_func, (1e-5, sigma_max), init, rtol=1e-5, atol=1e-5, method="RK45")
        nfe = solution.nfev
        zp = solution.y[:, -1]
        z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
        delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
        prior_logp = torch.distributions.Normal(0, sigma_max).log_prob(z).sum()
        bpd = -(prior_logp + delta_logp) / np.log(2)
        N = np.prod(shape[1:])
        bpd = bpd / N
        return bpd, z, nfe
    
def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn_karras(model, xx, tt))(x, t, noise)

def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
    return div_fn


def drift_fn_karras(model, x, sigma):
    """The drift function of the reverse-time SDE."""
    denoised = model(x, sigma)
    return (x - denoised) / sigma


