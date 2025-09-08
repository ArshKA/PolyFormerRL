import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

def process_gmm_params(mus, log_sigmas, logits_pi):
    """Transform raw network outputs into valid GMM parameters.

    Args:
        mus (Tensor): Mean for each component with shape ``[B, K, D]``.
        log_sigmas (Tensor): Log standard deviations with shape ``[B, K, D]``.
        logits_pi (Tensor): Mixture logits with shape ``[B, K]``.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: ``(weights, mu, sigma)`` where ``weights``
        sums to 1 over components, ``mu`` are the means and ``sigma`` are the
        standard deviations.
    """
    w = F.softmax(logits_pi, dim=-1)
    mu = mus
    sigma = torch.exp(log_sigmas)
    return w, mu, sigma

def gmm_sample(w, mu, sigma):
    """Sample from a Gaussian mixture distribution.

    Args:
        w (Tensor): Mixture weights of shape ``[B, K]``.
        mu (Tensor): Means of shape ``[B, K, D]``.
        sigma (Tensor): Standard deviations of shape ``[B, K, D]``.

    Returns:
        Tuple[Tensor, Tensor]: ``(action, component_idx)`` where ``action`` has
        shape ``[B, D]`` and ``component_idx`` the selected component per batch
        element with shape ``[B]``.
    """
    cat = Categorical(w)
    component_idx = cat.sample()
    batch_indices = torch.arange(w.size(0), device=w.device)
    mu_c = mu[batch_indices, component_idx]
    sigma_c = sigma[batch_indices, component_idx]
    action = Normal(mu_c, sigma_c).sample()
    return action, component_idx

def gmm_log_prob(action, w, mu, sigma, component_idx=None):
    """Compute log-probability of ``action`` under the GMM.

    Args:
        action (Tensor): Action tensor of shape ``[B, D]``.
        w (Tensor): Mixture weights ``[B, K]``.
        mu (Tensor): Means ``[B, K, D]``.
        sigma (Tensor): Standard deviations ``[B, K, D]``.
        component_idx (Tensor, optional): If provided, compute the log
            probability using only the selected component. Shape ``[B]``.

    Returns:
        Tensor: Log-probabilities with shape ``[B]``.
    """
    batch_size, num_components, dim = mu.shape
    if component_idx is not None:
        batch_indices = torch.arange(batch_size, device=mu.device)
        log_w = torch.log(w[batch_indices, component_idx])
        mu_c = mu[batch_indices, component_idx]
        sigma_c = sigma[batch_indices, component_idx]
        log_prob = Normal(mu_c, sigma_c).log_prob(action).sum(-1)
        return log_w + log_prob
    # General case using log-sum-exp across components
    log_w = torch.log(w)
    action = action.unsqueeze(1).expand(-1, num_components, -1)
    comp_log_prob = Normal(mu, sigma).log_prob(action).sum(-1)
    return torch.logsumexp(log_w + comp_log_prob, dim=1)
