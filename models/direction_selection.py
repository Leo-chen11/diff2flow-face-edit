import torch


def direction_selection_loss(alpha, marginal_weight=0.2):
    """Encourage sharp per-sample selection (low conditional entropy) while keeping
    all K directions used across the batch (high marginal entropy).

    Args:
        alpha: (B, K) softmax mixture weights for the active attribute
        marginal_weight: how much to reward uniform batch-level usage

    Returns:
        loss: scalar — minimize this
        logs: dict of detached diagnostic tensors
    """
    eps = 1e-8
    # Conditional entropy: low = each sample has a clear favourite direction
    cond_ent = -(alpha * (alpha + eps).log()).sum(dim=-1).mean()

    # Marginal entropy: high = all directions get used across the batch
    marginal = alpha.mean(dim=0)                                    # (K,)
    marginal_ent = -(marginal * (marginal + eps).log()).sum()

    loss = cond_ent - marginal_weight * marginal_ent

    logs = {
        'dir_cond_entropy':    cond_ent.detach(),
        'dir_marginal_entropy': marginal_ent.detach(),
        'dir_max_prob':        alpha.max(dim=-1).values.mean().detach(),
    }
    return loss, logs


def direction_usage_stats(alpha):
    """Per-direction usage statistics for WandB logging.

    Args:
        alpha: (B, K) softmax mixture weights

    Returns:
        logs: dict — dir_usage_k (mean weight) and dir_top1_ratio_k (fraction of samples
              that rank direction k first)
    """
    logs = {}
    mean_usage = alpha.mean(dim=0)                  # (K,)
    top1 = alpha.argmax(dim=-1)                      # (B,)
    K = alpha.shape[1]
    for k in range(K):
        logs[f'dir_usage_{k}']      = mean_usage[k].detach()
        logs[f'dir_top1_ratio_{k}'] = (top1 == k).float().mean().detach()
    return logs
