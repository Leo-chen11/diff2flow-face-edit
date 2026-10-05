from math import log, pi
import torch
import torch.distributed as dist


def modify_one_attribute(attributes: torch.Tensor, mode='negative'):
    """Pick one attribute column at random (the same for the whole batch) and
    flip it. Returns (idx of shape (1,), flipped 0-1 attributes). Only the
    'negative' mode training uses is left."""
    if mode != 'negative':
        raise ValueError(f"modify_one_attribute: only mode='negative' is supported, got {mode!r}")
    attributes = attributes.to(torch.float32)
    bs, columns = attributes.shape[:2]
    new_attributes = attributes.detach().clone()
    idx = torch.randint(0, columns, (1,))
    new_attributes[torch.arange(bs), idx] = 1. - new_attributes[torch.arange(bs), idx]
    return idx, new_attributes


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2
