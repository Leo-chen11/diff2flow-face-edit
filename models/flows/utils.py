from math import log, pi
import torch
import torch.distributed as dist


def modify_one_attribute(attributes:torch.Tensor,idx:int=-1,scale:float=None,mode='random'):
    """

    Args:
        attributes (torch.Tensor): 0-1 tensor
        idx (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """    
    attributes = attributes.to(torch.float32)
    bs,columns = attributes.shape[:2]
    new_attributes = attributes.detach().clone()
    
    if mode == 'keep':
        return None,new_attributes
    
    if idx < 0 or idx >= columns:
        # if modified index is not given , randomly generate one !
        if mode=='random':
            idx = torch.randint(0,columns,(bs,))
            new_attributes[torch.arange(bs), idx] = torch.randint(0, 2, (bs,),dtype=torch.float32).to(new_attributes)
        elif mode == 'uniform':
            idx = torch.randint(0,columns,(bs,))
            new_attributes[torch.arange(bs), idx] = torch.rand((bs,),dtype=torch.float32).to(new_attributes)
        elif mode=='negative':
            idx = torch.randint(0,columns,(1,))
            new_attributes[torch.arange(bs), idx] = 1. -new_attributes[torch.arange(bs), idx]
        
    elif scale is not None:
        new_attributes[attributes[:,idx]==1,idx] = 1.0 - scale
        new_attributes[attributes[:,idx]==0,idx] = scale
    else:
        new_attributes[:,idx] = 1 - new_attributes[:,idx]
    
    return idx,new_attributes


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
