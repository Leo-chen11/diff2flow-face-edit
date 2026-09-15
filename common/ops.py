import collections.abc as container_abcs

from PIL import Image
import torch
import torch.distributed as dist
from torch import nn

def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def turn_on_spectral_norm(module):
    module_output = module
    # if isinstance(module, torch.nn.Conv2d):
    #     if module.out_channels != 1 and module.in_channels > 4:
    #         module_output = nn.utils.spectral_norm(module)
    # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #     module_output = nn.utils.spectral_norm(module)
    for name, child in module.named_children():
        module_output.add_module(name, turn_on_spectral_norm(child))
    del module
    return module_output


def normalize(input, mean, std):
    mean = torch.Tensor(mean).to(input.device)
    std = torch.Tensor(std).to(input.device)
    return input.sub(mean[None, :, None, None]).div(std[None, :, None, None])


# from https://github.com/NVlabs/DG-Net/blob/0abf564a853ea6ec3f38ab71a4a69f7f23b19d24/networks.py#L155
# regularize real grad


def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda(non_blocking=True)
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_to_cuda(d) for d in data))
    else:
        return data


def label2onehot(labels, num_class):
    code = torch.eye(num_class)[labels.long().squeeze()]
    if len(code.size()) > 1:
        return code
    return code.unsqueeze(0).to(labels)


def label2map(labels, num_class, size):
    return onehot2map(label2onehot(labels, num_class), size).to(labels)


def onehot2map(onehots, size):
    return onehots.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, size, size)


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is not None:
        rt /= world_size
    return rt


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`  #du: namekey = k[7:]
        new_state_dict[namekey] = v
    return new_state_dict


# from common.nn.insightface import iresnet50
# preprocess for insightface image input

# initial all parameters to zero
