"""ArcFace identity model (Deng et al., CVPR 2019) for the identity-preservation
metric: the IR-SE50 network of TreB1eN/InsightFace_Pytorch, as used for the ID
loss of pSp / e4e. The weights are model_ir_se50.pth from those projects'
READMEs; nothing here downloads them.

common/id_loss.py (the training ID loss, called id_arc in the eval output) is
NOT this model: it is facenet-pytorch InceptionResnetV1 (VGGFace2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear, PReLU, Sequential

from models.e4e.helpers import Flatten, bottleneck_IR_SE, get_blocks, l2_norm


class Backbone(nn.Module):
    """IR-SE50, 112 x 112 input, 512-d L2-normalised embedding."""

    def __init__(self, drop_ratio=0.4):
        super().__init__()
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512), Dropout(drop_ratio), Flatten(),
                                       Linear(512 * 7 * 7, 512), BatchNorm1d(512, affine=True))
        self.body = Sequential(*[bottleneck_IR_SE(b.in_channel, b.depth, b.stride)
                                 for block in get_blocks(50) for b in block])

    def forward(self, x):
        return l2_norm(self.output_layer(self.body(self.input_layer(x))))


class ArcFaceJudge(nn.Module):
    """extract(x): x in [-1, 1], (B, 3, 256, 256) -> (B, 512) unit vectors; the
    cosine similarity of two faces is the dot product. Crop and pooling as in
    pSp's IDLoss: rows 35:223, columns 32:220 of the 256 image, average-pooled
    to 112."""

    def __init__(self, weights_path, device='cuda'):
        super().__init__()
        self.net = Backbone()
        state = torch.load(weights_path, map_location='cpu')
        state = state.get('state_dict', state) if isinstance(state, dict) else state
        state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
        res = self.net.load_state_dict(state, strict=False)
        if res.missing_keys:
            raise RuntimeError(f'{weights_path} is not an IR-SE50 ArcFace checkpoint '
                               f'(missing {res.missing_keys[:4]} ...)')
        self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.pool = nn.AdaptiveAvgPool2d((112, 112))

    @torch.no_grad()
    def extract(self, x):
        if x.shape[-1] != 256:
            x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        return self.net(self.pool(x[:, :, 35:223, 32:220]))
