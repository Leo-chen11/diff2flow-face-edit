import torch
import torch.nn as nn
import torch.nn.functional as F


class IDLoss(nn.Module):
    def __init__(self, crop=True):
        super(IDLoss, self).__init__()
        # facenet-pytorch InceptionResnetV1 (VGGFace2). The insightface branch that
        # used to come first imported a module this repo never had, so this was
        # always the model in use.
        from facenet_pytorch import InceptionResnetV1
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        self.input_size = 160

        self.facenet.eval()
        self.crop = crop

    def extract_features(self, x):
        # x: [-1, 1] tensor [B, 3, H, W]
        if self.crop:
            w = x.size(-1)
            scale = lambda v: int(v * w / 256)
            crop_h, x1, x2 = scale(188), scale(35), scale(32)
            x = x[:, :, x1:x1 + crop_h, x2:x2 + crop_h]
        if x.size(-1) != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        return self.facenet(x)

    def forward(self, input, recon):
        # input, recon: [-1, 1] tensors
        with torch.no_grad():
            e1 = F.normalize(self.extract_features(input), dim=1)
        e2 = F.normalize(self.extract_features(recon), dim=1)
        return -(e1 * e2).sum(dim=1).mean()
