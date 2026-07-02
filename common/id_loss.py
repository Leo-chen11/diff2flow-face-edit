import torch
import torch.nn as nn
import torch.nn.functional as F


class IDLoss(nn.Module):
    def __init__(self, crop=True, backbone='r50'):
        super(IDLoss, self).__init__()
        try:
            from common.nn.insightface import iresnet34, iresnet50, iresnet100
            if backbone == 'r50':
                self.facenet = iresnet50(pretrained=True)
            elif backbone == 'r100':
                self.facenet = iresnet100(pretrained=True)
            else:
                self.facenet = iresnet34(pretrained=True)
            self.input_size = 112
        except (ImportError, ModuleNotFoundError):
            from facenet_pytorch import InceptionResnetV1
            self.facenet = InceptionResnetV1(pretrained='vggface2')
            self.input_size = 160

        self.facenet.eval()
        self.crop = crop
        self.embeddings = None

    @torch.no_grad()
    def extract_dataset(self, loader):
        embeddings = []
        for inputs in loader:
            images = inputs[0].cuda()
            embeddings.append(F.normalize(self.extract_features(images), dim=1))
        self.embeddings = torch.cat(embeddings, dim=0)

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
