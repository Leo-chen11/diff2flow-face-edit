import torch
import torch.nn as nn
import torch.nn.functional as F


# ── BiSeNet (from facexlib) ────────────────────────────────────────────────

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        shortcut = x if self.downsample is None else self.downsample(x)
        return self.relu(shortcut + residual)


def _make_layer(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for _ in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan))
    return nn.Sequential(*layers)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_layer(64, 64, 2, stride=1)
        self.layer2 = _make_layer(64, 128, 2, stride=2)
        self.layer3 = _make_layer(128, 256, 2, stride=2)
        self.layer4 = _make_layer(256, 512, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8  = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, num_class):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan)
        self.conv_out = nn.Conv2d(mid_chan, num_class, kernel_size=1, bias=False)

    def forward(self, x):
        feat = self.conv(x)
        return self.conv_out(feat), feat


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = self.sigmoid(self.bn_atten(self.conv_atten(F.avg_pool2d(feat, feat.size()[2:]))))
        return feat * atten


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        avg = self.conv_avg(F.avg_pool2d(feat32, feat32.size()[2:]))
        feat32_up = self.conv_head32(self.arm32(feat32) + F.interpolate(avg, feat32.size()[2:], mode='nearest'))
        feat16_up = self.conv_head16(self.arm16(feat16) + F.interpolate(feat32_up, feat16.size()[2:], mode='nearest'))
        return feat8, F.interpolate(feat16_up, feat8.size()[2:], mode='nearest'), feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        feat = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten = self.sigmoid(self.conv2(self.relu(self.conv1(F.avg_pool2d(feat, feat.size()[2:])))))
        return feat * atten + feat


class BiSeNet(nn.Module):
    def __init__(self, num_class=19):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_class)
        self.conv_out16 = BiSeNetOutput(128, 64, num_class)
        self.conv_out32 = BiSeNetOutput(128, 64, num_class)

    def forward(self, x):
        h, w = x.size()[2:]
        feat8, feat_cp8, feat_cp16 = self.cp(x)
        out, _ = self.conv_out(self.ffm(feat8, feat_cp8))
        return F.interpolate(out, (h, w), mode='bilinear', align_corners=True)


# ── FaceParser wrapper ─────────────────────────────────────────────────────
# BiSeNet label map (19 classes):
#   0=background, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
#   6=glasses, 7=l_ear, 8=r_ear, 9=earring, 10=nose, 11=mouth,
#   12=u_lip, 13=l_lip, 14=neck, 15=neck_l, 16=cloth, 17=hair, 18=hat
#
# Face mask = classes 1–14, 17  (skin, features, hair — excluding background/cloth/hat)
FACE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]


class FaceParser(nn.Module):
    def __init__(self, weights_path='./data/parsing_bisenet.pth'):
        super().__init__()
        self.net = BiSeNet(num_class=19)
        state = torch.load(weights_path, map_location='cpu')
        self.net.load_state_dict(state, strict=False)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def get_mask(self, x, blur_sigma=3):
        """
        x: [-1,1] tensor [B,3,H,W]
        returns: soft face mask [B,1,H,W] in [0,1], 1=face region
        """
        h, w = x.shape[2:]
        inp = F.interpolate(x, 512, mode='bilinear', align_corners=False)
        inp = (inp * 0.5 + 0.5 - self.mean) / self.std
        logits = self.net(inp)                        # [B, 19, 512, 512]
        seg = logits.argmax(dim=1)                    # [B, 512, 512]

        mask = torch.zeros_like(seg, dtype=torch.float32)
        for c in FACE_CLASSES:
            mask = mask + (seg == c).float()
        mask = mask.unsqueeze(1)                      # [B,1,512,512]

        if blur_sigma > 0:
            ks = blur_sigma * 6 + 1
            pad = ks // 2
            weight = self._gaussian_kernel(ks, blur_sigma, x.device)
            mask = F.conv2d(mask, weight, padding=pad)

        mask = F.interpolate(mask, (h, w), mode='bilinear', align_corners=False)
        return mask.clamp(0, 1)

    def _gaussian_kernel(self, ks, sigma, device):
        coords = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None] * g[None, :]
        return kernel.view(1, 1, ks, ks)

    def composite(self, orig, edited):
        """
        Blend edited face onto original background.
        orig, edited: [-1,1] tensors [B,3,H,W]
        returns: composited tensor [-1,1]
        """
        mask = self.get_mask(orig)
        return edited * mask + orig * (1 - mask)
