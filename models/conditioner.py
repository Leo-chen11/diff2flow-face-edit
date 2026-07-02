import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.attribute_estimator import AttributeEstimator


def _to_01(images):
    return (images + 1.0).mul(0.5).clamp(0.0, 1.0)


def _normalize(images, mean, std):
    mean = images.new_tensor(mean).view(1, 3, 1, 1)
    std = images.new_tensor(std).view(1, 3, 1, 1)
    return (images - mean) / std


def _load_openai_clip(model_name, device):
    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            'CLIP conditioner requires the OpenAI CLIP package. Install it with:\n'
            '  pip install git+https://github.com/openai/CLIP.git\n'
            'or run the training with --conditioner_backbone resnet to keep the original conditioner.'
        ) from exc
    model, _ = clip.load(model_name, device=device, jit=False)
    return model.float()


class FrozenResNetFeature(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        if backbone == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
            self.out_dim = 512
        elif backbone == 'resnet34':
            model = torchvision.models.resnet34(pretrained=True)
            self.out_dim = 512
        elif backbone == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
            self.out_dim = 2048
        elif backbone == 'resnet101':
            model = torchvision.models.resnet101(pretrained=True)
            self.out_dim = 2048
        else:
            raise ValueError(f'Unsupported frozen ResNet backbone: {backbone}')
        model.fc = nn.Identity()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def forward(self, images):
        x = F.interpolate(_to_01(images), (224, 224), mode='bilinear', align_corners=False)
        x = _normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return self.model(x)


class FrozenCLIPFeature(nn.Module):
    def __init__(self, model_name='ViT-B/32'):
        super().__init__()
        # Loaded lazily because clip.load needs the final device.
        self.model_name = model_name
        self.__dict__['_clip_model'] = None
        self.out_dim = 512

    def _ensure_model(self, device):
        if self.__dict__['_clip_model'] is None:
            model = _load_openai_clip(self.model_name, device)
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
            visual_width = getattr(model.visual, 'output_dim', None)
            if visual_width is not None:
                self.out_dim = int(visual_width)
            self.__dict__['_clip_model'] = model

    def forward(self, images):
        self._ensure_model(images.device)
        x = F.interpolate(_to_01(images), (224, 224), mode='bicubic', align_corners=False)
        x = _normalize(
            x,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        return self.__dict__['_clip_model'].encode_image(x).float()


class FrozenFeatureAttributeConditioner(nn.Module):
    """Frozen visual feature extractor plus trainable projection to attr_cond."""

    def __init__(self, attr_dim, mode='clip', resnet_backbone='resnet50',
                 clip_model='ViT-B/32', hidden_dim=256):
        super().__init__()
        self.attr_dim = int(attr_dim)
        self.mode = mode
        if mode == 'clip':
            self.resnet = None
            self.clip = FrozenCLIPFeature(clip_model)
            feat_dim = self.clip.out_dim
        elif mode == 'resnet_clip':
            self.resnet = FrozenResNetFeature(resnet_backbone)
            self.clip = FrozenCLIPFeature(clip_model)
            feat_dim = self.resnet.out_dim + self.clip.out_dim
        else:
            raise ValueError(f'Unsupported frozen feature conditioner mode: {mode}')

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.attr_dim),
        )

    def forward(self, images):
        feats = []
        with torch.no_grad():
            if self.resnet is not None:
                feats.append(F.normalize(self.resnet(images), dim=1))
            feats.append(F.normalize(self.clip(images), dim=1))
            feat = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        return torch.sigmoid(self.projector(feat))


class IdentityAttributeConditioner(nn.Module):
    """Build the SDFlow condition c = [identity condition, attribute condition]."""

    def __init__(self, attr_dim, id_dim=32, attr_backbone='resnet50', id_scale=0.25,
                 id_feature_dim=512, conditioner_backbone='resnet',
                 clip_model='ViT-B/32', fused_hidden_dim=256):
        super(IdentityAttributeConditioner, self).__init__()
        self.attr_dim = int(attr_dim)
        self.id_dim = int(id_dim)
        self.id_scale = float(id_scale)
        self.conditioner_backbone = conditioner_backbone
        self.condition_dim = self.id_dim + self.attr_dim

        if conditioner_backbone == 'resnet':
            self.attr_estimator = AttributeEstimator(
                backbone=attr_backbone,
                attribute_dim=self.attr_dim,
            )
        elif conditioner_backbone in {'clip', 'resnet_clip'}:
            self.attr_estimator = FrozenFeatureAttributeConditioner(
                attr_dim=self.attr_dim,
                mode=conditioner_backbone,
                resnet_backbone=attr_backbone,
                clip_model=clip_model,
                hidden_dim=fused_hidden_dim,
            )
        else:
            raise ValueError(
                f'Unknown conditioner_backbone={conditioner_backbone}. '
                'Use resnet, clip, or resnet_clip.'
            )
        self.id_projector = nn.Sequential(
            nn.Linear(id_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.id_dim),
        )

    def make_condition(self, images, latents=None, id_extractor=None):
        attr_cond = self.attr_estimator(images)
        if isinstance(attr_cond, tuple):
            attr_cond = attr_cond[0]

        if id_extractor is None:
            id_cond = images.new_zeros(images.size(0), self.id_dim)
        else:
            with torch.no_grad():
                id_feat = F.normalize(id_extractor.extract_features(images), dim=1)
            id_cond = self.id_projector(id_feat) * self.id_scale

        cond = torch.cat([id_cond, attr_cond], dim=1)
        return cond, id_cond, attr_cond
