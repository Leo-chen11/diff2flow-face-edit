import copy

import torch
import torch.nn as nn
import torchvision


def mlp(in_c, hid_c, out_c, n_layer=0):
    layers = [nn.Linear(in_c, hid_c),
              nn.BatchNorm1d(hid_c),
              nn.ReLU(True)]
    for _ in range(n_layer):
        layers += [
            nn.Linear(hid_c, hid_c),
            nn.BatchNorm1d(hid_c),
            nn.ReLU(True)
        ]
    layers.append(nn.Linear(hid_c, out_c))
    return nn.Sequential(*layers)


def _convnext_tiny(pretrained):
    """torchvision ConvNeXt-T with its classification layer removed (768-d
    pooled features). ImageNet weights only when training from scratch; a
    saved checkpoint overwrites them anyway."""
    try:
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = torchvision.models.convnext_tiny(weights=weights)
    except AttributeError:   # torchvision < 0.13
        net = torchvision.models.convnext_tiny(pretrained=pretrained)
    net.classifier[-1] = nn.Identity()
    return net


class AttributeClassifier(nn.Module):
    """40 CelebA attribute heads (+ an age head) on a backbone.

    r18 / r34 / r50: the original layout (the age head is a second copy of
    the backbone). convnext_t: a different architecture family, for a
    training teacher that does not share the ResNet teacher's blind spots;
    its age head is a linear layer on the shared features, since nothing
    uses the age output."""

    def __init__(self, backbone='r34', pretrained=True):
        super(AttributeClassifier, self).__init__()
        import torchvision
        self.age_from_features = False
        if backbone == 'convnext_t':
            self.extractor = _convnext_tiny(pretrained)
            in_dim = 768
            self.attr_heads = nn.ModuleList([mlp(in_dim, in_dim, 1, 2) for _ in range(40)])
            self.age_heads = nn.Linear(in_dim, 6)
            self.age_from_features = True
            return
        if backbone == 'r34':
            backbone = torchvision.models.resnet34(pretrained=pretrained)
            in_dim = 512
        elif backbone == 'r50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            in_dim = 2048
        elif backbone == 'r18':
            backbone = torchvision.models.resnet18(pretrained=pretrained)
            in_dim = 512
        else:
            raise NotImplementedError
        self.extractor = copy.deepcopy(backbone)
        self.extractor.fc = nn.Identity()

        self.attr_heads = nn.ModuleList([mlp(in_dim, in_dim, 1, 2) for _ in range(40)])

        self.age_heads = copy.deepcopy(backbone)
        self.age_heads.fc = nn.Linear(in_dim, 6)

    def forward(self, x):
        outs = []

        features = self.extractor(x)
        for i, head in enumerate(self.attr_heads):
            outs.append(head(features))

        outs.append(self.age_heads(features if self.age_from_features else x))
        outs = torch.cat(outs, dim=1)

        preds = (torch.sigmoid(outs) > 0.5).float()
        preds = torch.cat([preds[:, :40], preds[:, 40:].sum(dim=1).unsqueeze(1)], dim=1)
        return outs, preds

    def forward_attr(self, x):
        features = self.extractor(x)
        outs = []
        for i, head in enumerate(self.attr_heads):
            outs.append(head(features))

        outs = torch.cat(outs, dim=1)
        preds = (torch.sigmoid(outs) > 0.5).float()
        return outs, preds

class TeacherEnsemble(nn.Module):
    """Several frozen AttributeClassifiers read as one teacher (evaluation
    side of train_sdflow.py --extra_teacher_weights). forward() returns the
    logit of the MEAN probability in the classifier's (logits, preds) shape,
    so code written for one teacher works unchanged; each() gives every
    teacher's probabilities for rules that need all of them to agree."""

    def __init__(self, teachers, combine='mean'):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.combine = combine

    def each(self, x):
        return torch.stack([torch.sigmoid(t(x)[0][:, :40]) for t in self.teachers])

    def forward(self, x):
        p = self.each(x).mean(0).clamp(1e-6, 1 - 1e-6)
        logits = torch.log(p) - torch.log1p(-p)
        return logits, (p > 0.5).float()


class AttributeEstimator(nn.Module):
    def __init__(self, backbone='resnet34',attribute_dim=40):
        super(AttributeEstimator, self).__init__()
        self.attribute_dim = attribute_dim
        if backbone == 'alexnet':
            model = torchvision.models.alexnet(pretrained=False)
            model.classifier[-1] = nn.Linear(4096, attribute_dim)
        elif backbone == 'vgg16':
            model = torchvision.models.vgg16(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, attribute_dim)
        elif backbone == 'vgg16bn':
            model = torchvision.models.vgg16_bn(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, attribute_dim)
        elif backbone == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        elif backbone == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        elif backbone == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        elif backbone == 'resnet101':
            model = torchvision.models.resnet101(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        elif backbone == 'inceptionv3':
            model = torchvision.models.inception_v3(pretrained=False, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        else:
            raise NotImplementedError('backbone [%s] is not implemented' % backbone)
        self.backbone = model
        self.sigmoid = nn.Sigmoid()
    def forward(self,img,latent=None):
        bs = img.size(0)
        out = self.sigmoid(self.backbone(img))
        out = out.view(bs, self.attribute_dim)
        return out