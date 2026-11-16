"""DeltaEdit (Lyu et al., "DeltaEdit: Exploring Text-free Training for Text-Driven Image Manipulation",
CVPR 2023) as a baseline, run with the OFFICIAL released face model (github.com/Yueming6568/DeltaEdit,
checkpoints/net_face.pth). The official repository has no licence file, so this is an independent
implementation of its inference (scripts/inference_real.py) with the same module and parameter names:
the released checkpoint loads strictly.

The method: a DeltaMapper predicts a change of the StyleSpace code (the 6048 per-channel styles of the
17 convolutions of StyleGAN2-FFHQ) from that code and a CLIP condition [image feature of the source,
normalised text direction]; the text direction is E(target prompt) - E(neutral prompt), each the mean
over prompt templates. Channels whose relevance to the direction (StyleCLIP's matrix fs3 . direction)
is below a threshold are left unchanged (the official 0.03).

Used here:
  * the source's StyleSpace code is computed from this repo's e4e W+ latent, with this repo's
    generator (the same rosinality StyleGAN2-FFHQ weights); toRGB styles are not edited (official)
  * the CLIP image feature is read from the real photo, as the official real-image script does
    (resized to 256, then its x7 upsample + 8x8 average pool to 224, fed in [-1, 1] without CLIP's
    own normalisation -- how the released model was trained)
  * prompts (PROMPTS): the official style "face with ..." against the neutral "face". A removal has
    no official recipe: Male / Young use the opposite prompt ("female face", "old face"); the others
    use the negated direction of their "add" prompt (removal='negate' negates for all five)
  * strength: the predicted change is multiplied by --scales (1 = the official edit)
The released model was trained on 58,000 FFHQ photos chosen at random by its authors (plus sampled
faces), so some of this repo's test faces were probably among its training inputs.

Files (not in this repo): <dir>/net_face.pth (the official Google Drive link in its README) and
<dir>/fs3.npy (models/stylegan2/npy_ffhq/fs3.npy of the official repository).
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.stylegan2.model import PixelNorm

NEUTRAL = 'face'
# attr: (prompt that adds it, prompt that removes it or None = negate the add direction)
PROMPTS = {
    15: ('face with eyeglasses', None),
    20: ('male face', 'female face'),
    39: ('young face', 'old face'),
    31: ('face with smile', None),
    5: ('face with bangs', None),
}
# The prompt templates of the official utils/map_tool.py (OpenAI CLIP's ImageNet templates, as in
# StyleCLIP's global directions, without 'a photo of many {}.').
TEMPLATES = [
    'a bad photo of a {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.',
    'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.',
    'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
    'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
    'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
    'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.',
    'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
    'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.',
    'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.',
    'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
    'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.',
    'a photo of a small {}.', 'a tattoo of the {}.',
]
COARSE, MEDIUM = 3 * 512, 7 * 512          # StyleSpace split: 3 x 512 | 4 x 512 | the remaining 2464
S_DIM = 6048


class EqualLinear(nn.Module):
    """StyleGAN2's equalised linear layer followed by its fused leaky ReLU (slope 0.2, gain sqrt 2),
    with the parameter names of the generator's own layer. The bias is added on the LAST dimension, so
    the layer also takes the (B, layers, 512) inputs of the coarse and medium mappers -- the official
    repository changes its fused_leaky_relu for exactly that; the generator's version (and its
    compiled CUDA kernel) puts the bias on dimension 1."""

    def __init__(self, in_dim, out_dim, lr_mul=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale) + self.bias * self.lr_mul
        return F.leaky_relu(out, negative_slope=0.2) * math.sqrt(2)


class Mapper(nn.Module):
    def __init__(self, in_channel=512, out_channel=512, norm=True, num_layers=4):
        super().__init__()
        layers = [PixelNorm()] if norm else []
        layers.append(EqualLinear(in_channel, out_channel))
        for _ in range(num_layers - 1):
            layers.append(EqualLinear(out_channel, out_channel))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)


class DeltaMapper(nn.Module):
    """Style module (sm_*), condition module (cm_*) and fusion module (fm_*), each at three levels."""

    def __init__(self):
        super().__init__()
        self.sm_coarse = Mapper(512, 512)
        self.sm_medium = Mapper(512, 512)
        self.sm_fine = Mapper(2464, 2464)
        self.cm_coarse = Mapper(1024, 512)
        self.cm_medium = Mapper(1024, 512)
        self.cm_fine = Mapper(1024, 2464)
        self.fm_coarse = Mapper(1024, 512, norm=False)
        self.fm_medium = Mapper(1024, 512, norm=False)
        self.fm_fine = Mapper(4928, 2464, norm=False)

    def forward(self, s, c):
        B = s.size(0)
        s_c = self.sm_coarse(s[:, :COARSE].view(B, 3, 512))
        s_m = self.sm_medium(s[:, COARSE:MEDIUM].view(B, 4, 512))
        s_f = self.sm_fine(s[:, MEDIUM:])
        c_c, c_m, c_f = self.cm_coarse(c), self.cm_medium(c), self.cm_fine(c)
        x_c = self.fm_coarse(torch.cat([s_c, c_c[:, None].expand(-1, 3, -1)], dim=2)).reshape(B, COARSE)
        x_m = self.fm_medium(torch.cat([s_m, c_m[:, None].expand(-1, 4, -1)], dim=2)).reshape(B, MEDIUM - COARSE)
        x_f = self.fm_fine(torch.cat([s_f, c_f], dim=1))
        return torch.cat([x_c, x_m, x_f], dim=1)


def style_convs(G):
    """The 17 modulated convolutions whose styles make up the StyleSpace code, with the W+ row each reads."""
    return [(G.conv1.conv, 0)] + [(c.conv, i + 1) for i, c in enumerate(G.convs)]


def style_code(G, latent):
    """W+ (B, 18, 512) -> StyleSpace code (B, 6048)."""
    return torch.cat([conv.modulation(latent[:, row]) for conv, row in style_convs(G)], dim=1)


def synthesize(G, latent, delta_s=None):
    """G(latent) with delta_s (B, 6048) added to the styles of the 17 convolutions (toRGB untouched)."""
    hooks = []
    if delta_s is not None:
        start = 0
        for conv, _ in style_convs(G):
            piece = delta_s[:, start:start + conv.in_channel]
            start += conv.in_channel
            hooks.append(conv.modulation.register_forward_hook(lambda m, i, out, piece=piece: out + piece))
        assert start == delta_s.size(1), (start, delta_s.shape)
    try:
        return G([latent], input_is_latent=True, randomize_noise=False)[0]
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def text_feature(clip_model, prompt, device):
    """Mean over the templates of the normalised text embeddings, normalised (official zeroshot_classifier)."""
    import clip
    tokens = clip.tokenize([t.format(prompt) for t in TEMPLATES]).to(device)
    e = clip_model.encode_text(tokens).float()
    e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
    return e / e.norm()


class DeltaEdit:
    def __init__(self, deltaedit_dir, attrs, device='cuda', threshold=0.03, removal='auto', clip_model=None):
        ckpt, fs3 = os.path.join(deltaedit_dir, 'net_face.pth'), os.path.join(deltaedit_dir, 'fs3.npy')
        for path, where in ((ckpt, 'the pretrained DeltaMapper link of the official README'),
                            (fs3, 'models/stylegan2/npy_ffhq/fs3.npy of the official repository')):
            if not os.path.exists(path):
                raise SystemExit(f'no {path}: it is {where}')
        self.device = device
        self.net = DeltaMapper()
        self.net.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        self.net.to(device).eval()
        if clip_model is None:
            import clip
            clip_model, _ = clip.load('ViT-B/32', device=device, jit=False)
        self.clip = clip_model.eval()
        relevance = torch.from_numpy(np.load(fs3).astype(np.float32)).to(device)        # (6048, 512)
        assert relevance.shape == (S_DIM, 512), relevance.shape
        bad = [a for a in attrs if a not in PROMPTS]
        if bad:
            raise SystemExit(f'no DeltaEdit prompt for attributes {bad} (models/deltaedit.py PROMPTS)')
        neutral = text_feature(self.clip, NEUTRAL, device)
        self.dt, self.keep = {}, {}                 # (attr, +1 / -1) -> unit direction, channels kept
        for a in attrs:
            add, rm = PROMPTS[a]
            d_add = text_feature(self.clip, add, device) - neutral
            d_rm = -d_add if (rm is None or removal == 'negate') else text_feature(self.clip, rm, device) - neutral
            for sign, d in ((1, d_add), (-1, d_rm)):
                self.keep[(a, sign)] = ((relevance @ d).abs() >= threshold).float()
                self.dt[(a, sign)] = d / d.norm().clamp(min=1e-5)
            how = 'negated add direction' if (rm is None or removal == 'negate') else repr(rm)
            print(f'[DeltaEdit] attribute {a}: add {add!r} ({int(self.keep[(a, 1)].sum())} channels), '
                  f'remove {how} ({int(self.keep[(a, -1)].sum())} channels)')

    @torch.no_grad()
    def image_feature(self, img):
        """CLIP feature of the real photos (B, 3, H, W) in [-1, 1], the official preprocessing."""
        x = F.interpolate(img.float(), size=(256, 256), mode='bilinear', align_corners=False, antialias=True)             if img.shape[-1] != 256 else img.float()
        x = F.avg_pool2d(F.interpolate(x, scale_factor=7, mode='nearest'), kernel_size=8)
        c = self.clip.encode_image(x.to(self.clip.visual.conv1.weight.dtype)).float()
        return c / c.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def edit(self, G, latent, c_img, attr, direction, strength):
        """Edited faces for W+ latents (B, 18, 512); direction (B,) is +1 (add) or -1 (remove)."""
        s = style_code(G, latent)
        out = torch.zeros_like(s)
        for sign in (1, -1):
            pick = (direction > 0) if sign > 0 else (direction <= 0)
            if not pick.any():
                continue
            cond = torch.cat([c_img[pick], self.dt[(attr, sign)].expand(int(pick.sum()), -1)], dim=1)
            out[pick] = self.net(s[pick], cond) * self.keep[(attr, sign)]
        return synthesize(G, latent, out * float(strength)).clamp(-1, 1)
