"""Content condition for the ControlNet branch (the half of DC-ControlNet this
project never had).

WHAT WAS MISSING. DC-ControlNet conditions each element on two things: its
LAYOUT (where it is) and its CONTENT (what it should look like). This
project's adaptation only ever built the layout half -- --controlnet_region_cond
feeds a region mask into AttributeControlEncoder. Nothing told the encoder what
the edited region should LOOK like. All it had was a scalar attr_delta plus the
source latent, so for aging it had to invent "old skin" from nothing. Every
long run converged on the cheapest thing that moves the scores (a colour
shift, blotchy pale skin, flat grey hair) instead of real aging texture. Male
sources fared worst (CLIP fail 32% vs 2.2% for female).

WHAT CONTENT IS HERE. A texture description of a REAL face that is already
in the target state: for an aging (Young rm) edit, a real old face of the
same gender; for a rejuvenating (Young add) edit, a real young face of the
same gender. It is never the same person as the source.

The description is the per-channel mean/std of frozen VGG16 features
(relu2_2, relu3_3), pooled separately inside the BiSeNet skin mask and the
BiSeNet hair mask. This is the statistic style transfer uses to describe
texture (AdaIN / Gram). It separates wrinkles from pores from spots from
stubble, which the single high-frequency-energy number behind
--age_skin_hf_loss_weight cannot. Spatial pooling throws away layout, so
the reference's pose and face shape do not transfer. Brightness is removed
before VGG (luminance only, region mean subtracted), so the reference's skin
tone and lighting do not transfer either. Only texture is left.

Reference statistics are measured on G(w_ref), the StyleGAN reconstruction
of the reference, not the raw photo. The edited output also comes out of G,
so the target is something G can actually produce. Real-photo sensor noise
and JPEG texture are not reachable targets and would teach the model to
paint noise.

WHY INPUT *AND* LOSS. Content given only as an input is ignored: no existing
loss gets smaller when the output resembles a random other person. So the
same statistic is also a loss target (see content_match_loss): the edited
region's texture must move toward the reference's, by as much as the edit
strength asks. The input tells the encoder WHICH reference to match. Without
it, matching a different random reference every step can only be satisfied
on average, which is exactly the washed-out look this is meant to fix.

Pieces:
  RegionTextureStats  frozen VGG16 -> (B, D) masked mean/std + per-region validity
  ContentBank         precomputed reference stats (scripts/precompute_content_bank.py)
  ContentEncoder      (B, D) z-scored stats -> (B, hidden) additive bias into
                      AttributeControlEncoder's trunk; last layer zero-init, so
                      step 0 is bit-identical to a run without it
  ContentContext      eval-side bundle: picks a reference and returns the bias
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# BiSeNet labels (see common/face_parser.py): 1=skin, 17=hair.
CONTENT_SKIN_CLASSES = [1]
CONTENT_HAIR_CLASSES = [17]
CONTENT_REGIONS = ('skin', 'hair')

# The only attribute content is wired for. Male/Eyeglasses would each need their
# own reference buckets and region choice; one variable at a time.
CONTENT_ATTRS = (39,)

# Bank bucket ids: which target state a reference shows.
BUCKET_OLD = 0     # target of a Young rm edit (source young -> make old)
BUCKET_YOUNG = 1   # target of a Young add edit (source old -> make young)

# torchvision vgg16.features indices whose OUTPUT is the tapped ReLU.
_VGG_TAPS = {'relu2_2': 8, 'relu3_3': 15}
_VGG_CHANNELS = {'relu2_2': 128, 'relu3_3': 256}


def _load_vgg16_features():
    import torchvision
    try:
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        vgg = torchvision.models.vgg16(weights=weights)
    except AttributeError:   # torchvision < 0.13
        vgg = torchvision.models.vgg16(pretrained=True)
    return vgg.features[:max(_VGG_TAPS.values()) + 1]


class RegionTextureStats(nn.Module):
    """Frozen VGG16 texture statistics inside BiSeNet regions.

    stats(img, masks) -> (B, D) with D = 2 regions * 2 (mean, std) * (128 + 256)
    = 1536, and valid (B, 2): False where a region covers < min_frac of the
    image. A bald head has no hair statistics; those entries are zeroed and
    left out of the loss, never compared against a real reference.

    Differentiable w.r.t. img (VGG weights are frozen, the input is not), which
    is what the content loss needs. Masks are treated as constants.
    """

    def __init__(self, res=256, min_frac=0.01, erode=4):
        super().__init__()
        self.res = int(res)
        self.min_frac = float(min_frac)
        # Pool over an ERODED mask: features within a few pixels of the region
        # boundary mostly see the boundary edge itself (hair/skin/background
        # transition), which is not the texture being described.
        self.erode = int(erode)
        self.features = _load_vgg16_features().eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.per_region_dim = 2 * sum(_VGG_CHANNELS.values())
        self.dim = self.per_region_dim * len(CONTENT_REGIONS)

    def train(self, mode=True):
        # Always frozen / eval, whatever the caller's module tree does.
        return super().train(False)

    def _vgg_taps(self, x):
        # Each tap is a ReLU output followed by MaxPool/end, so torchvision's
        # inplace ReLUs never overwrite a tapped tensor.
        out = {}
        stop = max(_VGG_TAPS.values())
        inv = {v: k for k, v in _VGG_TAPS.items()}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in inv:
                out[inv[i]] = x
            if i >= stop:
                break
        return out

    def _region_stats(self, img, mask):
        """img (B,3,res,res) in [-1,1]; mask (B,1,res,res) in [0,1] (already eroded)."""
        # Luminance only, with the REGION mean removed: skin tone and lighting
        # are not texture, and must not be copied from another person.
        y = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]) * 0.5 + 0.5
        area = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        y_mean = (y * mask).sum(dim=(2, 3), keepdim=True) / area
        y = (y - y_mean + 0.5).expand(-1, 3, -1, -1)
        x = (y - self.mean) / self.std
        taps = self._vgg_taps(x)
        parts = []
        for name in _VGG_TAPS:
            f = taps[name]
            m = F.interpolate(mask, f.shape[-2:], mode='area')
            a = m.sum(dim=(2, 3)).clamp(min=1e-3)                    # (B,1)
            mu = (f * m).sum(dim=(2, 3)) / a                         # (B,C)
            var = ((f - mu[..., None, None]).pow(2) * m).sum(dim=(2, 3)) / a
            sd = (var + 1e-6).sqrt()
            parts += [mu, sd]
        return torch.cat(parts, dim=1)

    def region_masks(self, face_parser, img):
        """BiSeNet skin / hair masks for img ([-1,1], any size), hard, at self.res,
        eroded. Returns {'skin': (B,1,r,r), 'hair': ...}. No gradient."""
        with torch.no_grad():
            x = F.interpolate(img, (self.res, self.res), mode='bilinear', align_corners=False)
            out = {}
            for name, classes in (('skin', CONTENT_SKIN_CLASSES), ('hair', CONTENT_HAIR_CLASSES)):
                m = face_parser.get_region_mask(x, classes, blur_sigma=0)
                m = (m > 0.5).float()
                if self.erode > 0:
                    k = 2 * self.erode + 1
                    m = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=self.erode)
                out[name] = m
        return out

    def stats(self, img, masks):
        """img: (B,3,H,W) in [-1,1]; masks from region_masks().
        Returns (stats (B, D), valid (B, 2) bool)."""
        x = F.interpolate(img, (self.res, self.res), mode='bilinear', align_corners=False)
        feats, valid = [], []
        for name in CONTENT_REGIONS:
            m = masks[name].to(dtype=x.dtype)
            ok = m.mean(dim=(1, 2, 3)) >= self.min_frac
            s = self._region_stats(x, m)
            feats.append(s * ok[:, None].to(s.dtype))
            valid.append(ok)
        return torch.cat(feats, dim=1), torch.stack(valid, dim=1)


class ContentBank:
    """Precomputed reference statistics (see scripts/precompute_content_bank.py).

    File layout (torch.save dict):
      stats (N, D) float    raw RegionTextureStats output on G(w_ref)
      valid (N, 2) bool     per-region validity (skin, hair)
      male  (N,)   bool     reference gender (dataset Male pred >= 0.5)
      bucket (N,)  long     BUCKET_OLD / BUCKET_YOUNG
      files        list     dataset paths, to never pick the source itself
      dim_mean, dim_std (D) z-score statistics over valid entries
      config       dict     res, erode, min_frac, thresholds (must match training)
    """

    def __init__(self, path, device='cpu'):
        d = torch.load(path, map_location='cpu')
        self.path = path
        self.stats = d['stats'].float().to(device)
        self.valid = d['valid'].bool().to(device)
        self.male = d['male'].bool()
        self.bucket = d['bucket'].long()
        self.files = list(d['files'])
        self.file_to_row = {f: i for i, f in enumerate(self.files)}
        self.dim_mean = d['dim_mean'].float().to(device)
        self.dim_std = d['dim_std'].float().clamp(min=1e-4).to(device)
        self.config = dict(d.get('config', {}))
        self.dim = int(self.stats.shape[1])
        self.num_regions = int(self.valid.shape[1])
        self.region_dim = self.dim // self.num_regions
        # Candidate lists per (bucket, male). Fallback to the bucket's other
        # gender only if one is empty (reported by summary()).
        self.pools = {}
        for b in (BUCKET_OLD, BUCKET_YOUNG):
            for g in (True, False):
                idx = ((self.bucket == b) & (self.male == g)).nonzero().view(-1)
                self.pools[(b, g)] = idx

    def summary(self):
        parts = []
        for (b, g), idx in sorted(self.pools.items()):
            parts.append(f'{"old" if b == BUCKET_OLD else "young"}/{"M" if g else "F"}={idx.numel()}')
        return ', '.join(parts)

    def zscore(self, stats, valid):
        """Raw stats -> z-scored, invalid regions zeroed."""
        z = (stats - self.dim_mean) / self.dim_std
        mask = valid.repeat_interleave(self.region_dim, dim=1).to(z.dtype)
        return z * mask

    def sample(self, want_old, male, exclude_files=None, generator=None):
        """want_old, male: (B,) bool (male may be None -> any gender).
        Returns row indices (B,) long. Never returns a row whose file is in
        exclude_files[b] (the source itself)."""
        B = int(want_old.numel())
        rows = []
        want_old = want_old.view(-1).cpu().tolist()
        male = male.view(-1).cpu().tolist() if male is not None else [None] * B
        for b in range(B):
            bucket = BUCKET_OLD if want_old[b] else BUCKET_YOUNG
            if male[b] is None:
                pool = torch.cat([self.pools[(bucket, True)], self.pools[(bucket, False)]])
            else:
                pool = self.pools[(bucket, bool(male[b]))]
                if pool.numel() == 0:
                    pool = self.pools[(bucket, not bool(male[b]))]
            if pool.numel() == 0:
                raise RuntimeError(f'content bank {self.path} has no entries for bucket {bucket}')
            skip = None
            if exclude_files is not None and exclude_files[b] is not None:
                skip = self.file_to_row.get(exclude_files[b])
            for _ in range(8):
                r = int(pool[int(torch.randint(pool.numel(), (1,), generator=generator))])
                if r != skip:
                    break
            rows.append(r)
        return torch.tensor(rows, dtype=torch.long)

    def lookup(self, rows, device):
        rows = rows.to(self.stats.device)
        return self.stats[rows].to(device), self.valid[rows].to(device)


class ContentEncoder(nn.Module):
    """z-scored reference stats (B, D) -> additive bias (B, hidden_dim) on
    AttributeControlEncoder's trunk pre-activation (see its forward(),
    content_bias).

    Additive on an existing layer instead of widening that layer's input: the
    control_encoder checkpoint keeps exactly the same keys and shapes, so a run
    can --resume_dir from any earlier checkpoint. Only this module starts
    fresh, and its last layer is zero-init, so the resumed model's output at
    step 0 is unchanged.

    No ReLU after the zero-init layer: ReLU'(0) = 0 in PyTorch would make it a
    dead layer that never leaves zero.

    present (B,) zeroes the bias per sample: non-age edits, and age edits hit
    by --content_cond_dropout. Absent content gives exactly the no-content
    behaviour.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, content_z, present):
        return self.net(content_z) * present.view(-1, 1).to(content_z.dtype)


def content_match_loss(edit_z, edit_valid, target_z, target_valid, region_dim):
    """Smooth-L1 in z-scored stat space, averaged over the regions valid in
    BOTH the edit and the target (a region missing on either side is not
    compared). Returns (loss scalar, number of compared regions)."""
    both = (edit_valid & target_valid)                                   # (B, R)
    m = both.repeat_interleave(region_dim, dim=1).to(edit_z.dtype)       # (B, D)
    per = F.smooth_l1_loss(edit_z, target_z, reduction='none', beta=1.0) * m
    n = both.sum()
    if n == 0:
        return edit_z.new_zeros([]), 0
    return per.sum() / (n * region_dim), int(n)


class ContentContext:
    """Eval-side bundle attached to a control_encoder as .content_ctx by
    evaluation/evaluate_sdflow.py load_models(). edit_single_attribute /
    edit_multi_attribute ask it for a bias, so every script that edits through
    those functions (render_preview, the probes, ...) gets content with no
    change on their side.

    References are drawn with a fixed-seed generator, so repeated evals of the
    same checkpoint see the same references in the same order."""

    def __init__(self, bank, encoder, male_local_idx, seed=0, enabled=True):
        self.bank = bank
        self.encoder = encoder
        self.male_local_idx = male_local_idx
        self.enabled = bool(enabled)
        self.generator = torch.Generator().manual_seed(int(seed))

    @torch.no_grad()
    def bias(self, attr_global_idx, attr_cond, is_rm):
        """None when content does not apply (attribute not in CONTENT_ATTRS,
        or disabled): the caller then runs exactly as without content."""
        if not self.enabled or attr_global_idx not in CONTENT_ATTRS:
            return None
        device = attr_cond.device
        male = (attr_cond[:, self.male_local_idx] >= 0.5) if self.male_local_idx is not None else None
        # Young rm (source young, is_rm True) -> target old.
        rows = self.bank.sample(is_rm.view(-1), male, generator=self.generator)
        stats, valid = self.bank.lookup(rows, device)
        z = self.bank.zscore(stats, valid)
        present = torch.ones(z.size(0), device=device)
        return self.encoder(z, present)
