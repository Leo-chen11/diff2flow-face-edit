"""AXIS 1: WHERE does each attribute's evidence live, derived not hand-written?

Every region this project uses today is hand-specified: LOCAL_REGION_CLASSES
maps eyeglasses to BiSeNet classes [2,3,4,5,6], AGE_TEXTURE_REGION_CLASS maps
age to [1], and gender(20) maps to nothing at all, so
--controlnet_region_cond feeds it an all-ones mask -- zero spatial
information. That approach does not scale: CelebA has 40 attributes and no
one is going to hand-write a region for Chubby, Pale_Skin or Oval_Face, and
several of them (gender first among them) have no single BiSeNet class to
point at because the label set has no jaw, forehead or cheek.

This script replaces the hand-written table with a measurement. For an
attribute k it asks the project's own attribute teacher
(models/attribute_estimator.py AttributeClassifier, the r34 whose 40
independent heads share one ResNet trunk) WHERE the evidence for k is, by
Grad-CAM on head k. The answer comes out automatically, for any of the 40
attributes, with nothing hand-specified.

SIGNED, NOT RECTIFIED. Standard Grad-CAM takes ReLU of the weighted
activation sum because it asks "what supports this class". An edit needs
more than that: an ADD edit has to know where the evidence AGAINST the
attribute currently sits, and a REMOVE edit where the evidence FOR it sits.
So the map is kept signed and both halves are saved; the consumer takes
relu(+cam) or relu(-cam) per direction. Rectifying here would throw away
exactly the half the add direction needs.

WHY THIS IS A CHEAP DECISION POINT. The mechanism this feeds (swapping the
region_cond input channel from a BiSeNet mask to this map) costs a training
run. Looking at the maps costs minutes. If Male's map does not concentrate
on jaw/brow/hairline, or Pale_Skin's is not diffuse where age's is
localized, then "classifier attribution defines the region" is false for
this classifier and the training run is not worth starting. Run --preview
FIRST and look at the montage; only then --dump.

Nothing here trains or touches a checkpoint. It reads the attribute teacher
and the dataset.

Usage:
    # 1. LOOK FIRST -- montage of a few faces with the CAM overlaid
    python -m scripts.precompute_attr_saliency --preview \
        --attrs 15 20 39 --num_faces 8 --out_dir ./saliency_probe

    # 2. only if the maps look right -- dump the whole index
    python -m scripts.precompute_attr_saliency --dump \
        --attrs 15 20 39 --out ./data/ffhq_attr_saliency.pth
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils import data

from common.ops import load_network
from models.attribute_estimator import AttributeClassifier
from models.dataset import SDFlowDataset

# Same six names evaluation/evaluate_sdflow.py knows; anything else prints
# as its bare CelebA index, which is enough to read a montage by.
ATTR_NAMES = {15: 'Eyeglasses', 20: 'Male', 24: 'No_Beard', 31: 'Smiling',
              33: 'Wavy_Hair', 39: 'Young'}

# The teacher is fed 256x256 everywhere in this project (see
# evaluate_sdflow.py's attr_teacher(src_face_256) call sites), in the
# generator's own [-1, 1] range with no extra normalization. Matching that
# exactly matters: a map computed under a different input convention would
# describe a classifier this project never actually queries.
TEACHER_RES = 256


def _resolve_tap(extractor, name):
    """The ResNet stage whose activations Grad-CAM reads.

    layer4 is 8x8 at 256 input, layer3 is 16x16. 16x16 is the better default
    here: the consumer resizes this map up to a 64x64+ injection ladder, and
    8x8 upsampled 8x is too coarse to distinguish a jaw from a cheek -- which
    is precisely the distinction this whole axis exists to recover.
    """
    if not hasattr(extractor, name):
        raise ValueError(f'attribute teacher has no stage {name!r}')
    return getattr(extractor, name)


def signed_gradcam(teacher, tap, x, attr_ids):
    """Signed Grad-CAM for several attributes on one batch.

    Returns (B, len(attr_ids), h, w), the weighted activation sum WITHOUT
    the usual ReLU (see module docstring).

    The teacher's parameters are frozen with requires_grad_(False) by every
    caller in this project, so nothing would build an autograd graph on its
    own -- x.requires_grad_(True) is what puts the activations on the graph.
    Gradient is taken w.r.t. the tapped ACTIVATION, not the input, so this is
    Grad-CAM proper rather than an input-gradient saliency map.
    """
    acts = {}

    def hook(_m, _inp, out):
        acts['a'] = out

    handle = tap.register_forward_hook(hook)
    try:
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits, _ = teacher.forward_attr(x)
            a = acts['a']                                    # (B, C, h, w)
            maps = []
            for k in attr_ids:
                # retain_graph: the trunk forward is shared across every
                # attribute in attr_ids, so it has to survive each backward.
                grad, = torch.autograd.grad(
                    logits[:, k].sum(), a, retain_graph=True)
                weights = grad.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
                maps.append((weights * a).sum(dim=1))           # (B, h, w)
        return torch.stack(maps, dim=1).detach()
    finally:
        handle.remove()


def normalize_signed(cam):
    """Scale each map to [-1, 1] by its own peak magnitude.

    Per-sample, per-attribute: the raw weighted sums differ in scale by
    orders of magnitude between attributes (different heads, different logit
    scales), so a shared normalizer would make the weaker attribute's map
    look like noise next to the stronger one. Preserves sign and the
    relative shape within a map, which is all the consumer needs.
    """
    peak = cam.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return cam / peak


def overlay(face_pm1, cam, size=256):
    """Face with a signed CAM painted over it: red = evidence FOR the
    attribute, blue = evidence AGAINST. Grey where the map is flat, which is
    itself the reading for a genuinely diffuse attribute."""
    face = F.interpolate(face_pm1[None], (size, size), mode='bilinear',
                         align_corners=False)[0]
    face01 = (face.clamp(-1, 1) + 1) * 0.5
    m = F.interpolate(cam[None, None], (size, size), mode='bilinear',
                      align_corners=False)[0, 0]
    pos = m.clamp(min=0)
    neg = (-m).clamp(min=0)
    tint = torch.stack([pos, torch.zeros_like(pos), neg], dim=0)
    blended = (face01 * (1 - 0.55 * (pos + neg)) + 0.55 * tint).clamp(0, 1)
    arr = (blended * 255).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def build_teacher(args):
    teacher = AttributeClassifier(backbone=args.attr_backbone)
    teacher.load_state_dict(load_network(args.attribute_weights))
    teacher.cuda().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def build_loader(args, batch_size):
    img_transform = T.Compose([
        T.ToTensor(), T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    # train=True, matching training/train_sdflow.py's train_dataset -- NOT
    # the train=False convention every other probe script in this project
    # uses. SDFlowDataset splits on df['split'] == (not train): train=True
    # and train=False are DISJOINT sets of images, not a superset/subset
    # relationship. The whole point of --dump is to build an index that
    # train_sdflow.py's training loop can look attributes up in DURING
    # TRAINING, which iterates train_dataset (train=True) -- an index built
    # from the eval split would silently cover none of the images training
    # actually sees.
    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file,
        train=True, transform=img_transform,
    )
    loader = data.DataLoader(dataset, shuffle=False, batch_size=batch_size,
                             num_workers=args.workers, drop_last=False)
    return dataset, loader


def run_preview(args, teacher, tap):
    _dataset, loader = build_loader(args, batch_size=1)
    rows = []
    seen = 0
    for img, _latent, _pred in loader:
        if seen >= args.num_faces:
            break
        img = img.cuda()
        x = F.interpolate(img, (TEACHER_RES, TEACHER_RES), mode='bilinear',
                          align_corners=False)
        cam = normalize_signed(signed_gradcam(teacher, tap, x, args.attrs))[0]

        panels = [_label(_to_pil(img[0], args.tile), 'source')]
        for i, k in enumerate(args.attrs):
            name = ATTR_NAMES.get(k, f'attr{k}')
            spread = cam[i].abs().mean().item()   # low => diffuse/flat map
            panels.append(_label(overlay(img[0], cam[i], args.tile),
                                 f'{name}  mean|cam|={spread:.2f}'))
        rows.append(_hstack(panels))
        seen += 1

    if not rows:
        raise SystemExit('no samples loaded.')
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'attr_saliency_preview.png')
    _vstack(rows).save(path)
    print(f'\nsaved -> {path}')
    print('\nWHAT TO CHECK, before spending a training run on this:')
    print('  Male(20)       should light the JAW / BROW RIDGE / HAIRLINE.')
    print('                 If it lights only the mouth or is flat, the')
    print('                 premise fails for the attribute that needed it most.')
    print('  Young(39)      should light SKIN + EYE REGION (and often hair).')
    print('  Eyeglasses(15) should light the EYE REGION tightly -- this is the')
    print('                 sanity check: a known-local attribute must look local,')
    print('                 or the method is broken rather than the attribute.')
    print('  A diffuse map is a RESULT, not a failure: it says the attribute')
    print('  genuinely has no region, and the framework should treat it as global.')
    print('  Red = evidence FOR, blue = evidence AGAINST (add edits need blue).')


def run_dump(args, teacher, tap):
    dataset, loader = build_loader(args, batch_size=args.batch)
    maps = []
    files = []
    done = 0
    for img, _latent, _pred in loader:
        img = img.cuda()
        x = F.interpolate(img, (TEACHER_RES, TEACHER_RES), mode='bilinear',
                          align_corners=False)
        cam = normalize_signed(signed_gradcam(teacher, tap, x, args.attrs))
        # float16: these are normalized to [-1,1] and get bilinearly resized
        # to the injection resolution anyway, so fp16 costs nothing real and
        # halves a file that has one map per (image, attribute).
        maps.append(cam.cpu().half())
        done += img.size(0)
        files.extend(dataset.image_list[done - img.size(0):done])
        if done % (args.batch * 50) < args.batch:
            print(f'  {done}/{len(dataset)}')
    maps = torch.cat(maps, dim=0)
    payload = {
        'maps': maps,                  # (N, len(attrs), h, w) fp16, signed [-1,1]
        'attrs': list(args.attrs),     # absolute CelebA indices, column order
        'files': files,                # same order, so a consumer can index by name
        'teacher_weights': args.attribute_weights,
        'tap': args.tap,
        'teacher_res': TEACHER_RES,
        'signed': True,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    torch.save(payload, args.out)
    print(f'\nsaved {tuple(maps.shape)} -> {args.out}')
    print('  maps are SIGNED and peak-normalized per (image, attribute).')
    print('  A consumer wanting a one-sided region takes relu(+cam) for the')
    print('  rm direction and relu(-cam) for add, then resizes to its own')
    print('  resolution -- same role the BiSeNet mask plays today.')


# ── small montage helpers (kept local; scripts/dump_attr_failures.py's are
# shaped around its src/edit pair layout, not a variable-width row) ────────

def _to_pil(img_tensor, size):
    x = F.interpolate(img_tensor[None].clamp(-1, 1), (size, size),
                      mode='bilinear', align_corners=False)[0]
    x = ((x + 1) * 0.5 * 255).byte().cpu()
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def _label(pil, text):
    out = Image.new('RGB', (pil.width, pil.height + 18), (20, 20, 20))
    out.paste(pil, (0, 18))
    ImageDraw.Draw(out).text((4, 4), text, fill=(210, 210, 210))
    return out


def _hstack(panels):
    w = sum(p.width for p in panels)
    h = max(p.height for p in panels)
    out = Image.new('RGB', (w, h), (20, 20, 20))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width
    return out


def _vstack(rows):
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows)
    out = Image.new('RGB', (w, h), (20, 20, 20))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preview', action='store_true',
                      help='Render a montage and stop. RUN THIS FIRST -- it is '
                           'the cheap check on whether the premise holds.')
    mode.add_argument('--dump', action='store_true',
                      help='Compute maps for the whole index and save them.')

    p.add_argument('--attrs', nargs='*', type=int, default=[15, 20, 39],
                   help='Absolute CelebA indices. Any of the 40 works -- the '
                        'point of this script is that nothing is hand-specified '
                        'per attribute, so 13 (Chubby) or 26 (Pale_Skin) are '
                        'legitimate things to look at even though this project '
                        'does not edit them.')
    p.add_argument('--tap', default='layer3',
                   help='Which ResNet stage to read activations from. layer3 is '
                        '16x16 at 256 input, layer4 is 8x8 (see _resolve_tap).')
    p.add_argument('--num_faces', type=int, default=8, help='--preview only.')
    p.add_argument('--tile', type=int, default=256, help='--preview tile size.')
    p.add_argument('--out_dir', default='./saliency_probe', help='--preview only.')
    p.add_argument('--out', default='./data/ffhq_attr_saliency.pth',
                   help='--dump only.')
    p.add_argument('--batch', type=int, default=8, help='--dump only.')
    p.add_argument('--workers', type=int, default=2)

    p.add_argument('--attr_backbone', default='r34',
                   help="Teacher backbone. 'r34' matches "
                        './data/r34_a40_age_256_classifier.pth, the model this '
                        'project trains against -- and therefore the one whose '
                        'idea of "where the evidence is" the edit is actually '
                        'chasing.')
    p.add_argument('--attribute_weights',
                   default='./data/r34_a40_age_256_classifier.pth')
    p.add_argument('--index_file',  default='./data/ffhq.txt')
    p.add_argument('--image_root',  default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file',  default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--img_size', type=int, default=512)

    args = p.parse_args()
    teacher = build_teacher(args)
    tap = _resolve_tap(teacher.extractor, args.tap)
    print(f'teacher={args.attribute_weights} backbone={args.attr_backbone} '
          f'tap={args.tap} attrs={args.attrs}')
    if args.preview:
        run_preview(args, teacher, tap)
    else:
        run_dump(args, teacher, tap)
