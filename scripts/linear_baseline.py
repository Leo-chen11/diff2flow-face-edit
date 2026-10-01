"""Training-free linear baseline: how far do the bank's data directions get on
their own, under the SAME protocol evaluate_sdflow.py scores the model with?

The trained editor is, in essence, the bank's dataset directions weighted per
face by a learned gate and magnitude (plus a bounded residual, ControlNet and
the flow). This script keeps the directions and drops everything learned:

    w' = w + sign * alpha * d

    single   d = the mean of the attribute's K slots (one vector for everyone;
             the InterFaceGAN-style baseline).
    strata   d = the mean of the slots of the source face's own stratum (gender x
             age / glasses, whatever the bank was built with). Hard routing, no
             learned gate. The gap single -> strata is what stratifying the
             directions is worth; strata -> the trained model is what the learned
             gate, magnitude, flow and ControlNet add.

Protocol (so numbers line up with evaluate_sdflow.py --edit_direction indep):
  * same sources: the loader order and the num_samples / batch break are the
    eval's, so batch 4 and 250 gives the same 252 faces; the source is the
    StyleGAN reconstruction G(w), downsampled to 256 with the default
    (nearest) F.interpolate;
  * direction (add vs remove) from the independent classifier's source score,
    success = that classifier's score crossing 0.5, sources with an unclear
    score (0.35-0.65) excluded from accuracy, as in the eval;
  * ID_ind = facenet (casia-webface) cosine to the source, LPIPS (alex) to the
    source, both over ALL edited faces (the eval averages them over faces where
    any judge is clear, which is nearly all);
  * the JSON has the eval's layout, one file per variant with alpha as the
    "scale", so scripts/merge_eval_scales.py reads it and the accuracy-vs-ID
    curve can be set beside the trained model's.

alpha 1.0 is the displacement the attribute's high and low groups were observed
to differ by, so it is not the eval's edit_scale; compare at matched ID.

Usage:
    python -m scripts.linear_baseline \
        --direction_bank_path ./data/direction_bank_multi.pth \
        --independent_attr_weights ./data/r50_celebahq_eval.pth \
        --num_samples 250 --batch 4 \
        --out_prefix ./output/baselines/linear_multi
    python -m scripts.merge_eval_scales ./output/baselines/linear_multi_strata.json --judge indep
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

MALE, YOUNG, GLASSES = 20, 39, 15      # CelebA indices used to route sources to strata
VARIANTS = ('single', 'strata')


# ── pure helpers (unit-tested without a GPU) ───────────────────────────────

def is_clear(score, low=0.35, high=0.65):
    return score > high or score < low


def strict_success(src, edit):
    return edit < 0.5 if src > 0.5 else edit > 0.5


def summ(values):
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {'mean': float(a.mean()), 'p10': float(np.percentile(a, 10)),
            'p50': float(np.percentile(a, 50)), 'p90': float(np.percentile(a, 90)),
            'n': int(a.size)}


def slot_groups(names, K):
    """Slot indices of each stratum, in the order of `names` (the bank's
    stratification entry). The bank lays K slots out stratum-major, K/len(names)
    sub-style slots per stratum. Returns None when the entry carries no strata
    (e.g. the K-times-repeated "weighted_avg_k1" of a collapsed age bank)."""
    if not names or len(set(names)) == 1 or K % len(names) != 0:
        return None
    per = K // len(names)
    return [list(range(i * per, (i + 1) * per)) for i in range(len(names))]


def stratum_index(names, male, young, glasses):
    """Per-sample index into `names`: the entry whose tokens (e.g. 'male_young',
    'young_noglasses') all hold for the sample. male/young/glasses: (B,) bool."""
    B = male.shape[0]
    out = torch.zeros(B, dtype=torch.long)
    toks = [set(n.split('_')) for n in names]
    for b in range(B):
        have = {'male' if male[b] else 'female', 'young' if young[b] else 'old',
                'glasses' if glasses[b] else 'noglasses'}
        for j, t in enumerate(toks):
            if t <= have:
                out[b] = j
                break
    return out


def build_direction_tables(raw, strat_names, attribute_index, K):
    """raw: (A, K, 18, 512) = layer_norms * direction_units.
    Returns {attr: {'single': (1,18,512), 'strata': (S,18,512), 'names': [...]}}."""
    tables = {}
    for i, a in enumerate(attribute_index):
        single = raw[i].mean(dim=0, keepdim=True)
        names = list(strat_names.get(a, []))
        groups = slot_groups(names, K)
        if groups is None:
            tables[a] = {'single': single, 'strata': single, 'names': None}
        else:
            tables[a] = {'single': single,
                         'strata': torch.stack([raw[i, g].mean(dim=0) for g in groups]),
                         'names': names}
    return tables


def directions_for(tables, a, variant, male, young, glasses, device):
    """(B, 18, 512) direction per sample for attribute a under a variant."""
    t = tables[a]
    B = male.shape[0]
    if variant == 'single' or t['names'] is None:
        return t['single'].to(device).expand(B, -1, -1)
    idx = stratum_index(t['names'], male.cpu(), young.cpu(), glasses.cpu()).to(device)
    return t['strata'].to(device)[idx]


def run_baseline(loader, num_samples, G_fn, cls_fn, id_fn, lpips_fn, tables, attribute_index,
                 attr_names, alphas, variants, route_by='indep', device='cpu'):
    """Core loop. G_fn(w)->(B,3,H,W) in [-1,1]; cls_fn(img256)->(B,40) probabilities;
    id_fn(img256)->(B,D) unit embeddings; lpips_fn(a,b)->(B,) distances.
    Returns {variant: {str(alpha): {attr: {...}, 'overall': {...}}}}."""
    acc = {}
    seen = 0
    for latent, pred in loader:
        if seen >= num_samples:
            break
        latent = latent.to(device)
        B = latent.size(0)
        src = G_fn(latent).clamp(-1, 1)
        src256 = F.interpolate(src, (256, 256))
        p_src = cls_fn(src256)
        id_src = id_fn(src256)
        if route_by == 'pred':
            male, young, glasses = (pred[:, MALE] > 0.5), (pred[:, YOUNG] > 0.5), (pred[:, GLASSES] > 0.5)
        else:
            male, young, glasses = (p_src[:, MALE] > 0.5), (p_src[:, YOUNG] > 0.5), (p_src[:, GLASSES] > 0.5)
        for variant in variants:
            for ai, a in enumerate(attribute_index):
                name = attr_names[a]
                dirs = directions_for(tables, a, variant, male, young, glasses, device)
                s = p_src[:, a]
                sign = torch.where(s < 0.5, torch.ones_like(s), -torch.ones_like(s))
                for alpha in alphas:
                    new = latent + (sign.view(B, 1, 1) * alpha) * dirs
                    ed256 = F.interpolate(G_fn(new).clamp(-1, 1), (256, 256))
                    p_ed = cls_fn(ed256)
                    idc = (id_src * id_fn(ed256)).sum(dim=1)
                    lp = lpips_fn(src256, ed256).flatten()
                    others = [attribute_index.index(o) for o in attribute_index if o != a]
                    leak = (p_ed[:, attribute_index] - p_src[:, attribute_index]).abs()[:, others].mean(dim=1) \
                        if others else torch.zeros(B)
                    m = acc.setdefault(variant, {}).setdefault(str(alpha), {}).setdefault(name, {})
                    for b in range(B):
                        m.setdefault('id_indep', []).append(idc[b].item())
                        m.setdefault('lpips', []).append(lp[b].item())
                        m.setdefault('leak_indep', []).append(leak[b].item())
                        sb = s[b].item()
                        if is_clear(sb):
                            ok = float(strict_success(sb, p_ed[b, a].item()))
                            m.setdefault('acc_indep', []).append(ok)
                            m.setdefault('acc_indep_add' if sb < 0.5 else 'acc_indep_rm', []).append(ok)
        seen += B

    out = {}
    for variant, by_alpha in acc.items():
        out[variant] = {}
        for alpha, by_attr in by_alpha.items():
            res = {'num_samples': seen}
            pooled = {}
            for name, m in by_attr.items():
                res[name] = {k: summ(v) for k, v in m.items()}
                for k, v in m.items():
                    pooled.setdefault(k, []).extend(v)
            res['overall'] = {k: summ(v) for k, v in pooled.items()}
            out[variant][alpha] = res
    return out


def print_tables(results, attr_names_used):
    for variant, by_alpha in results.items():
        print(f'\n=== variant: {variant} ===')
        print(f'  {"alpha":>6} {"ID_ind":>7} {"AccInd":>7} {"LPIPS":>7}   ' +
              '  '.join(f'{n:>10}' for n in attr_names_used))
        for alpha, res in by_alpha.items():
            o = res['overall']
            cells = []
            for n in attr_names_used:
                a = (res.get(n) or {}).get('acc_indep')
                cells.append(f'{a["mean"] * 100:9.1f}%' if a else f'{"--":>10}')
            print(f'  {float(alpha):>6.2f} {o["id_indep"]["mean"]:>7.4f} '
                  f'{o["acc_indep"]["mean"] * 100:6.1f}% {o["lpips"]["mean"]:>7.4f}   ' + '  '.join(cells))


# ── GPU entry point ─────────────────────────────────────────────────────────

@torch.no_grad()
def main(args):
    import torchvision.transforms as T
    from torch.utils import data
    from common.ops import load_network
    from evaluation.evaluate_sdflow import ATTR_NAMES, IndependentIDJudge
    from models.attribute_estimator import AttributeClassifier
    from models.dataset import SDFlowDataset
    from models.stylegan2.model import Generator

    device = 'cuda'
    bank = torch.load(args.direction_bank_path, map_location='cpu')
    du, ln = bank['direction_units'].float(), bank['layer_norms'].float()
    if du.ndim == 3:
        du, ln = du.unsqueeze(1), ln.unsqueeze(1)
    K = int(bank.get('num_k', du.shape[1]))
    bank_attrs = [int(a) for a in bank['attribute_index']]
    attrs = [a for a in (args.attrs or bank_attrs) if a in bank_attrs]
    if not attrs:
        raise SystemExit(f'none of --attrs {args.attrs} are in the bank {bank_attrs}')
    sel = [bank_attrs.index(a) for a in attrs]
    raw = (ln.unsqueeze(-1) * du)[sel]
    for i, a in enumerate(attrs):
        if not torch.isfinite(raw[i]).all():
            raise SystemExit(f'attr {a}: non-finite directions in the bank; rebuild it.')
    tables = build_direction_tables(raw, bank.get('stratification', {}), attrs, K)
    names = {a: ATTR_NAMES.get(a, f'attr{a}') for a in attrs}
    for a in attrs:
        n = tables[a]['names']
        print(f'attr {a} ({names[a]}): K={K}, strata {n if n else "none (single direction)"}')

    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.to(device).eval()
    cls = AttributeClassifier(backbone=args.independent_attr_backbone)
    cls.load_state_dict(load_network(args.independent_attr_weights))
    cls.to(device).eval()
    id_judge = IndependentIDJudge(device, pretrained=args.id_indep_pretrained)
    import lpips
    lp = lpips.LPIPS(net='alex').to(device).eval()

    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file, train=False,
        transform=T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)),
                             T.Normalize(mean=0.5, std=0.5)]))
    loader = data.DataLoader(dataset, shuffle=False, batch_size=args.batch,
                             num_workers=4, drop_last=False)

    def batches():                     # (latent, pred) only, same order as the eval's loader
        for _img, latent, pred in loader:
            yield latent, pred

    results = run_baseline(
        batches(), args.num_samples,
        G_fn=lambda w: G([w], input_is_latent=True, randomize_noise=False)[0],
        cls_fn=lambda x: torch.sigmoid(cls(x)[0]),
        id_fn=lambda x: id_judge.extract(x),
        lpips_fn=lambda a, b: lp(a, b),
        tables=tables, attribute_index=attrs, attr_names=names,
        alphas=args.alphas, variants=args.variants, route_by=args.route_by, device=device)

    print_tables(results, [names[a] for a in attrs])
    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    os.makedirs(out_dir, exist_ok=True)
    for variant, by_alpha in results.items():
        doc = {'config': {'method': f'linear_{variant}', 'direction_bank_path': args.direction_bank_path,
                          'attribute_index': attrs, 'num_samples': args.num_samples,
                          'batch': args.batch, 'route_by': args.route_by,
                          'independent_attr_weights': args.independent_attr_weights,
                          'edit_direction': 'indep', 'step': None,
                          'note': 'alpha is the scale key; alpha 1.0 = observed high/low group displacement'},
               **by_alpha}
        path = f'{args.out_prefix}_{variant}.json'
        with open(path, 'w') as f:
            json.dump(doc, f, indent=2)
        print(f'saved {path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--direction_bank_path', required=True)
    p.add_argument('--attrs', nargs='*', type=int, default=None, help='Default: every attribute in the bank.')
    p.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument('--alphas', nargs='+', type=float,
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument('--route_by', default='indep', choices=['indep', 'pred'],
                   help="Who says which stratum a source is in: the independent classifier on "
                        "the source reconstruction (default) or the dataset's r34 predictions.")
    p.add_argument('--independent_attr_weights', required=True)
    p.add_argument('--independent_attr_backbone', default='r50')
    p.add_argument('--id_indep_pretrained', default='casia-webface')
    p.add_argument('--num_samples', type=int, default=250)
    p.add_argument('--batch', type=int, default=4, help='Keep 4 to see the eval\'s 252 faces.')
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--out_prefix', required=True, help='Writes <prefix>_<variant>.json')
    main(p.parse_args())
