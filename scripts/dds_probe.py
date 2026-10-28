"""Can DDS alone make the edit? A few-minute check before spending a training run on it.

For each attribute, takes test faces the R50 judge reads clearly, and optimizes
their W+ latent (layers below the DDS fine-layer cutoff, as training does) with
the DDS loss and nothing else -- no teacher, no ID loss. Every --report_every
steps it records:

  R50     share of faces whose R50 reading crossed 0.5 the requested way
  r18     the same with the CelebA r18 judge, on faces it reads clearly and
          like the R50 on the source
  id_arc  ArcFace cosine to the source reconstruction

Adam normalizes the gradient's scale, so this measures the DIRECTION DDS
pushes in, which no loss weight can fix. Reading the result:

  R50 and r18 both rise well past chance while id_arc stays at the level the
  trained model edits at  -> DDS carries real attribute signal under this
                             configuration; worth training with it
  R50 rises, r18 does not  -> DDS finds a shortcut one judge accepts
  neither rises            -> noise at this configuration; change it (size,
                             CFG, timesteps, samples) before training

Run once per configuration and compare, e.g.

    python -m scripts.dds_probe --checkpoint_dir ./output/SDFlow/<run> \\
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \\
        --dds_image_size 256 --dds_cfg 1.0 --tag base
    python -m scripts.dds_probe ... --dds_image_size 512 --dds_cfg 7.5 --dds_n_samples 4 --dds_contrast --tag full
"""
import csv
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms as T
from torch.utils import data

import evaluation.evaluate_sdflow as E
from evaluation.evaluate_sdflow import (
    ATTR_NAMES, _latest_step, apply_run_config, build_optional_judges, build_parser,
    load_models, resize_for_judge,
)
from models.arcface import ArcFaceJudge
from models.dataset import SDFlowDataset
from models.diffusion_guidance import FrozenDiffusionDDSGuidance

CLEAR_LO, CLEAR_HI = 0.35, 0.65
# training's defaults: --dds_fine_layer_start 7, --age_dds_fine_layer_start 12,
# and --gender_dds_fine_layer_start 12 in the dds arms
FINE_START = {39: 12, 20: 12}
LOCAL_REGION_CLASSES = {15: [2, 3, 4, 5, 6]}   # as training/train_sdflow.py


def main():
    p = build_parser()
    p.add_argument('--attrs', nargs='+', type=int, default=None,
                   help='CelebA ids to probe; default the run\'s edited attributes.')
    p.add_argument('--num_faces', type=int, default=32, help='Faces per attribute.')
    p.add_argument('--probe_batch', type=int, default=4)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--lr', type=float, default=0.01, help='Adam step on W+.')
    p.add_argument('--report_every', type=int, default=10)
    p.add_argument('--fine_start', type=int, default=7,
                   help='W+ layers at/above this stay fixed (attributes in FINE_START override it).')
    p.add_argument('--dds_model_id', default='SG161222/Realistic_Vision_V5.1_noVAE')
    p.add_argument('--dds_vae_model_id', default='stabilityai/sd-vae-ft-mse')
    p.add_argument('--dds_image_size', type=int, default=256)
    p.add_argument('--dds_cfg', type=float, default=1.0)
    p.add_argument('--dds_tmin', type=int, default=50)
    p.add_argument('--dds_tmax', type=int, default=700)
    p.add_argument('--dds_n_samples', type=int, default=1)
    p.add_argument('--dds_contrast', action='store_true',
                   help='Source prompt = the source\'s own state (--dds_contrast_prompt).')
    p.add_argument('--no_face_mask', action='store_true')
    p.add_argument('--arcface_weights', default='./data/model_ir_se50.pth')
    p.add_argument('--tag', default='probe')
    p.set_defaults(eval_resize='area')
    args = p.parse_args()
    args = apply_run_config(args)
    E._RESIZE_MODE = args.eval_resize
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (R50 judge).')

    _, _, G, id_criterion, _, _, _ = load_models(args)
    _, _, _, r50, _, celeb = build_optional_judges(args, args.attribute_index, id_criterion)
    arc = ArcFaceJudge(args.arcface_weights)
    parser = None
    if not args.no_face_mask:
        from common.face_parser import FaceParser
        parser = FaceParser(weights_path=args.face_parser_weights).cuda().eval()
    dds = FrozenDiffusionDDSGuidance(
        model_id=args.dds_model_id, vae_model_id=args.dds_vae_model_id or None,
        image_size=args.dds_image_size, timestep_min=args.dds_tmin, timestep_max=args.dds_tmax,
        guidance_scale=args.dds_cfg, fp16=True).cuda()
    for m in (G, r50, arc):
        for q in m.parameters():
            q.requires_grad_(False)

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.probe_batch, num_workers=2)
    attrs = list(args.attrs or args.attribute_index)

    def gen(w):
        return G([w], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)

    def r50_probs(x):
        return torch.sigmoid(r50(resize_for_judge(x, (256, 256)))[0])[:, :40]

    # clear sources per attribute
    picked = {a: [] for a in attrs}
    with torch.no_grad():
        for _, latent, _ in loader:
            latent = latent.cuda()
            sp = r50_probs(gen(latent))
            for a in attrs:
                for i in range(latent.shape[0]):
                    if len(picked[a]) < args.num_faces and (sp[i, a] > CLEAR_HI or sp[i, a] < CLEAR_LO):
                        picked[a].append(latent[i])
            if all(len(v) >= args.num_faces for v in picked.values()):
                break

    rows = []
    for a in attrs:
        fs = FINE_START.get(a, args.fine_start)
        hist = {}
        lat_all = torch.stack(picked[a])
        for b0 in range(0, lat_all.shape[0], args.probe_batch):
            lat = lat_all[b0:b0 + args.probe_batch]
            with torch.no_grad():
                src = gen(lat)
                s256 = resize_for_judge(src, (256, 256))
                sp = r50_probs(src)[:, a]
                sc = celeb.scores(s256)[:, a] if celeb is not None else None
                src_arc = arc.extract(s256)
                mask = None
                if parser is not None:
                    mask = (parser.get_region_mask(src, LOCAL_REGION_CLASSES[a])
                            if a in LOCAL_REGION_CLASSES else parser.get_mask(src))
            add = sp < 0.5
            tgt = add.float()
            delta = torch.zeros_like(lat[:, :fs], requires_grad=True)
            opt = torch.optim.Adam([delta], lr=args.lr)
            abs_idx = torch.full((lat.shape[0],), a, device=lat.device, dtype=torch.long)
            for step in range(args.steps + 1):
                w = torch.cat([lat[:, :fs] + delta, lat[:, fs:]], dim=1)
                face = gen(w)
                if step % args.report_every == 0:
                    with torch.no_grad():
                        f256 = resize_for_judge(face, (256, 256))
                        ep = r50_probs(face)[:, a]
                        ok = torch.where(add, ep > 0.5, ep < 0.5).float()
                        idc = torch.nn.functional.cosine_similarity(arc.extract(f256), src_arc, dim=-1)
                        h = hist.setdefault(step, {'r50': [], 'r18': [], 'id': []})
                        h['r50'] += ok.tolist()
                        h['id'] += idc.tolist()
                        if sc is not None:
                            ec = celeb.scores(f256)[:, a]
                            keep = ((sc > CLEAR_HI) | (sc < CLEAR_LO)) & ((sc > 0.5) == (sp > 0.5))
                            h['r18'] += torch.where(add, ec > 0.5, ec < 0.5).float()[keep].tolist()
                if step == args.steps:
                    break
                loss, _ = dds(src_images=src, edit_images=face, attr_abs_idx=abs_idx,
                              target_values=tgt, face_mask=mask, contrast_source=args.dds_contrast,
                              n_samples=args.dds_n_samples)
                opt.zero_grad()
                loss.backward()
                opt.step()

        name = ATTR_NAMES.get(a, f'attr{a}')
        print(f'\n{name} (W+ layers 0-{fs - 1}, {lat_all.shape[0]} faces)')
        print(f'  {"step":>5}{"R50 %":>8}{"r18 %":>8}{"id_arc":>8}')
        for step in sorted(hist):
            h = hist[step]
            mean = lambda v: sum(v) / len(v) if v else float('nan')
            r = {'attribute': name, 'step': step, 'r50': mean(h['r50']), 'r18': mean(h['r18']),
                 'id_arc': mean(h['id']), 'n': len(h['r50'])}
            rows.append(r)
            print(f'  {step:>5}{r["r50"] * 100:8.1f}{r["r18"] * 100:8.1f}{r["id_arc"]:8.3f}')

    out = os.path.join(args.checkpoint_dir, f'dds_probe_{args.tag}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['attribute', 'step', 'r50', 'r18', 'id_arc', 'n'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
