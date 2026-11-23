"""How long an edit takes, for our model and every baseline, on the same faces and GPU.

What is timed (CUDA-synchronised wall clock, after --warmup untimed calls, median over the calls):

  latent    the W+ edit alone: everything a method does from the source to the edited latent
            (ours: conditioner reading + flow encode + flow decode + the direction bank / delta cap)
  G         rendering the edited latent with StyleGAN2 at 1024 px (the same for every method)
  total     latent + G, per edited face
  NFE       velocity-network evaluations of the flow, encode + decode (flow methods)
  peak MB   peak GPU memory of the method's calls

and, for our model, the slider case: the source is read and encoded ONCE and decoded at
--slider_steps strengths, so the cost of each further adjustment is a decode (reported per strength,
the encode shared out).
DeltaEdit renders inside its own edit, so only its total is measured.

Our flow (and SDFlow / StyleFlow) is timed under every --ode variant:
    trained          the solver the run was trained with (dopri5, tol 1e-5), log-det integrated
    fast             the same solver, latent only (--fast_edit: no divergence term)
    dopri5@1e-3+fast an adaptive solver at another tolerance
    rk4x4+fast       a fixed-step solver, 4 steps
Their QUALITY under each variant comes from scripts/editing_curves.py with the same --ode_* flags
(scripts/run_speed_study.sh runs both and scripts/solver_tradeoff.py puts them in one table).

    python -m scripts.speed_benchmark --checkpoint_dir <run> --step 130000 \\
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \\
        --methods ours sdflow styleflow adatrans gradctrl plugen deltaedit lt interfacegan

Writes <out_dir>/speed.json and speed.md. Run nothing else on the GPU meanwhile: other jobs
change every number. Times are for this GPU; quote the GPU with them.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms as T
from torch.utils import data

from evaluation.evaluate_sdflow import (
    apply_run_config, bank_guided_delta, build_parser, cap_delta_norm, edited_attr_value, load_models,
)
import evaluation.evaluate_sdflow as E
from models.dataset import SDFlowDataset
from models.flows.flow import FIXED_STEP_SOLVERS, configure_inference, flow_nfe

DEFAULT_SCALE = {          # a strength near R50 90% accuracy for each method (its curves' units)
    'ours': 0.8, 'sdflow': 1.0, 'styleflow': 1.0, 'adatrans': 10, 'gradctrl': 9, 'plugen': 1.0,
    'deltaedit': 2.0, 'lt': 1.0, 'interfacegan': 3.0,
}
FLOW_METHODS = ('ours', 'sdflow', 'styleflow')


def parse_variant(spec):
    """'trained' | 'fast' | SOLVER[@TOL|xSTEPS][+fast] -> dict for configure_inference."""
    fast = spec.endswith('+fast') or spec == 'fast'
    core = spec[:-5] if spec.endswith('+fast') else spec
    if core in ('trained', 'fast'):
        return {'solver': 'dopri5', 'tol': 1e-5, 'steps': None, 'skip_divergence': fast}
    solver, tol, steps = core, None, None
    if '@' in core:
        solver, tol = core.split('@'); tol = float(tol)
    elif 'x' in core:
        solver, steps = core.split('x'); steps = int(steps)
    if solver in FIXED_STEP_SOLVERS and not steps:
        raise SystemExit(f'--ode variant {spec}: {solver} is fixed-step, write {solver}xSTEPS')
    return {'solver': solver, 'tol': tol, 'steps': steps, 'skip_divergence': fast}


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key, fn, *a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        self.t.setdefault(key, []).append(time.perf_counter() - t0)
        return out


def summary(ts, per):
    """median / mean / std in ms per face."""
    v = [x * 1000 / per for x in ts]
    return {'median_ms': statistics.median(v), 'mean_ms': statistics.mean(v),
            'std_ms': statistics.pstdev(v) if len(v) > 1 else 0.0, 'calls': len(v)}


def main():
    p = build_parser()
    p.add_argument('--methods', nargs='+', default=['ours'],
                   choices=['ours', 'sdflow', 'styleflow', 'adatrans', 'gradctrl', 'plugen', 'deltaedit', 'lt',
                            'interfacegan'])
    p.add_argument('--ode', nargs='+', default=['trained', 'fast', 'dopri5@1e-3+fast', 'rk4x4+fast', 'rk4x8+fast'],
                   help='solver variants for the flow methods (see the module docstring)')
    p.add_argument('--batch_sizes', nargs='+', type=int, default=[1, 8])
    p.add_argument('--num_faces', type=int, default=32, help='timed faces per batch size (dev faces)')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--slider_steps', type=int, default=5)
    p.add_argument('--attrs', nargs='+', type=int, default=None, help='default: the run\'s attributes')
    p.add_argument('--scale', action='append', default=[], metavar='METHOD=VALUE', help='override DEFAULT_SCALE')
    p.add_argument('--out_dir', default='./output/speed')
    # baselines (the defaults of scripts/run_heldout_curves.sh)
    p.add_argument('--sdflow_ckpt_dir', default='output/sdflow_original_5attr/save_models')
    p.add_argument('--sdflow_step', type=int, default=17549)
    p.add_argument('--sdflow_attrs', nargs='+', type=int, default=[15, 20, 39, 31, 5])
    p.add_argument('--sdflow_img_size', type=int, default=256)
    p.add_argument('--sdflow_backbone', default='auto')
    p.add_argument('--styleflow_ckpt', default='./output/baselines/styleflow/styleflow.pth')
    p.add_argument('--adatrans_dir', default='./output/baselines/adatrans')
    p.add_argument('--gradctrl_dir', default='./output/baselines/gradctrl')
    p.add_argument('--plugen_dir', default='./output/baselines/plugen')
    p.add_argument('--deltaedit_dir', default='./data/deltaedit')
    p.add_argument('--lt_dir', default='./output/baselines/lt')
    p.add_argument('--interfacegan_boundaries', default='./data/interfacegan_boundaries.pth')
    args = apply_run_config(p.parse_args())
    if not torch.cuda.is_available():
        raise SystemExit('needs a GPU')
    scale = dict(DEFAULT_SCALE)
    for kv in args.scale:
        k, v = kv.split('='); scale[k] = float(v)
    variants = {v: parse_variant(v) for v in args.ode}
    os.makedirs(args.out_dir, exist_ok=True)

    prior, conditioner, G, id_criterion, _, _, bank = load_models(args)
    attrs = list(args.attrs or args.attribute_index)
    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root, latents_file=args.latent_file,
                       preds_file=args.preds_file, train=False, transform=tf)

    # ── the methods: each a function (img, latent, pred, g, d, sc) -> edited latent ───────────────
    from scripts import editing_curves as EC
    editors, notes, flows = {}, {}, {}

    def ours_latent(img, latent, pred, g, d, sc, tm=None, key=''):
        tm = tm or Timer()
        li = args.attribute_index.index(g)
        _, id_cond, attr_cond = tm(key + 'cond', conditioner.make_condition, img, latent, id_criterion)
        zp = torch.zeros(latent.size(0), 18, 1, device=latent.device)
        z, _ = tm(key + 'encode', prior, latent, torch.cat([id_cond, attr_cond], 1), zp)
        nfe_e = flow_nfe(prior)
        new_attr = attr_cond.clone()
        new_attr[:, li] = edited_attr_value(attr_cond[:, li], sc, g, direction=d)
        raw, _ = tm(key + 'decode', prior, z, torch.cat([id_cond, new_attr], 1), zp, reverse=True)
        nfe_d = flow_nfe(prior)
        if bank is not None:
            delta = bank_guided_delta(bank, raw - latent, new_attr - attr_cond, li, latent, attr_cond, id_cond, [g])
        else:
            delta = cap_delta_norm(raw - latent, E._FINAL_DELTA_MAX_NORM)
        return latent + delta, (nfe_e, nfe_d)
    if 'ours' in args.methods:
        editors['ours'] = ours_latent
        flows['ours'] = prior

    for m in args.methods:
        if m == 'ours':
            continue
        try:
            if m == 'sdflow':
                sdf = EC.OfficialSDFlow(args.sdflow_ckpt_dir, args.sdflow_step, args.sdflow_attrs,
                                        img_size=args.sdflow_img_size, backbone=args.sdflow_backbone)

                def sdflow_latent(img, latent, pred, g, d, sc, sdf=sdf):
                    if sdf.img_size and img.shape[-1] != sdf.img_size:
                        img = torch.nn.functional.interpolate(img, (sdf.img_size,) * 2, mode='bilinear',
                                                              align_corners=False)
                    r = sdf.labeldist(img, latent)
                    z = sdf.flow(latent, r, torch.zeros(latent.size(0), 18, 1).to(latent))[0]
                    ne = flow_nfe(sdf.flow)
                    new = r.clone()
                    new[:, sdf.attrs.index(g)] = torch.where(d > 0, 1.0 + sc, -sc)
                    w = sdf.flow(z, new, reverse=True)
                    return w, (ne, flow_nfe(sdf.flow))
                editors[m], flows[m] = sdflow_latent, sdf.flow
                notes[m] = [a for a in attrs if a in sdf.attrs]
            elif m == 'styleflow':
                from models.baselines import styleflow_cnf, styleflow_edit
                ck = torch.load(args.styleflow_ckpt, map_location='cpu')
                sflow = styleflow_cnf(len(ck['attrs'])).cuda().eval()
                sflow.load_state_dict(ck['state_dict'])
                sattrs = list(ck['attrs'])

                def styleflow_latent(img, latent, pred, g, d, sc, sflow=sflow, sattrs=sattrs):
                    w = styleflow_edit(sflow, latent, pred[:, sattrs], sattrs.index(g), d, sc, attr=g)
                    return w, (None, None)
                editors[m], flows[m] = styleflow_latent, sflow
                notes[m] = [a for a in attrs if a in sattrs]
            elif m == 'adatrans':
                from models.adatrans import adatrans_edit
                ada = EC.load_adatrans(args.adatrans_dir, attrs)
                editors[m] = lambda img, latent, pred, g, d, sc, ada=ada: (adatrans_edit(ada[g], latent, d, sc), None)
            elif m == 'gradctrl':
                from models.gradctrl import gradctrl_edit
                gattrs = [a for a in attrs if a != 5]                     # no Bangs classifier
                gc = EC.load_gradctrl(args.gradctrl_dir, gattrs)
                editors[m] = lambda img, latent, pred, g, d, sc, gc=gc: (
                    gradctrl_edit(latent=latent, direction=d, steps=sc, **gc[g]), None)
                notes[m] = gattrs
            elif m == 'plugen':
                from models.plugen import plugen_edit
                pg, pattrs = EC.load_plugen(args.plugen_dir, attrs)
                editors[m] = lambda img, latent, pred, g, d, sc, pg=pg, pattrs=pattrs: (
                    plugen_edit(pg, latent, pattrs.index(g), d, sc, None), None)
            elif m == 'lt':
                from models.baselines import LatentTransformer
                nets = {}
                for a in attrs:
                    path = os.path.join(args.lt_dir, f'tnet_{a}.pth')
                    path = path if os.path.exists(path) else path + '.tar'
                    nets[a] = LatentTransformer().cuda().eval()
                    nets[a].load_state_dict(torch.load(path, map_location='cpu'))
                editors[m] = lambda img, latent, pred, g, d, sc, nets=nets: (nets[g].edit(latent, d * sc), None)
            elif m == 'interfacegan':
                dirs = EC.interfacegan_directions(args.interfacegan_boundaries, attrs)
                editors[m] = lambda img, latent, pred, g, d, sc, dirs=dirs: (
                    latent + (d * sc).view(-1, 1, 1) * dirs[g].to(latent), None)
            elif m == 'deltaedit':
                from models.deltaedit import DeltaEdit
                dattrs = [a for a in attrs if a != 39]                    # Young not covered by the paper
                de = DeltaEdit(args.deltaedit_dir, dattrs)
                editors[m] = ('renders', de)
                notes[m] = dattrs
        except (SystemExit, FileNotFoundError, OSError, KeyError) as exc:
            print(f'  - {m}: skipped ({exc})')

    # ── timing ─────────────────────────────────────────────────────────────────────────────────
    results = []
    for bs in args.batch_sizes:
        loader = data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
        batches = []
        for img, latent, pred in loader:
            pred = pred.float().reshape(pred.size(0), -1)[:, :40]
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            batches.append((img.cuda(), latent.cuda(), pred.cuda()))
            if len(batches) * bs >= args.num_faces + args.warmup * bs:
                break
        for m, ed in editors.items():
            m_attrs = notes.get(m, attrs)
            for vname, vkw in (variants.items() if m in FLOW_METHODS else [('-', None)]):
                if vkw is not None:
                    desc = configure_inference(flows[m], **vkw)
                else:
                    desc = ''
                tm = Timer()
                nfe = []
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for k, (img, latent, pred) in enumerate(batches):
                        g = m_attrs[k % len(m_attrs)]
                        d = torch.where(pred[:, g] > 0.5, -1.0, 1.0)
                        sc = scale[m]
                        warm = k < args.warmup
                        key = 'warm_' if warm else ''
                        if isinstance(ed, tuple):                        # DeltaEdit renders itself
                            de = ed[1]
                            tm(key + 'total', lambda: de.edit(G, latent, de.image_feature(img), g, d, sc))
                            continue
                        if m == 'ours':
                            w, n = tm(key + 'latent', ed, img, latent, pred, g, d, sc, tm, key + 'part_')
                        else:
                            w, n = tm(key + 'latent', ed, img, latent, pred, g, d, sc)
                        if n and n[0] is not None and not warm:
                            nfe.append(n)
                        tm(key + 'G', lambda: G([w], input_is_latent=True, randomize_noise=False)[0])
                        if not warm and m == 'ours' and args.slider_steps > 1:
                            # slider: the source encoded ONCE, then decoded at several strengths
                            li = args.attribute_index.index(g)

                            def slider():
                                _, idc, ac = conditioner.make_condition(img, latent, id_criterion)
                                zp = torch.zeros(latent.size(0), 18, 1, device=latent.device)
                                z, _ = prior(latent, torch.cat([idc, ac], 1), zp)
                                for j in range(args.slider_steps):
                                    na = ac.clone()
                                    na[:, li] = edited_attr_value(ac[:, li], sc * (j + 1) / args.slider_steps, g,
                                                                  direction=d)
                                    prior(z, torch.cat([idc, na], 1), zp, reverse=True)
                            tm('slider', slider)
                peak = torch.cuda.max_memory_allocated() / 2 ** 20
                row = {'method': m, 'variant': vname, 'solver': desc, 'batch': bs,
                       'attrs': m_attrs, 'scale': scale[m], 'peak_mb': round(peak)}
                n_timed = len(batches) - args.warmup
                for key in ('latent', 'G', 'total', 'part_cond', 'part_encode', 'part_decode'):
                    if key in tm.t:
                        row[key] = summary(tm.t[key], bs)
                if 'latent' in tm.t and 'G' in tm.t:
                    row['total'] = summary([a + b for a, b in zip(tm.t['latent'], tm.t['G'])], bs)
                if nfe:
                    row['nfe_encode'] = statistics.mean(x[0] for x in nfe)
                    row['nfe_decode'] = statistics.mean(x[1] for x in nfe)
                if 'slider' in tm.t:
                    s = summary(tm.t['slider'], bs)
                    row['slider_ms_per_face'] = s['median_ms']
                    row['slider_note'] = f'conditioner + one encode + {args.slider_steps} decodes'
                row['timed_batches'] = n_timed
                results.append(row)
                tot = row.get('total', {}).get('median_ms', float('nan'))
                lat = row.get('latent', {}).get('median_ms', float('nan'))
                print(f'{m:12s} {vname:18s} batch {bs:2d}: latent {lat:8.2f} ms  total {tot:8.2f} ms/face'
                      + (f'  NFE {row["nfe_encode"]:.0f}+{row["nfe_decode"]:.0f}' if 'nfe_encode' in row else '')
                      + f'  peak {row["peak_mb"]} MB')
        # restore the trained solver for the next batch size's first method
        for f in flows.values():
            configure_inference(f, solver='dopri5', tol=1e-5, steps=None, skip_divergence=False)

    gpu = torch.cuda.get_device_name(0)
    meta = {'gpu': gpu, 'checkpoint_dir': args.checkpoint_dir, 'step': args.step, 'num_faces': args.num_faces,
            'warmup': args.warmup, 'torch': torch.__version__, 'variants': variants}
    with open(os.path.join(args.out_dir, 'speed.json'), 'w') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2, default=str)

    lines = [f'# Edit speed ({gpu}; {args.num_faces} faces per batch size, median over calls, ms per edited face)', '',
             '| method | ODE | batch | latent edit | render (G) | total | NFE enc+dec | slider: each extra strength | peak MB |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in results:
        f = lambda k: f'{r[k]["median_ms"]:.1f}' if k in r else '--'
        nfe = f'{r["nfe_encode"]:.0f}+{r["nfe_decode"]:.0f}' if 'nfe_encode' in r else '--'
        sl = '--'
        if 'slider_ms_per_face' in r:     # (conditioner + encode + K decodes) / K
            sl = f'{r["slider_ms_per_face"] / args.slider_steps:.1f}'
        lines.append(f'| {r["method"]} | {r["solver"] or r["variant"]} | {r["batch"]} | {f("latent")} | {f("G")} | '
                     f'{f("total")} | {nfe} | {sl} | {r["peak_mb"]} |')
    lines += ['', 'latent edit: source -> edited W+ (ours: conditioner + flow encode + decode + bank / cap). '
              'slider: one encode, then a decode per strength (latent only), per strength. DeltaEdit renders inside its '
              'edit: total only. Strengths: ' + ', '.join(f'{m} {scale[m]:g}' for m in editors) + '.']
    open(os.path.join(args.out_dir, 'speed.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print(f'\nsaved {args.out_dir}/speed.json and speed.md')


if __name__ == '__main__':
    main()
