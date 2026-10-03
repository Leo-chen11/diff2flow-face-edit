"""Smoke-test multi-resolution ControlNet injection before spending a training run.

The one thing that can silently go wrong with feature-map injection is a
channel/resolution mismatch: Generator.forward() does a bare `out = out +
skips[res]`, so a width that doesn't match the generator's own at that
resolution either raises deep inside the generator or, worse, broadcasts
into something that trains but is meaningless. This checks, on the real
Generator:

  1. the encoder emits a tensor at every requested resolution, with the
     exact channel count the generator has there;
  2. the generator accepts them and the image actually CHANGES (a silent
     no-op is the failure mode the single-resolution version shipped with
     for months -- see AttributeControlEncoder's docstring on the inert
     zero-init stack);
  3. each band's contribution is separable, so you can see which
     resolution is doing the work;
  4. what the parameter count and activation memory actually cost.

Usage (from the project root):
    python scripts/check_controlnet_injection.py --controlnet_res 64 128
    python scripts/check_controlnet_injection.py --controlnet_res 64 128 256 --batch 4
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.control_encoder import (          # noqa: E402
    AttributeControlEncoder, add_skips, clip_skips, skips_norm_per_sample,
    stylegan2_channels,
)
from models.stylegan2.model import Generator  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--controlnet_res', nargs='*', type=int, default=[64, 128])
    p.add_argument('--num_attrs', type=int, default=3)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--per_direction', action='store_true', default=True)
    p.add_argument('--no-per_direction', dest='per_direction', action='store_false')
    p.add_argument('--latent_cond', action='store_true', default=True)
    p.add_argument('--no-latent_cond', dest='latent_cond', action='store_false')
    p.add_argument('--controlnet_init_gain', type=float, default=1.0)
    p.add_argument('--controlnet_max_norm', type=float, default=0.0)
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    B = args.batch
    chan = stylegan2_channels()

    print(f'=== Building encoder: res={args.controlnet_res}, '
          f'per_direction={args.per_direction}, latent_cond={args.latent_cond} ===')
    enc = AttributeControlEncoder(
        num_attrs=args.num_attrs,
        out_res=args.controlnet_res,
        per_direction=args.per_direction,
        latent_cond=args.latent_cond,
        init_gain=args.controlnet_init_gain,
    ).to(device)

    n_params = sum(p_.numel() for p_ in enc.parameters())
    print(f'  slots={enc.num_slots}  parameters={n_params/1e6:.1f}M')
    print(f'  channel width per band: '
          + ', '.join(f'{r}->{c}ch' for r, c in enc.res_channels.items()))

    # ── 1. shapes match the generator's own feature widths ────────────────
    attr_delta = torch.zeros(B, args.num_attrs, device=device)
    attr_idx = torch.arange(B, device=device) % args.num_attrs
    is_rm = torch.arange(B, device=device) % 2 == 0
    attr_delta[torch.arange(B), attr_idx] = torch.where(is_rm, -0.8, 0.8)
    latent = torch.randn(B, 18, 512, device=device)

    skips = enc(attr_delta, attr_idx, is_rm=is_rm, latent=latent)
    print('\n=== 1. emitted bands ===')
    ok = True
    for r in sorted(skips):
        t = skips[r]
        want = (B, chan[r], r, r)
        good = tuple(t.shape) == want
        ok &= good
        print(f'  {r:>4}x{r:<4} shape={tuple(t.shape)} '
              f'{"OK" if good else f"MISMATCH, generator has {want}"}  '
              f'norm/sample={t.reshape(B, -1).norm(dim=1).mean():.4f}')
    if not ok:
        raise SystemExit('FAILED: a band does not match the generator width; '
                         'the addition inside Generator.forward() would break.')
    print(f'  combined norm/sample = {skips_norm_per_sample(skips).mean():.4f} '
          f'(this is what --controlnet_reg_weight penalizes)')

    # helpers keep their shape contract
    clipped = clip_skips(skips, 0.5)
    assert set(clipped) == set(skips)
    doubled = add_skips(skips, skips)
    assert all(torch.allclose(doubled[r], skips[r] * 2) for r in skips)
    print('  clip_skips / add_skips: OK')

    # ── 2. the generator accepts them AND the image changes ───────────────
    print('\n=== 2. generator round-trip ===')
    G = Generator(size=args.size, style_dim=512, n_mlp=8).to(device).eval()
    for p_ in G.parameters():
        p_.requires_grad_(False)

    with torch.no_grad():
        base = G([latent], skips=None, input_is_latent=True, randomize_noise=False)[0]
        # Scale the injection up so the effect is visible against an
        # untrained generator; init_gain=1 against a feature map whose own
        # norm is in the thousands is deliberately near-invisible at step 0.
        loud = {r: t * 200.0 for r, t in skips.items()}
        edited = G([latent], skips=loud, input_is_latent=True, randomize_noise=False)[0]
    delta = (edited - base).abs().mean().item()
    print(f'  image shape {tuple(base.shape)}, mean |change| = {delta:.6f}')
    if delta == 0.0:
        raise SystemExit('FAILED: injection is a no-op -- the generator never added it.')
    print('  injection reaches the image: OK')

    # ── 3. per-band contribution, separately ──────────────────────────────
    print('\n=== 3. per-band contribution (each band alone, gain x200) ===')
    for r in sorted(skips):
        with torch.no_grad():
            one = G([latent], skips={r: loud[r]}, input_is_latent=True,
                    randomize_noise=False)[0]
        d = (one - base).abs().mean().item()
        print(f'  {r:>4}x{r:<4} alone -> mean |change| = {d:.6f}')
    print('  a band showing 0.000000 never reaches the image -- check that the '
          'generator actually produces that resolution.')

    # ── 4. gradients flow to every band ───────────────────────────────────
    print('\n=== 4. gradient flow ===')
    enc.zero_grad(set_to_none=True)
    skips_g = enc(attr_delta, attr_idx, is_rm=is_rm, latent=latent)
    out = G([latent], skips=skips_g, input_is_latent=True, randomize_noise=False)[0]
    out.mean().backward()
    gain_grad = enc.log_gain.grad
    if gain_grad is None:
        raise SystemExit('FAILED: no gradient reached log_gain.')
    for i, r in enumerate(enc.out_res):
        g = gain_grad[:, i].abs().sum().item()
        print(f'  {r:>4}x{r:<4} log_gain |grad| = {g:.3e} '
              f'{"" if g > 0 else "  <-- DEAD, this band will never learn"}')
    head_grads = [
        (r, sum(p_.grad.abs().sum().item()
                for p_ in enc.heads[s][i].parameters() if p_.grad is not None))
        for i, r in enumerate(enc.out_res) for s in range(enc.num_slots)
    ]
    live = sum(1 for _, g in head_grads if g > 0)
    print(f'  output convs receiving gradient: {live}/{len(head_grads)} '
          f'(only slots present in this batch should be live: '
          f'{sorted(set((attr_idx * 2 + is_rm.long()).tolist())) if args.per_direction else sorted(set(attr_idx.tolist()))})')

    if device.type == 'cuda':
        print(f'\n=== 5. memory ===')
        print(f'  peak allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB '
              f'(batch {B}, generator {args.size}px, no SD/CLIP/ArcFace loaded)')

    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
