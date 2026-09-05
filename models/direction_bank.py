import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(x):
    x = x.clamp(min=1e-6)
    return torch.log(torch.expm1(x))


class AttributeDirectionBank(nn.Module):
    """Dataset-level W+ attribute directions used to filter raw flow deltas.

    Each attribute stores K directions; a lightweight gate network learns
    per-sample mixture weights alpha_k(w, attr) conditioned on the source latent:
        guided_delta = sum_k alpha_k * magnitude * direction_k  +  residual_scale * residual

    With num_k=1 (default) behavior is identical to the original single-direction bank.
    Old bank files (direction_units.ndim==3) are loaded automatically as K=1.
    """

    def __init__(
        self,
        num_attrs=3,
        num_layers=18,
        latent_dim=512,
        num_k=1,
        bank_path=None,
        attribute_index=None,
        residual_scale=0.05,
        freeze_directions=True,
        per_attr_residual_scale=None,
        residual_max_norm=None,
        per_attr_direction_scale=None,
        per_attr_layer_scale=None,
        per_attr_delta_max_norm=None,
        guided_delta_max_norm=None,
        use_attr_lora=False,
        attr_lora_rank=4,
        signed_magnitude_input=False,
        gate_usage_ema_decay=0.98,
    ):
        super().__init__()
        self.num_attrs = int(num_attrs)
        self.num_layers = int(num_layers)
        self.latent_dim = int(latent_dim)
        self.num_k = int(num_k)

        direction_units = torch.zeros(self.num_attrs, self.num_k, self.num_layers, self.latent_dim)
        layer_norms = torch.ones(self.num_attrs, self.num_k, self.num_layers)

        if bank_path is not None:
            bank = torch.load(bank_path, map_location="cpu")
            du = bank["direction_units"].float()
            ln = bank["layer_norms"].float()

            # Backward compat: old format shape is (num_attrs, 18, 512)
            if du.ndim == 3:
                du = du.unsqueeze(1)   # -> (num_attrs, 1, 18, 512)
            if ln.ndim == 2:
                ln = ln.unsqueeze(1)   # -> (num_attrs, 1, 18)

            bank_attrs = [int(x) for x in bank.get("attribute_index", [])]
            if attribute_index is not None and bank_attrs:
                wanted = [int(x) for x in attribute_index]
                order = [bank_attrs.index(x) for x in wanted]
                du = du[order]
                ln = ln[order]

            bank_k = du.shape[1]
            if bank_k < self.num_k:
                # tile and add small noise to break symmetry among copies
                reps = (self.num_k + bank_k - 1) // bank_k
                du = du.repeat(1, reps, 1, 1)[:, :self.num_k]
                ln = ln.repeat(1, reps, 1)[:, :self.num_k]
                noise = torch.randn_like(du) * 0.01
                du = du + noise
                print(f"[DirectionBank] bank K={bank_k} tiled to requested K={self.num_k}")
            elif bank_k > self.num_k:
                du = du[:, :self.num_k]
                ln = ln[:, :self.num_k]
                print(f"[DirectionBank] bank K={bank_k} truncated to K={self.num_k}")

            direction_units = du
            layer_norms = ln
            print(f"[DirectionBank] loaded from {bank_path}")

        expected_du = (self.num_attrs, self.num_k, self.num_layers, self.latent_dim)
        if tuple(direction_units.shape) != expected_du:
            raise ValueError(
                f"direction_units must have shape {expected_du}, got {tuple(direction_units.shape)}"
            )
        expected_ln = (self.num_attrs, self.num_k, self.num_layers)
        if tuple(layer_norms.shape) != expected_ln:
            raise ValueError(
                f"layer_norms must have shape {expected_ln}, got {tuple(layer_norms.shape)}"
            )

        direction_units = F.normalize(direction_units, dim=-1, eps=1e-8)
        if freeze_directions:
            self.register_buffer("direction_units", direction_units)
            print("[DirectionBank] directions frozen")
        else:
            self.direction_units = nn.Parameter(direction_units)
            print("[DirectionBank] directions trainable")

        # magnitude_net: same interface as before; initialized from mean norm across K
        self.magnitude_net = nn.Sequential(
            nn.Linear(self.num_attrs, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_attrs * self.num_layers),
        )
        with torch.no_grad():
            prior_norms = layer_norms.mean(dim=1).clamp(min=1e-4)   # (num_attrs, 18)
            self.magnitude_net[-1].bias.copy_(_inverse_softplus(prior_norms.reshape(-1)))
            self.magnitude_net[-1].weight.mul_(0.01)

        # Optional per-attribute LoRA-style adapter on top of magnitude_net's
        # shared hidden representation. magnitude_net is ONE MLP shared across
        # all attributes; this gives each attribute its own small low-rank
        # correction (A: hidden->rank, B: rank->num_layers) instead of only a
        # shared path, without training a separate full network per attribute.
        # B is zero-initialized, so this is a strict no-op until trained --
        # safe to turn on for a bank that already has a trained magnitude_net.
        # Feeding magnitude_net the SIGNED attr_delta lets an attribute use a
        # different step size for adding vs removing it. With .abs() the two
        # are forced to the same magnitude, and only the final sign differs --
        # so "make masculine" and "make feminine" must travel exactly as far.
        # A scale sweep on a trained model shows that assumption is wrong: the
        # strong directions saturate while the weak ones keep improving with
        # more displacement (AccCeleb at edit_scale 1.0 / 1.2 / 1.4):
        #
        #   Eyeglasses add  91.5  91.5  92.4    saturated
        #   Eyeglasses rm   98.4  98.4  98.4    saturated
        #   Male       add  86.8  91.4  93.4    flattening
        #   Male       rm   40.0  50.4  57.8    still climbing
        #   Young      add  58.5  63.4  68.3    still climbing
        #   Young      rm   90.9  94.9  98.3    near ceiling
        #
        # Which side is weak differs per attribute (Male on rm, Young on add),
        # so a global edit_scale increase overpays: it buys nothing on the
        # saturated directions while spending identity on every sample. Making
        # the magnitude signed moves that compensation inside the model, so
        # edit_scale can stay at 1.0.
        self.signed_magnitude_input = bool(signed_magnitude_input)
        self.use_attr_lora = bool(use_attr_lora)
        if self.use_attr_lora:
            rank = int(attr_lora_rank)
            hidden_dim = self.magnitude_net[0].out_features   # 64
            self.attr_lora_A = nn.Parameter(torch.randn(self.num_attrs, rank, hidden_dim) * 0.01)
            self.attr_lora_B = nn.Parameter(torch.zeros(self.num_attrs, self.num_layers, rank))
            print(f"[DirectionBank] per-attribute LoRA adapter enabled (rank={rank})")

        # gate_net: learns per-sample mixture weights over K directions (only when K>1)
        if self.num_k > 1:
            self.gate_net = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.num_attrs * self.num_k),
            )
            # init near-uniform: zero bias -> softmax gives 1/K for all components
            with torch.no_grad():
                self.gate_net[-1].weight.mul_(0.01)
                self.gate_net[-1].bias.zero_()
        else:
            self.gate_net = None

        # gate_usage_ema: (num_attrs, num_k) running average of how much each
        # K-slot actually gets used, per attribute -- feeds both gate_load_balance_loss()
        # and the per-attribute entropy wandb logs. WHY AN EMA INSTEAD OF THE RAW
        # PER-STEP BATCH MEAN: nothing in this project's loss previously supervised
        # gate_net's routing AT ALL -- it only ever got gradient indirectly through
        # the final guided_delta, with no signal rewarding correct (demographic-
        # appropriate) OR even diverse routing. Measured on a real trained checkpoint
        # (eyeglasses, K=12 from --K 4 x --substyle_k 3): the gate collapsed within
        # the first few thousand steps onto ~2 of the 12 slots for 81% of samples,
        # regardless of the source face's actual gender/age -- including almost NEVER
        # routing female_young samples to the female_young-conditioned slots that
        # --extreme_min_conf specifically cleaned up, which is why that direction-bank
        # fix alone did not move the eyeglasses-add failure rate. A single training
        # step's batch (e.g. --batch 4) spread over K=12 slots is far too noisy to
        # regularize directly -- an EMA smooths that out across many steps, matching
        # the pattern this project already uses for --balance_ema_decay.
        # persistent=False: this is a training-time running stat, not part of the
        # frozen/learned state a checkpoint needs to reproduce inference -- excluding
        # it from state_dict avoids key-mismatch noise when eval scripts load_state_dict
        # with strict=False anyway.
        if self.num_k > 1:
            self.register_buffer(
                "gate_usage_ema",
                torch.full((self.num_attrs, self.num_k), 1.0 / self.num_k),
                persistent=False,
            )
        else:
            self.gate_usage_ema = None
        self.gate_usage_ema_decay = float(gate_usage_ema_decay)

        # residual_scale: (num_attrs,) — learned, not hand-tuned. per_attr_residual_scale
        # (or the scalar residual_scale) only sets the *initial* value; gradient descent
        # on the normal training losses (id/reg/changed/preserve) decides from there how
        # much each attribute should trust the flow's own local freedom (residual) vs the
        # dataset-level direction. Softplus reparam keeps it positive with no artificial
        # upper bound -- guided_delta_max_norm and reg_loss are the safety rails against
        # runaway growth, not a per-attribute cap decided in advance.
        if per_attr_residual_scale is not None:
            scales = torch.tensor([float(s) for s in per_attr_residual_scale], dtype=torch.float)
            if scales.shape[0] != self.num_attrs:
                raise ValueError(
                    f"per_attr_residual_scale length {scales.shape[0]} != num_attrs {self.num_attrs}"
                )
        else:
            scales = torch.full((self.num_attrs,), float(residual_scale))
        self.residual_scale_raw = nn.Parameter(_inverse_softplus(scales))   # (num_attrs,)
        self.residual_max_norm = float(residual_max_norm) if residual_max_norm is not None else None

        # Optional safety controls applied to the final guided delta. These are
        # especially useful for Age/Young, whose dataset-level direction can
        # contain broad texture and identity changes when it is applied to all
        # W+ layers at full strength.
        if per_attr_direction_scale is not None:
            direction_scale = torch.tensor(
                [float(s) for s in per_attr_direction_scale],
                dtype=torch.float,
            )
            if direction_scale.shape[0] != self.num_attrs:
                raise ValueError(
                    "per_attr_direction_scale length "
                    f"{direction_scale.shape[0]} != num_attrs {self.num_attrs}"
                )
        else:
            direction_scale = torch.ones(self.num_attrs, dtype=torch.float)
        self.register_buffer("direction_scale", direction_scale)

        if per_attr_layer_scale is not None:
            layer_scale = torch.tensor(per_attr_layer_scale, dtype=torch.float)
            if tuple(layer_scale.shape) != (self.num_attrs, self.num_layers):
                raise ValueError(
                    "per_attr_layer_scale must have shape "
                    f"({self.num_attrs}, {self.num_layers}), got {tuple(layer_scale.shape)}"
                )
        else:
            layer_scale = torch.ones(self.num_attrs, self.num_layers, dtype=torch.float)
        self.register_buffer("layer_scale", layer_scale)

        if per_attr_delta_max_norm is not None:
            delta_max_norm = torch.tensor(
                [float(v) for v in per_attr_delta_max_norm],
                dtype=torch.float,
            )
            if delta_max_norm.shape[0] != self.num_attrs:
                raise ValueError(
                    "per_attr_delta_max_norm length "
                    f"{delta_max_norm.shape[0]} != num_attrs {self.num_attrs}"
                )
        else:
            delta_max_norm = torch.zeros(self.num_attrs, dtype=torch.float)
        self.register_buffer("delta_max_norm", delta_max_norm)
        self.guided_delta_max_norm = (
            float(guided_delta_max_norm) if guided_delta_max_norm is not None else None
        )
        self.last_logs = {}
        self._last_alpha = None   # (B, num_attrs, K) — set each forward, used for selection loss

    def current_residual_scale(self):
        """(num_attrs,) learned residual_scale values, always positive."""
        return F.softplus(self.residual_scale_raw)

    def _gate_weights(self, latent, B, device, dtype):
        """Return (B, num_attrs, K) softmax mixture weights."""
        if self.num_k == 1:
            return self.direction_units.new_ones(B, self.num_attrs, 1)
        if latent is not None and self.gate_net is not None:
            w = latent.mean(dim=1).to(device=device, dtype=dtype)        # (B, 512)
            logits = self.gate_net(w)                                      # (B, A*K)
            return F.softmax(logits.view(B, self.num_attrs, self.num_k), dim=-1)
        return self.direction_units.new_ones(B, self.num_attrs, self.num_k) / self.num_k

    def forward(self, flow_delta, attr_delta, attr_idx=None, latent=None):
        B = flow_delta.size(0)
        device = flow_delta.device
        dtype = flow_delta.dtype
        dirs = self.direction_units.to(device=device, dtype=dtype)   # (A, K, 18, 512)
        attr_delta = attr_delta.to(device=device, dtype=dtype)

        # ── Magnitudes ────────────────────────────────────────────────────
        # Signed input -> add and rm can learn different magnitudes; .abs()
        # forces them equal. The sign of the edit still comes from attr_delta
        # below either way, so this only affects how far each side travels.
        mag_input = attr_delta if self.signed_magnitude_input else attr_delta.abs()
        mag_hidden = torch.tanh(self.magnitude_net[0](mag_input))          # (B, 64)
        mag_logits = self.magnitude_net[2](mag_hidden)                     # (B, A*L)
        mag_logits = mag_logits.view(B, self.num_attrs, self.num_layers)

        if self.use_attr_lora and attr_idx is not None:
            attr_idx_long = attr_idx.view(-1).long()
            A_sel = self.attr_lora_A[attr_idx_long].to(device=device, dtype=dtype)   # (B, rank, 64)
            B_sel = self.attr_lora_B[attr_idx_long].to(device=device, dtype=dtype)   # (B, L, rank)
            lora_h = torch.einsum('brh,bh->br', A_sel, mag_hidden)                    # (B, rank)
            lora_delta = torch.einsum('blr,br->bl', B_sel, lora_h)                    # (B, L)
            mag_logits = mag_logits.scatter_add(
                1, attr_idx_long.view(-1, 1, 1).expand(-1, 1, self.num_layers),
                lora_delta.unsqueeze(1),
            )

        magnitudes = F.softplus(mag_logits)

        if attr_idx is not None:
            mask = torch.zeros(B, self.num_attrs, device=device, dtype=dtype)
            mask.scatter_(1, attr_idx.view(-1, 1).long(), 1.0)
            magnitudes = magnitudes * mask.unsqueeze(-1)

        signed_magnitudes = magnitudes * attr_delta.unsqueeze(-1)    # (B, A, 18)

        # ── Gate mixture ──────────────────────────────────────────────────
        alpha = self._gate_weights(latent, B, device, dtype)          # (B, A, K)
        self._last_alpha = alpha                                        # expose for selection loss

        # Update the per-attribute gate usage EMA from THIS batch's active
        # attribute(s) only -- alpha for an attribute other than the one(s)
        # actually being edited this step never receives gradient (its
        # signed_magnitudes are masked to exactly 0 below, see forward()'s
        # docstring on gate_usage_ema), so folding it into the EMA would
        # just average in untrained noise.
        #
        # gate_usage_ema itself is updated under no_grad -- it is a plain
        # buffer (requires_grad=False), used ONLY for the dir_gate_entropy_per_attr
        # LOG (a smoothed, low-variance number to look at, not a value gradients
        # ever need to flow through). self._last_gate_diversity_loss below is a
        # SEPARATE, genuinely differentiable quantity computed from this same
        # step's live `alpha` (no detach) -- that is the one train_sdflow.py's
        # --dir_gate_diversity_weight actually optimizes. An earlier version of
        # this method computed the trained loss FROM gate_usage_ema directly,
        # which silently contributed ZERO gradient (the buffer has no grad_fn),
        # making --dir_gate_diversity_weight a complete no-op -- confirmed by
        # training a real run with it that showed no change in gate collapse
        # behavior traceable to this loss. Fixed here; the EMA is for display only.
        if self.num_k > 1 and attr_idx is not None:
            attr_idx_long = attr_idx.view(-1).long()
            div_losses = []
            for a in attr_idx_long.unique():
                m = attr_idx_long == a
                batch_mean = alpha[m, a, :].mean(dim=0)   # (K,) -- LIVE, keeps gradient
                if self.training:
                    with torch.no_grad():
                        self.gate_usage_ema[a].mul_(self.gate_usage_ema_decay).add_(
                            batch_mean.detach(), alpha=1.0 - self.gate_usage_ema_decay
                        )
                p = batch_mean.clamp(min=1e-8)
                p = p / p.sum()
                entropy = -(p * p.log()).sum()
                max_entropy = math.log(self.num_k)
                div_losses.append((max_entropy - entropy) / max_entropy)
            self._last_gate_diversity_loss = torch.stack(div_losses).mean()
        else:
            self._last_gate_diversity_loss = torch.zeros([], device=device, dtype=dtype)

        # mix_dirs: weighted sum of K direction vectors per attribute
        mix_dirs = (alpha.unsqueeze(-1).unsqueeze(-1)                  # (B, A, K, 1, 1)
                    * dirs.unsqueeze(0)).sum(dim=2)                    # (B, A, 18, 512)

        # dir_delta: (B, 18, 512)
        dir_delta = (signed_magnitudes.unsqueeze(-1) * mix_dirs).sum(dim=1)

        # ── Residual orthogonal to the FULL span of all A*K directions ────
        # Proper orthogonal-complement projection via pseudo-inverse, replacing
        # the old sequential Gram-Schmidt subtraction: subtracting projections
        # one direction at a time is only correct when the directions are
        # mutually orthogonal, which A*K dataset-level directions are not.
        # The old version left order-dependent direction components inside the
        # residual, so "residual" silently double-counted the bank directions
        # (worse for stratified K>1 banks). pinv handles rank deficiency and
        # zero/duplicate directions safely.
        M = self.num_attrs * self.num_k
        # (A, K, 18, 512) -> per-layer direction matrix D: (18, 512, M)
        D = dirs.reshape(M, self.num_layers, self.latent_dim).permute(1, 2, 0)
        pinv_D = torch.linalg.pinv(D)                                  # (18, M, 512)
        coeff = torch.einsum('lms,bls->blm', pinv_D, flow_delta)       # (B, 18, M)
        proj = torch.einsum('lsm,blm->bls', D, coeff)                  # (B, 18, 512)
        residual = flow_delta - proj

        # Clip per-sample residual norm to prevent explosion from large DDS gradients.
        if self.residual_max_norm is not None:
            r_norm = residual.reshape(B, -1).norm(dim=1)               # (B,)
            clip = (self.residual_max_norm / r_norm.clamp(min=1e-8)).clamp(max=1.0)
            residual = residual * clip.view(B, 1, 1)

        # Per-attribute residual scale, learned via gradient descent. Gathered per-sample
        # (not just attr_idx[0]) so mixed-attribute batches from --attribute_sampling
        # random are handled correctly, not only the cycle-mode case where every sample
        # in a batch shares one attribute.
        scales = self.current_residual_scale().to(device=device, dtype=dtype)   # (num_attrs,)
        if attr_idx is not None:
            attr_idx_long = attr_idx.view(-1).long()
            rs = scales[attr_idx_long].view(B, 1, 1)
        else:
            rs = scales.mean()
        guided_delta = dir_delta + rs * residual

        guided_delta_pre_clip = guided_delta
        active_direction_scale = self.direction_scale.to(device=device, dtype=dtype).mean()
        active_delta_max_norm = torch.zeros([], device=device, dtype=dtype).detach()
        active_global_delta_max_norm = torch.zeros([], device=device, dtype=dtype).detach()
        if attr_idx is not None:
            attr_idx_long = attr_idx.view(-1).long()
            direction_scale = self.direction_scale.to(device=device, dtype=dtype)
            layer_scale = self.layer_scale.to(device=device, dtype=dtype)
            delta_max_norm = self.delta_max_norm.to(device=device, dtype=dtype)

            active_direction_scale = direction_scale[attr_idx_long].mean()
            active_delta_max_norm = delta_max_norm[attr_idx_long].mean()
            guided_delta = guided_delta * direction_scale[attr_idx_long].view(B, 1, 1)
            guided_delta = guided_delta * layer_scale[attr_idx_long].view(B, self.num_layers, 1)

            max_norm = delta_max_norm[attr_idx_long]
            if (max_norm > 0).any():
                g_norm = guided_delta.reshape(B, -1).norm(dim=1)
                clip = torch.ones_like(g_norm)
                capped = max_norm > 0
                clip[capped] = (
                    max_norm[capped] / g_norm[capped].clamp(min=1e-8)
                ).clamp(max=1.0)
                guided_delta = guided_delta * clip.view(B, 1, 1)
        else:
            guided_delta = guided_delta * active_direction_scale

        # Optional global final cap. This is intentionally applied after all
        # per-attribute/layer controls as a last safety rail for stage fine-tuning.
        if self.guided_delta_max_norm is not None and self.guided_delta_max_norm > 0:
            g_norm = guided_delta.reshape(B, -1).norm(dim=1)
            clip = (self.guided_delta_max_norm / g_norm.clamp(min=1e-8)).clamp(max=1.0)
            guided_delta = guided_delta * clip.view(B, 1, 1)
            active_global_delta_max_norm = torch.tensor(float(self.guided_delta_max_norm), device=device, dtype=dtype).detach()

        # ── Logging ───────────────────────────────────────────────────────
        with torch.no_grad():
            flow_norm = flow_delta.reshape(B, -1).norm(dim=1).mean()
            dir_norm = dir_delta.reshape(B, -1).norm(dim=1).mean()
            residual_norm = residual.reshape(B, -1).norm(dim=1).mean()
            guided_pre_clip_norm = guided_delta_pre_clip.reshape(B, -1).norm(dim=1).mean()
            guided_norm = guided_delta.reshape(B, -1).norm(dim=1).mean()
            logs = {
                "dir_bank_flow_delta_norm": flow_norm.detach(),
                "dir_bank_dir_delta_norm": dir_norm.detach(),
                "dir_bank_residual_norm": residual_norm.detach(),
                "dir_bank_guided_delta_norm_pre_clip": guided_pre_clip_norm.detach(),
                "dir_bank_guided_delta_norm": guided_norm.detach(),
                "dir_bank_residual_scale": scales.detach(),
                "dir_bank_active_direction_scale": active_direction_scale.detach(),
                "dir_bank_active_delta_max_norm": active_delta_max_norm.detach(),
                "dir_bank_global_delta_max_norm": active_global_delta_max_norm.detach(),
            }
            if self.num_k > 1:
                # Raw current-step entropy, kept for backward compat -- averages
                # over EVERY attribute row including ones not active this step
                # (their alpha is untrained noise, see gate_usage_ema comment
                # above), so this number is noisier and less meaningful than the
                # per-attribute EMA entropy below. Prefer dir_gate_entropy_per_attr.
                entropy = -(alpha * (alpha + 1e-8).log()).sum(dim=-1).mean()
                logs["dir_gate_entropy"] = entropy.detach()
                # Per-attribute entropy computed from the EMA usage vector, not
                # this step's raw alpha -- see gate_usage_ema for why. Keyed by
                # LOCAL attribute row index; train_sdflow.py maps this to the
                # actual CelebA attribute id for wandb.
                p = self.gate_usage_ema.clamp(min=1e-8)
                p = p / p.sum(dim=-1, keepdim=True)
                ema_entropy = -(p * p.log()).sum(dim=-1)   # (num_attrs,)
                logs["dir_gate_entropy_per_attr"] = ema_entropy.detach()
            self.last_logs = logs

        return guided_delta

    def orthogonality_loss(self):
        """Cross-attribute orthogonality, averaged over all K combinations."""
        dirs = self.direction_units   # (A, K, 18, 512)
        loss = torch.zeros([], device=dirs.device, dtype=dirs.dtype)
        count = 0
        for i in range(self.num_attrs):
            for j in range(i + 1, self.num_attrs):
                for ki in range(self.num_k):
                    for kj in range(self.num_k):
                        loss = loss + F.cosine_similarity(
                            dirs[i, ki], dirs[j, kj], dim=-1
                        ).abs().mean()
                        count += 1
        return loss / max(count, 1)

    def diversity_loss(self):
        """Intra-attribute diversity: penalize high cosine similarity among the K
        directions belonging to the same attribute."""
        if self.num_k <= 1:
            return torch.zeros([], device=self.direction_units.device, dtype=self.direction_units.dtype)
        dirs = self.direction_units   # (A, K, 18, 512)
        loss = torch.zeros([], device=dirs.device, dtype=dirs.dtype)
        count = 0
        for i in range(self.num_attrs):
            for ki in range(self.num_k):
                for kj in range(ki + 1, self.num_k):
                    loss = loss + F.cosine_similarity(
                        dirs[i, ki], dirs[i, kj], dim=-1
                    ).abs().mean()
                    count += 1
        return loss / max(count, 1)

    def gate_load_balance_loss(self):
        """Batch-level load-balancing loss for the K-mixture gate (Shazeer-style
        importance loss). Returns the value computed in the MOST RECENT
        forward() call (self._last_gate_diversity_loss) -- see that computation
        for why it must be built from THIS step's live `alpha`, not from
        gate_usage_ema (a plain buffer with no gradient; using it directly here
        was an earlier bug that made --dir_gate_diversity_weight a silent
        no-op). Call this AFTER calling direction_bank(...) in the same step.

        Returns, per attribute active in the last forward() call, how far that
        attribute's per-batch usage distribution sits from uniform (0 =
        perfectly uniform, 1 = fully collapsed onto one slot), averaged over
        whichever attribute(s) were active -- safe to add into the loss every
        step regardless of --attribute_sampling mode.

        WHAT THIS DOES NOT DO: this has no notion of which slot is "correct"
        for a given face (that would need a demographic label fed in as a
        target) -- it only discourages the gate from collapsing onto a
        minority of slots. Necessary, not sufficient, for the geometry a
        stratum-level fix like --extreme_min_conf produces to actually get
        used by the model.
        """
        if self.num_k <= 1:
            return torch.zeros([], device=self.direction_units.device, dtype=self.direction_units.dtype)
        loss = getattr(self, '_last_gate_diversity_loss', None)
        if loss is None:
            return torch.zeros([], device=self.direction_units.device, dtype=self.direction_units.dtype)
        return loss
