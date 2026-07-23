import torch
import torch.nn as nn
import torch.nn.functional as F


# CelebA attribute index → (positive prompt, negative prompt)
_ATTR_PROMPTS = {
    15: (
        "a face wearing eyeglasses",
        "a face without eyeglasses",
    ),
    20: (
        "a face of a male person",
        "a face of a female person",
    ),
    31: (
        "a smiling face",
        "a face with a neutral expression",
    ),
    33: (
        "a face with wavy hair",
        "a face with straight hair",
    ),
    39: (
        "a realistic face photo of a young adult with smooth skin and youthful facial features",
        "a realistic face photo of an older adult with wrinkles, aged skin texture, nasolabial folds, and mature facial features",
    ),
}

_GENERIC_PROMPTS = ("a face photo", "a distorted face photo")

_CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

_warned_unsupported: set = set()


class FrozenCLIPPromptLoss(nn.Module):
    """Frozen CLIP semantic loss for attribute editing, with two modes.

    'absolute' (original): pulls the EDITED image's CLIP embedding toward the
    positive/negative text prompt, independent of the source image:
        score = cos(clip_img_edit, clip_pos) - cos(clip_img_edit, clip_neg)
        loss  = softplus(-direction * score / temperature)
    This only constrains WHERE the edited image ends up in CLIP space. It has
    no opinion on the PATH taken to get there, so the model is free to reach
    a high "looks old" score via any correlated shortcut CLIP responds to --
    e.g. a warm/red color shift -- instead of genuine structural aging. A
    visual audit (scripts/dump_attr_failures.py, attr 39 direction rm) found
    exactly this: successful "aging" edits were almost always accompanied by
    a red/orange color-cast, while edits without the color shift mostly
    stayed visually young despite the loss pushing on them.

    'directional' (StyleGAN-NADA style): constrains the DIRECTION of change
    instead of the destination:
        img_delta  = clip_img_edit - clip_img_src         (normalized)
        text_delta = clip_pos - clip_neg                   (normalized)
        loss = 1 - cos(img_delta, direction * text_delta)
    This only rewards moving the image along the SAME axis in CLIP space that
    separates the pos/neg prompts, from THIS image's own starting point. A
    shortcut like a uniform color shift moves every image by roughly the same
    delta regardless of its content, which is much less likely to align with
    the pos/neg text axis than a real structural change specific to that
    face -- so directional mode is intended to close off the shortcut rather
    than just reward reaching the destination however possible.

    Cost: directional mode needs a source-image CLIP encode in addition to
    the edited-image encode (one extra forward through the CLIP image
    encoder per step).
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        image_size: int = 224,
        temperature: float = 1.0,
        mode: str = "absolute",
    ):
        super().__init__()
        if mode not in ("absolute", "directional"):
            raise ValueError(f"Unknown FrozenCLIPPromptLoss mode: {mode}")
        self.mode = mode
        try:
            import clip as openai_clip
        except ImportError as exc:
            raise ImportError(
                "OpenAI CLIP is required for FrozenCLIPPromptLoss.\n"
                "Install with:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        model, _ = openai_clip.load(clip_model, device="cpu", jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.clip = model
        self.image_size = int(image_size)
        self.temperature = float(temperature)
        self._tokenize = openai_clip.tokenize

        # Register CLIP normalisation constants as buffers so .cuda() moves them.
        self.register_buffer(
            "clip_mean",
            torch.tensor(_CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(_CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

        # Pre-encode all known attribute prompts and cache as buffers.
        self._build_text_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_text_cache(self):
        """Encode all prompt strings once and store as named buffers."""
        all_prompts = {}
        for attr_idx, (pos, neg) in _ATTR_PROMPTS.items():
            all_prompts[attr_idx] = (pos, neg)
        gen_pos, gen_neg = _GENERIC_PROMPTS

        flat_texts = []
        self._prompt_order = []   # list of (attr_idx_or_"generic", "pos"/"neg")
        for attr_idx, (pos, neg) in all_prompts.items():
            flat_texts.extend([pos, neg])
            self._prompt_order.extend([(attr_idx, "pos"), (attr_idx, "neg")])
        flat_texts.extend([gen_pos, gen_neg])
        self._prompt_order.extend([("generic", "pos"), ("generic", "neg")])

        tokens = self._tokenize(flat_texts)  # (N, context_length)
        with torch.no_grad():
            feats = self.clip.encode_text(tokens)   # (N, D)
            feats = F.normalize(feats.float(), dim=-1)

        # Store each as a separate buffer with a safe name.
        self._text_buf_names: dict = {}   # (attr_idx_or_"generic", "pos"/"neg") → buffer name
        for i, key in enumerate(self._prompt_order):
            attr_str = str(key[0]).replace("-", "_")
            buf_name = f"text_{attr_str}_{key[1]}"
            self.register_buffer(buf_name, feats[i])
            self._text_buf_names[key] = buf_name

    def _get_text_feat(self, attr_idx_int: int, polarity: str) -> torch.Tensor:
        """Return cached (D,) text feature for given attr and polarity."""
        key = (attr_idx_int, polarity)
        if key not in self._text_buf_names:
            global _warned_unsupported
            if attr_idx_int not in _warned_unsupported:
                print(
                    f"[FrozenCLIPPromptLoss] warning: attr {attr_idx_int} not in prompt dict, "
                    f"using generic prompts."
                )
                _warned_unsupported.add(attr_idx_int)
            key = ("generic", polarity)
        return getattr(self, self._text_buf_names[key])

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """[-1,1] BCHW → CLIP-normalised 224×224."""
        x = images.clamp(-1.0, 1.0)
        x = (x + 1.0) * 0.5                                         # [0, 1]
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x, (self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        x = (x - self.clip_mean) / self.clip_std
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_images(images)
        feats = self.clip.encode_image(x.to(dtype=self.clip_mean.dtype))
        return F.normalize(feats.float(), dim=-1)   # (B, D)

    def _text_deltas(self, attr_abs_idx: torch.Tensor, device):
        """Per-sample (pos, neg) text features gathered by attribute id."""
        attr_list = attr_abs_idx.detach().cpu().tolist()
        pos_feats = torch.stack(
            [self._get_text_feat(int(a), "pos") for a in attr_list], dim=0
        ).to(device)
        neg_feats = torch.stack(
            [self._get_text_feat(int(a), "neg") for a in attr_list], dim=0
        ).to(device)
        return pos_feats, neg_feats

    def forward(
        self,
        images: torch.Tensor,
        attr_abs_idx: torch.Tensor,
        target_values: torch.Tensor,
        reduction: str = "none",
        src_images: torch.Tensor = None,
    ):
        """
        Args:
            images:        (B, 3, H, W)  range [-1, 1]  -- the EDITED image
            attr_abs_idx:  (B,)  absolute CelebA attribute index
            target_values: (B,)  target attribute probability (detached)
            reduction:     'none' → return (B,) loss tensor
                           'mean' → return scalar
            src_images:    (B, 3, H, W)  range [-1, 1]  -- the SOURCE image.
                           REQUIRED when mode == 'directional'; ignored in
                           'absolute' mode.

        Returns:
            loss:  scalar or (B,) tensor depending on reduction
            logs:  dict of scalar metrics
        """
        if self.mode == "directional":
            if src_images is None:
                raise ValueError(
                    "FrozenCLIPPromptLoss(mode='directional') requires src_images."
                )
            return self._forward_directional(images, src_images, attr_abs_idx,
                                              target_values, reduction)
        return self._forward_absolute(images, attr_abs_idx, target_values, reduction)

    def _forward_absolute(self, images, attr_abs_idx, target_values, reduction):
        B = images.shape[0]

        img_feats = self._encode_images(images)
        pos_feats, neg_feats = self._text_deltas(attr_abs_idx, images.device)

        # ── Scores ────────────────────────────────────────────────────
        score_pos = (img_feats * pos_feats).sum(dim=-1)   # (B,)
        score_neg = (img_feats * neg_feats).sum(dim=-1)   # (B,)
        score     = score_pos - score_neg                  # (B,)

        # ── Direction-aware loss ───────────────────────────────────────
        direction  = torch.where(
            target_values.float() >= 0.5,
            images.new_ones(B),
            -images.new_ones(B),
        )
        loss_each = F.softplus(-direction * score / self.temperature)   # (B,)
        loss = loss_each.mean() if reduction == "mean" else loss_each

        logs = {
            "clip_target_loss":    loss_each.mean().detach(),
            "clip_score_mean":     score.mean().detach(),
            "clip_score_pos_mean": score_pos.mean().detach(),
            "clip_score_neg_mean": score_neg.mean().detach(),
            "clip_direction_mean": direction.mean().detach(),
        }
        return loss, logs

    def _forward_directional(self, images, src_images, attr_abs_idx, target_values, reduction):
        B = images.shape[0]

        # ── Image-space delta: where did the edit actually move the image ──
        feat_edit = self._encode_images(images)
        feat_src = self._encode_images(src_images)
        img_delta = F.normalize(feat_edit - feat_src, dim=-1, eps=1e-6)   # (B, D)

        # ── Text-space delta: where SHOULD it move, for this attribute ─────
        pos_feats, neg_feats = self._text_deltas(attr_abs_idx, images.device)
        text_delta = F.normalize(pos_feats - neg_feats, dim=-1, eps=1e-6)  # (B, D)

        direction = torch.where(
            target_values.float() >= 0.5,
            images.new_ones(B),
            -images.new_ones(B),
        )
        target_delta = direction.unsqueeze(-1) * text_delta   # (B, D)

        cos_sim = (img_delta * target_delta).sum(dim=-1)      # (B,) in [-1, 1]
        loss_each = 1.0 - cos_sim                              # (B,) in [0, 2]
        loss = loss_each.mean() if reduction == "mean" else loss_each

        logs = {
            "clip_target_loss":     loss_each.mean().detach(),
            "clip_directional_cos": cos_sim.mean().detach(),
            "clip_direction_mean":  direction.mean().detach(),
        }
        return loss, logs
