import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_edit_prompts(attr_abs_idx, target_values):
    prompts = []
    for attr, value in zip(attr_abs_idx.detach().cpu().tolist(), target_values.detach().cpu().tolist()):
        enabled = value >= 0.5
        if attr == 15:
            prompts.append(
                "a realistic face photo of a person wearing eyeglasses"
                if enabled else
                "a realistic face photo of a person without eyeglasses"
            )
        elif attr == 20:
            prompts.append(
                "a realistic face photo of a male person"
                if enabled else
                "a realistic face photo of a female person"
            )
        elif attr == 31:
            prompts.append(
                "a realistic face photo of a smiling person"
                if enabled else
                "a realistic face photo of a person with a neutral expression"
            )
        elif attr == 33:
            prompts.append(
                "a realistic face photo of a person with wavy hair"
                if enabled else
                "a realistic face photo of a person with straight hair"
            )
        elif attr == 39:
            # Structural/mid-layer aging cues, not skin-texture words (wrinkles,
            # smooth skin, forehead lines). The DDS gradient below only reaches
            # W+ layers < args.dds_fine_layer_start (coarse/mid); fine-grained
            # skin texture lives in the fine layers it never touches, so asking
            # for it here just wastes signal on something this loss can't move.
            prompts.append(
                "a realistic face photo of a young person with a full, firm "
                "jawline, high round cheeks, and a smooth brow"
                if enabled else
                "a realistic face photo of an elderly person with a sagging "
                "jawline, sunken cheeks, deep-set eyes, and a receding hairline"
            )
        else:
            prompts.append("a realistic face photo of a person")
    return prompts


class FrozenDiffusionDDSGuidance(nn.Module):
    """Frozen Stable Diffusion DDS guidance for edited StyleGAN images.

    The loss is an autograd surrogate whose gradient is the difference between
    the diffusion noise residual for the edited prompt and the source prompt.
    Diffusion/CLIP/VAE parameters stay frozen; gradients flow only through the
    edited image back to SDFlow/DirectionBank.
    """

    def __init__(
        self,
        model_id="SG161222/Realistic_Vision_V5.1_noVAE",
        vae_model_id="stabilityai/sd-vae-ft-mse",
        image_size=256,
        timestep_min=50,
        timestep_max=700,
        guidance_scale=1.0,
        fp16=True,
    ):
        super().__init__()
        try:
            from diffusers import AutoencoderKL, StableDiffusionPipeline
        except Exception as exc:
            raise ImportError(
                "Diffusion guidance requires diffusers. Install compatible packages, e.g.\n"
                "  conda run -n sdflow pip install 'diffusers==0.14.0' "
                "'huggingface-hub==0.14.1' 'transformers>=4.25,<4.31' accelerate safetensors"
            ) from exc

        self.image_size = int(image_size)
        self.timestep_min = int(timestep_min)
        self.timestep_max = int(timestep_max)
        self.guidance_scale = float(guidance_scale)
        self.fp16 = bool(fp16)

        dtype = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
        vae = None
        if vae_model_id:
            vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.set_progress_bar_config(disable=True)

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval()
        self.vae = pipe.vae.eval()
        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler
        self.latent_scale = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))

        for module in (self.text_encoder, self.vae, self.unet):
            for p in module.parameters():
                p.requires_grad_(False)

    def cuda(self, device=None):
        super().cuda(device=device)
        self.text_encoder.cuda(device=device)
        self.vae.cuda(device=device)
        self.unet.cuda(device=device)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        self.unet.to(*args, **kwargs)
        return self

    def _encode_text(self, prompts, device):
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        with torch.no_grad():
            return self.text_encoder(input_ids)[0]

    def _predict_noise(self, noisy_latents, timesteps, text_embeds):
        if self.guidance_scale <= 1.0:
            return self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample

        uncond_embeds = self._encode_text([""] * noisy_latents.shape[0], noisy_latents.device)
        latent_in = torch.cat([noisy_latents, noisy_latents], dim=0)
        t_in = torch.cat([timesteps, timesteps], dim=0)
        embeds = torch.cat([uncond_embeds, text_embeds], dim=0)
        noise_uncond, noise_cond = self.unet(latent_in, t_in, encoder_hidden_states=embeds).sample.chunk(2)
        return noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)

    def _encode_images(self, images):
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = F.interpolate(images, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        images = images.clamp(-1, 1).to(dtype=next(self.vae.parameters()).dtype)
        posterior = self.vae.encode(images).latent_dist
        if hasattr(posterior, "mean"):
            latents = posterior.mean
        else:
            latents = posterior.sample()
        return latents * self.latent_scale

    def forward(self, src_images, edit_images, attr_abs_idx, target_values, source_prompt=None,
                timestep_min=None, timestep_max=None):
        device = edit_images.device
        B = edit_images.shape[0]
        source_prompt = source_prompt or "a realistic face photo of a person"
        src_prompts = [source_prompt] * B
        edit_prompts = build_edit_prompts(attr_abs_idx, target_values)

        src_text = self._encode_text(src_prompts, device)
        edit_text = self._encode_text(edit_prompts, device)

        with torch.no_grad():
            src_latents = self._encode_images(src_images)
        edit_latents = self._encode_images(edit_images)

        num_train_steps = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        _t_min = self.timestep_min if timestep_min is None else int(timestep_min)
        _t_max = self.timestep_max if timestep_max is None else int(timestep_max)
        t_min = max(0, min(_t_min, num_train_steps - 1))
        t_max = max(t_min + 1, min(_t_max, num_train_steps))
        timesteps = torch.randint(t_min, t_max, (B,), device=device, dtype=torch.long)

        noise = torch.randn_like(edit_latents)
        noisy_edit = self.scheduler.add_noise(edit_latents, noise, timesteps)
        noisy_src = self.scheduler.add_noise(src_latents, noise, timesteps)

        autocast_ctx = torch.cuda.amp.autocast if (self.fp16 and torch.cuda.is_available()) else contextlib.nullcontext
        with torch.no_grad():
            with autocast_ctx():
                eps_src = self._predict_noise(noisy_src, timesteps, src_text)
        with autocast_ctx():
            eps_edit = self._predict_noise(noisy_edit, timesteps, edit_text)

        grad = (eps_edit - eps_src).detach()
        loss = (grad * edit_latents).mean()
        logs = {
            "diffusion_dds_grad_norm": grad.float().flatten(1).norm(dim=1).mean().detach(),
            "diffusion_timestep": timesteps.float().mean().detach(),
        }
        return loss, logs
