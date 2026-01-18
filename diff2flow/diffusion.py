import torch
import einops
import numpy as np
import torch.nn as nn
from functools import partial

from diff2flow.helpers import instantiate_from_config
from diff2flow.ddpm import GaussianDiffusion
from diff2flow.ddim import DDIMSampler


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def cosine_log_snr(t, eps=0.00001):
    """
    Returns log Signal-to-Noise ratio for time step t and image size 64
    eps: avoid division by zero
    """
    return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def shifted_cosine_log_snr(t, im_size: int, ref_size: int = 64):
    return cosine_log_snr(t) + 2 * np.log(ref_size / im_size)


def log_snr_to_alpha_bar(t):
    return sigmoid(cos_log_snr(t))


def shifted_cosine_alpha_bar(t, im_size: int, ref_size: int = 64):
    return sigmoid(shifted_cosine_log_snr(t, im_size, ref_size))


class ForwardDiffusion(nn.Module):
    def __init__(self,
                 im_size: int = 64,
                 n_diffusion_timesteps: int = 1000):
        super().__init__()
        self.n_diffusion_timesteps = n_diffusion_timesteps
        cos_alpha_bar_t = shifted_cosine_alpha_bar(
            np.linspace(0, 1, n_diffusion_timesteps),
            im_size=im_size
        ).astype(np.float32)
        self.register_buffer("alpha_bar_t", torch.from_numpy(cos_alpha_bar_t))
        
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps. In other
        words sample from q(x_t | x_0).

        Args:
            x_start: The initial data batch.
            t: The diffusion time-step (must be a single t).
            noise: If specified, the noise to use for the diffusion.
        Returns:
            A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        alpha_bar_t = self.alpha_bar_t[t]

        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise
        

""" Diffusion Wrapper adapted to FlowModel """


class DiffusionFlow(nn.Module):
    def __init__(
            self,
            net_cfg: dict,
            timesteps: int = 1000,
            beta_schedule: str = 'linear',
            loss_type: str = 'l2',
            parameterization: str = 'v',
            linear_start: float = 0.00085,
            linear_end: float = 0.0120,
            ddim_steps: int = 10,           # 10 timesteps default for FlowModel
    ):
        super().__init__()
        self.net = instantiate_from_config(net_cfg)

        self.diffusion_cfg = dict(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            zero_terminal_snr=False,
            loss_type=loss_type,
            parameterization=parameterization,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=8e-3,
            original_elbo_weight=0.,
            v_posterior=0.,
            l_simple_weight=1.0
        )
        self.diffusion = GaussianDiffusion(**self.diffusion_cfg)

        self.ddim_steps = ddim_steps
        self.ddim_sampler = DDIMSampler(self.diffusion)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        return self.net(x, t, **kwargs)

    def training_losses(self, x1: torch.Tensor, x0: torch.Tensor = None, **cond_kwargs):
        loss, _ = self.diffusion.training_losses(
            model=self.net,
            x_start=x1,
            model_kwargs=cond_kwargs,
            x_noise=x0,
        )
        return loss

    def generate(self, x: torch.Tensor, sample_kwargs=None, reverse=False, return_intermediates=False, **kwargs):
        """
        Args:
            x: source minibatch (bs, *dim)
            sample_kwargs: dict, additional sampling arguments
            reverse: bool
            return_intermediates: bool
            kwargs: conditioning information (e.g. context, context_ca) AND guidance_scale
        """
        if reverse:
            raise NotImplementedError("[DiffusionFlow] Reverse sampling not yet supported")

        sample_kwargs = sample_kwargs or {}

        # 1. 自動從 kwargs 或 sample_kwargs 抓取 guidance scale
        # 允許使用者傳入 guidance_scale 或 cfg_scale
        guidance_scale = kwargs.pop("guidance_scale", None)
        if guidance_scale is None:
            guidance_scale = sample_kwargs.get("cfg_scale", 1.0)

        # 2. 準備 unconditional conditioning (uc_cond)
        # 這裡假設如果開啟 guidance，必須提供 uc_cond (通常是空文本的 embedding)
        # 如果沒有特別提供，我們假設 context_ca (文字條件) 的部分要做空值替換，
        # 具體實作通常由外部傳入 uc_cond，這裡先保留介面
        uc_cond = kwargs.pop("uc_cond", None)
        if uc_cond is None:
            uc_cond = sample_kwargs.get("uc_cond", None)

        # 3. 建立帶有 CFG 功能的 Forward Function
        # 只有當 scale != 1.0 時才需要包裝，但為了統一邏輯，我們一律檢查
        forward_fn = partial(
            forward_with_cfg,
            model=self.net,
            cfg_scale=guidance_scale,
            uc_cond=uc_cond,
            cond_key=sample_kwargs.get("cond_key", "context_ca"),  # 預設 cond key 為 context_ca (文字)
        )

        # DDPM sampling
        if sample_kwargs.get("use_ddpm", False):
            out, intermediates = self.diffusion.p_sample_loop(
                model=forward_fn,  # 使用包裝過的模型
                noise=x,
                model_kwargs=kwargs,
                progress=sample_kwargs.get("progress", False),
                clip_denoised=sample_kwargs.get("clip_denoised", False),
                return_intermediates=True,
                intermediate_freq=sample_kwargs.get("intermediate_freq", (100 if return_intermediates else 1000)),
                pbar_desc=sample_kwargs.get("pbar_desc", "DDPM Sampling"),
                intermediate_key=sample_kwargs.get("intermediate_key", "sample"),
            )

        # DDIM sampling (這是你主要用的)
        else:
            # --- 修正開始：這裡原本傳的是 self.net，導致 CFG 失效 ---
            out, intermediates = self.ddim_sampler.sample(
                model=forward_fn,  # <--- 關鍵修改：傳入 forward_fn
                noise=x,
                model_kwargs=kwargs,
                ddim_steps=sample_kwargs.get("num_steps", self.ddim_steps),
                eta=sample_kwargs.get("eta", 0.),
                progress=sample_kwargs.get("progress", False),
                temperature=sample_kwargs.get("temperature", 1.),
                noise_dropout=sample_kwargs.get("noise_dropout", 0.),
                log_every_t=sample_kwargs.get("intermediate_freq", (10 if return_intermediates else 1000)),
                clip_denoised=sample_kwargs.get("clip_denoised", False),
                # 移除這裡傳入的 cfg_scale，因為已經包在 forward_fn 裡了
            )
            # --- 修正結束 ---

            key = sample_kwargs.get("intermediate_key", "x_inter")
            intermediates = intermediates[key]

        if return_intermediates:
            return torch.stack(intermediates, 0)
        return out

# def forward_with_cfg(x, t, model, cfg_scale=1.0, uc_cond=None, cond_key="y", **model_kwargs):
    """ Function to include sampling with Classifier-Free Guidance (CFG) """
    # if cfg_scale == 1.0:                                # without CFG
        # model_output = model(x, t, **model_kwargs)

    # else:                                               # with CFG
        # assert cond_key in model_kwargs, f"Condition key '{cond_key}' for CFG not found in model_kwargs"
        # assert uc_cond is not None, "Unconditional condition not provided for CFG"
        # kwargs = model_kwargs.copy()
        # c = kwargs[cond_key]
        # x_in = torch.cat([x] * 2)
        # t_in = torch.cat([t] * 2)
        # if uc_cond.shape[0] == 1:
            # uc_cond = einops.repeat(uc_cond, '1 ... -> bs ...', bs=x.shape[0])
        # _in = torch.cat([uc_cond, c])
        # kwargs[cond_key] = c_in
        # model_uc, model_c = model(x_in, t_in, **kwargs).chunk(2)
        # model_output = model_uc + cfg_scale * (model_c - model_uc)

    # return model_output


# diff2flow/diffusion.py

def forward_with_cfg(x, t, model, cfg_scale=1.0, uc_cond=None, cond_key="context_ca", **model_kwargs):
    """ Function to include sampling with Classifier-Free Guidance (CFG) """
    if cfg_scale == 1.0:
        return model(x, t, **model_kwargs)

    # 準備 CFG
    # 1. 複製輸入 x 和時間 t
    x_in = torch.cat([x] * 2)
    t_in = torch.cat([t] * 2)

    # 2. 複製所有 model_kwargs (包含 context, y 等等)
    # 這一步最關鍵！修正了 Batch Size 不匹配的問題
    kwargs = model_kwargs.copy()
    batch_size = x.shape[0]

    for k, v in kwargs.items():
        # 如果是 Tensor 且第一維度跟 Batch Size 一樣，就複製它
        if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
            # 特殊處理 cond_key (文字)，因為它要混合 uc_cond
            if k == cond_key:
                continue
            kwargs[k] = torch.cat([v] * 2)

    # 3. 處理文字條件 (Conditioning)
    assert cond_key in model_kwargs, f"Condition key '{cond_key}' not found"
    assert uc_cond is not None, "Unconditional condition (uc_cond) required for CFG"

    c = model_kwargs[cond_key]

    # 確保 uc_cond 的 batch size 正確
    if uc_cond.shape[0] == 1 and batch_size > 1:
        uc_cond = uc_cond.repeat(batch_size, 1, 1)  # 或者是 einops.repeat
    elif uc_cond.shape[0] != batch_size:
        # 如果 uc_cond 數量不對，嘗試修正 (防呆)
        uc_cond = uc_cond[:1].repeat(batch_size, 1, 1)

    # 串接文字條件：[空字串, 有字串]
    # 注意：通常 Uncond 在前或後取決於實作，這裡假設 [Uncond, Cond]
    c_in = torch.cat([uc_cond, c])
    kwargs[cond_key] = c_in

    # 4. 模型預測
    model_output = model(x_in, t_in, **kwargs)

    # 5. 拆分結果並計算 CFG
    model_uc, model_c = model_output.chunk(2)

    # 公式：Pred = Uncond + Scale * (Cond - Uncond)
    return model_uc + cfg_scale * (model_c - model_uc)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = ForwardDiffusion(im_size=64)

    plt.plot(d.alpha_bar_t)
    plt.plot(1 - d.alpha_bar_t)
    plt.savefig("cosine.png")
    plt.close()

    im = torch.ones((1, 3, 128, 128))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        xt = d.q_sample(im, i * 100)
        print(f"t={i*100}: {d.alpha_bar_t[i*100]}")
        ax.imshow(xt[0].permute(1, 2, 0).detach().numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("diffusion.png")
    plt.close()
