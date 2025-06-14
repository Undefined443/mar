import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
import scipy.stats as stats

from diffusion import create_diffusion
from timm.models.vision_transformer import Block


class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = CausalAttention(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CausalAttention(nn.Module):
    """
    由 MAE Encoder 修改而来
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth + decoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape
        seq_len /= 2  # 将 x 和 x_t 分开

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # FIXME 生成时使用 dropping
        if not self.training:
            x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        attn_mask = create_mask(self.buffer_size, seq_len)
        for block in self.encoder_blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.encoder_norm(x)
        x = x[:, self.buffer_size-1:]  # FIXME 去掉 buffer

        return x

    # def forward_with_cfg(self, x, t, c, cfg_scale):
    #     half = x[: len(x) // 2]
    #     combined = torch.cat([half, half], dim=0)
    #     model_out = self.forward(combined, t, c)
    #     eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    #     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    #     eps = torch.cat([half_eps, half_eps], dim=0)
    #     return torch.cat([eps, rest], dim=1)


def create_mask(cls_len, tok_len):
    mask = torch.ones(cls_len+2*tok_len, cls_len+2*tok_len)
    mask[:, :cls_len] = 0
    mask[cls_len:cls_len+tok_len, cls_len:cls_len+tok_len] = torch.triu(torch.ones(tok_len, tok_len), diagonal=1)
    mask[cls_len+tok_len:, cls_len-1:cls_len+tok_len-1] = torch.triu(torch.ones(tok_len, tok_len), diagonal=1)
    return mask