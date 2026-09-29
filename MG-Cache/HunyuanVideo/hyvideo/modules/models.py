from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attenion import attention, parallel_attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner

step_fit_kb = torch.load("./calibration_data/linear_fit_step.pth")
layer_score = torch.load('./calibration_data/layer_scores.pth')

def rel_l1(a: torch.Tensor ,b: torch.Tensor):
    l1 = torch.abs(a - b).mean()
    rel_l1 = l1 / (torch.abs(b).mean())  # 相对L1距离
    return rel_l1

def l1_distance(a: torch.Tensor ,b: torch.Tensor):
    l1 = torch.abs(a - b).mean()
    return l1

def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    cos_sim = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=-1)
    return 1-cos_sim

prefetch_stream = torch.cuda.Stream()

class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        pre_layer_input: torch.Tensor = None,
        pre_attn: torch.Tensor = None,
        cache_policy: str = "blank",
        beta: float = 0.1,
        step_i: int = 0,
        layer_i: int = 0,
        acc_error: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )

        update_layer_input = None
        update_attn = None
        is_skiped = False
        if cache_policy=="layer":
            acc_error += (rel_l1(pre_layer_input, torch.cat((img_modulated, txt_modulated), 1)) * layer_score[f'step_{step_i}'][f'layer_{layer_i}']).item()
            update_layer_input = torch.cat((img_modulated, txt_modulated), 1) # 更新参考变量列表
        if cache_policy=="layer" and acc_error < beta:
            attn = pre_attn # 复用缓存
            is_skiped = True
        else:
            if cache_policy in ["warmup", "layer"]:
                acc_error = 0.0
            img_qkv = self.img_attn_qkv(img_modulated)
            img_q, img_k, img_v = rearrange(
                img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
            )
            # Apply QK-Norm if needed
            img_q = self.img_attn_q_norm(img_q).to(img_v)
            img_k = self.img_attn_k_norm(img_k).to(img_v)

            # Apply RoPE if needed.
            if freqs_cis is not None:
                img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                assert (
                    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
                ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
                img_q, img_k = img_qq, img_kk

            txt_qkv = self.txt_attn_qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(
                txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
            )
            # Apply QK-Norm if needed.
            txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
            txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

            # Run actual attention.
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            v = torch.cat((img_v, txt_v), dim=1)
            assert (
                cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
            ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        
            # attention computation start
            if not self.hybrid_seq_parallel_attn:
                attn = attention(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=img_k.shape[0],
                )
            else:
                attn = parallel_attention(
                    self.hybrid_seq_parallel_attn,
                    q,
                    k,
                    v,
                    img_q_len=img_q.shape[1],
                    img_kv_len=img_k.shape[1],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv
                )
            
            # attention computation end
            if cache_policy in ["warmup", "layer"]:
                update_attn = attn

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt, update_layer_input, update_attn, is_skiped, acc_error


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        pre_layer_input: torch.Tensor = None,
        pre_attn: torch.Tensor = None,
        cache_policy: str = "blank",
        beta: float = 0.1,
        step_i: int = 0,
        layer_i: int = 0,
        acc_error: float = 0.0,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
                self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
            )
        update_layer_input = None
        update_attn = None
        is_skiped = False
        if cache_policy=="layer":
            acc_error += (rel_l1(pre_layer_input, x_mod) * layer_score[f'step_{step_i}'][f'layer_{layer_i}']).item()
            update_layer_input = x_mod # 更新参考变量列表
        if cache_policy == "layer" and acc_error < beta:
            attn = pre_attn # 复用缓存
            is_skiped = True
        else:
            if cache_policy in ["warmup", "layer"]:
                acc_error = 0.0

            q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

            # Apply QK-Norm if needed.
            q = self.q_norm(q).to(v)
            k = self.k_norm(k).to(v)

            # Apply RoPE if needed.
            if freqs_cis is not None:
                img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
                img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
                img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                assert (
                    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
                ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
                img_q, img_k = img_qq, img_kk
                q = torch.cat((img_q, txt_q), dim=1)
                k = torch.cat((img_k, txt_k), dim=1)

            # Compute attention.
            assert (
                cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
            ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        
            # attention computation start
            if not self.hybrid_seq_parallel_attn:
                attn = attention(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=x.shape[0],
                )
            else:
                attn = parallel_attention(
                    self.hybrid_seq_parallel_attn,
                    q,
                    k,
                    v,
                    img_q_len=img_q.shape[1],
                    img_kv_len=img_k.shape[1],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv
                )
            # attention computation end
            if cache_policy in ["warmup", "layer"]:
                update_attn = attn

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate), update_layer_input, update_attn, is_skiped, acc_error


class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = args.text_states_dim
        self.text_states_dim_2 = args.text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        self.is_cache_offload = True
        self.skip_timestep = []

        seq_len = int((1+(args.video_length-1)/4)*(args.video_size[0]/16)*(args.video_size[1]/16)+256)
        if self.is_cache_offload:
            self.layer_input_cache = [
                    torch.empty((1, seq_len, 3072), device='cpu', dtype=torch.bfloat16, pin_memory=True)
                    for _ in range(60)
                ]
            self.attn_cache = [
                    torch.empty((1, seq_len, 3072), device='cpu', dtype=torch.bfloat16, pin_memory=True)
                    for _ in range(60)
                ]
        else:
            self.layer_input_cache = [
                    torch.empty((1, seq_len, 3072), device='cuda', dtype=torch.bfloat16)
                    for _ in range(60)
                ]
            self.attn_cache = [
                    torch.empty((1, seq_len, 3072), device='cuda', dtype=torch.bfloat16)
                    for _ in range(60)
                ]

        self.timestep_cache = None

        self.pre_layer_input = None
        self.pre_attn = None

        # 每一层对应的layer cache累积误差值
        self.acc_error = [0.0 for _ in range(60)]
        self.skiped_layer_num = 0

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_cache_policy(self, i):
        if i in self.skip_timestep:
            cache_policy = "step"
        elif i==2:
            cache_policy = "warmup"
        elif i < 2:
            cache_policy = "blank"
        else:
            cache_policy = "layer"
        print(f'step[{i}]: {cache_policy}')
        return cache_policy

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
        step_index: int = 0,
        beta: float = 0.4,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        cache_policy = self.get_cache_policy(step_index)
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        if cache_policy == "step":
            img = step_fit_kb[f'step_{step_index}']["k"]*self.timestep_cache + step_fit_kb[f'step_{step_index}']["b"]
            self.timestep_cache = img
        else:
            # Prepare modulation vectors.
            vec = self.time_in(t)

            # text modulation
            vec = vec + self.vector_in(text_states_2)

            # guidance modulation
            if self.guidance_embed:
                if guidance is None:
                    raise ValueError(
                        "Didn't get guidance strength for guidance distilled model."
                    )

                # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
                vec = vec + self.guidance_in(guidance)

            # Embed image and text.
            img = self.img_in(img)
            if self.text_projection == "linear":
                txt = self.txt_in(txt)
            elif self.text_projection == "single_refiner":
                txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
            else:
                raise NotImplementedError(
                    f"Unsupported text_projection: {self.text_projection}"
                )

            txt_seq_len = txt.shape[1]
            img_seq_len = img.shape[1]

            # Compute cu_squlens and max_seqlen for flash attention
            cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = img_seq_len + txt_seq_len
            max_seqlen_kv = max_seqlen_q

            freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
            skiped_layer = []
            # --------------------- Pass through DiT blocks ------------------------
            if self.is_cache_offload:
                for i, block in enumerate(self.double_blocks):
                    if cache_policy=="layer":
                        with torch.cuda.stream(prefetch_stream):
                            prefetch_layer_input = self.layer_input_cache[i + 1].to('cuda', non_blocking=True)
                            prefetch_attn = self.attn_cache[i + 1].to('cuda', non_blocking=True)
                    double_block_args = [
                        img,
                        txt,
                        vec,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        freqs_cis,
                        self.pre_layer_input,
                        self.pre_attn,
                        cache_policy,
                        beta,
                        step_index,
                        i,
                        self.acc_error[i],
                    ]
                    img, txt, update_layer_input, update_attn, is_skiped, acc_error = block(*double_block_args)
                    if update_layer_input is not None:
                        self.layer_input_cache[i].copy_(update_layer_input, non_blocking=True)
                    if update_attn is not None:
                        self.attn_cache[i].copy_(update_attn, non_blocking=True)
                    self.acc_error[i] = acc_error
                    if cache_policy=="layer":
                        torch.cuda.current_stream().wait_stream(prefetch_stream)
                        self.pre_layer_input = prefetch_layer_input
                        self.pre_attn = prefetch_attn
                    if is_skiped:
                        skiped_layer.append(i)

                # Merge txt and img to pass through single stream blocks.
                x = torch.cat((img, txt), 1)
                if len(self.single_blocks) > 0:
                    for i, block in enumerate(self.single_blocks):
                        if cache_policy=="layer" or (cache_policy=="warmup" and i==39):
                            with torch.cuda.stream(prefetch_stream):
                                prefetch_layer_input = self.layer_input_cache[(i + 21)%60].to('cuda', non_blocking=True)
                                prefetch_attn = self.attn_cache[(i + 21)%60].to('cuda', non_blocking=True)
                        single_block_args = [
                            x,
                            vec,
                            txt_seq_len,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            (freqs_cos, freqs_sin),
                            self.pre_layer_input,
                            self.pre_attn,
                            cache_policy,
                            beta,
                            step_index,
                            (i+20),
                            self.acc_error[i+20],
                        ]
                        x, update_layer_input, update_attn, is_skiped, acc_error = block(*single_block_args)
                        if update_layer_input is not None:
                            self.layer_input_cache[i+20].copy_(update_layer_input, non_blocking=True)
                        if update_attn is not None:
                            self.attn_cache[i+20].copy_(update_attn, non_blocking=True)
                        self.acc_error[i+20] = acc_error
                        if cache_policy=="layer"or (cache_policy=="warmup" and i==39):
                            torch.cuda.current_stream().wait_stream(prefetch_stream)
                            self.pre_layer_input = prefetch_layer_input
                            self.pre_attn = prefetch_attn
                        if is_skiped:
                            skiped_layer.append(i+20)
            else:
                for i, block in enumerate(self.double_blocks):
                    double_block_args = [
                        img,
                        txt,
                        vec,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        freqs_cis,
                        self.layer_input_cache[i],
                        self.attn_cache[i],
                        cache_policy,
                        beta,
                        step_index,
                        i,
                        self.acc_error[i],
                    ]
                    img, txt, update_layer_input, update_attn, is_skiped, acc_error = block(*double_block_args)
                    if update_layer_input is not None:
                        self.layer_input_cache[i].copy_(update_layer_input, non_blocking=True)
                    if update_attn is not None:
                        self.attn_cache[i].copy_(update_attn, non_blocking=True)
                    self.acc_error[i] = acc_error
                    if is_skiped:
                        skiped_layer.append(i)

                # Merge txt and img to pass through single stream blocks.
                x = torch.cat((img, txt), 1)
                if len(self.single_blocks) > 0:
                    for i, block in enumerate(self.single_blocks):
                        single_block_args = [
                            x,
                            vec,
                            txt_seq_len,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            (freqs_cos, freqs_sin),
                            self.layer_input_cache[i+20],
                            self.attn_cache[i+20],
                            cache_policy,
                            beta,
                            step_index,
                            (i+20),
                            self.acc_error[i+20],
                        ]
                        x, update_layer_input, update_attn, is_skiped, acc_error = block(*single_block_args)
                        if update_layer_input is not None:
                            self.layer_input_cache[i+20].copy_(update_layer_input, non_blocking=True)
                        if update_attn is not None:
                            self.attn_cache[i+20].copy_(update_attn, non_blocking=True)
                        self.acc_error[i+20] = acc_error
                        if is_skiped:
                            skiped_layer.append(i+20)

            img = x[:, :img_seq_len, ...]

            if cache_policy == "layer":
                print(f'skiped layers:{skiped_layer}')
                self.skiped_layer_num += len(skiped_layer)

            # ---------------------------- Final layer ------------------------------
            img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
            self.timestep_cache = img

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}
