# -*- coding: utf-8 -*-
"""Nunchaku backend utilities."""

import torch

from ..tinychat.utils import convert_to_tinychat_w4x16y16_linear_weight
from ..utils import MmaWeightPackerBase, ceil_divide, pad

__all__ = [
    "convert_to_nunchaku_w4x4y16_linear_weight",
    "convert_to_nunchaku_w8x8y16_linear_weight",
    "convert_to_nunchaku_w4x16_linear_weight",
]


class NunchakuWeightPacker(MmaWeightPackerBase):
    def __init__(self, bits: int, warp_n: int = 128):
        super().__init__(bits=bits, warp_n=warp_n)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.dtype == torch.int32, f"quantized weight should be torch.int32, but got {weight.dtype}."
        n, k = weight.shape
        assert n % self.warp_n == 0, "output channel size should be divisible by warp_n."
        assert k % self.warp_k == 0, "input channel size should be divisible by warp_k."
        n_warps, k_warps = n // self.warp_n, k // self.warp_k
        weight = weight.reshape(
            n_warps,
            self.num_n_frags,
            self.n_pack_size,  # always 2
            self.num_n_lanes,  # constant 8
            # self.lane_n,  # constant 1
            k_warps,
            # self.num_k_frags,  # constant 1
            self.k_pack_size,  # always 2
            self.num_k_lanes,  # constant 4
            self.lane_k,  # always 8 = 32 bits / 4 bits
        )
        weight = weight.permute(0, 4, 1, 3, 6, 2, 5, 7).contiguous()
        assert weight.shape[3:-1] == (8, 4, 2, 2)
        if self.bits == 4:
            weight = weight.bitwise_and_(0xF)
            shift = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        elif self.bits == 8:
            weight = weight.bitwise_and_(0xFF)
            shift = torch.arange(0, 32, 8, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        else:
            raise NotImplementedError(f"weight bits {self.bits} is not supported.")
        return weight.view(dtype=torch.int8).view(n, -1)  # assume little-endian

    def pack_scale(self, scale: torch.Tensor) -> torch.Tensor:
        assert scale.dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
        n = scale.shape[0]
        # min pack size set to 32b/16b = 2
        # max pack size set to 128b/16b = 8
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        scale = scale.reshape(ceil_divide(n, warp_s), num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
        scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return scale.view(-1, n)  # the shape is just used for validation

    def pack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        """Pack Low-Rank Weight.

        Args:
            weight (`torch.Tensor`):
                low-rank weight tensor.
            down (`bool`):
                whether the weight is for down projection in low-rank branch.
        """
        assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
        lane_n, lane_k = 1, 2  # lane_n is always 1, lane_k is 32 bits // 16 bits = 2
        frag_n = self.n_pack_size * self.num_n_lanes * lane_n
        frag_k = self.k_pack_size * self.num_k_lanes * lane_k
        weight = pad(weight, divisor=(frag_n, frag_k), dim=(0, 1))
        if down:
            r, c = weight.shape
            r_frags, c_frags = r // frag_n, c // frag_k
            weight = weight.view(r_frags, frag_n, c_frags, frag_k).permute(2, 0, 1, 3)
        else:
            c, r = weight.shape
            c_frags, r_frags = c // frag_n, r // frag_k
            weight = weight.view(c_frags, frag_n, r_frags, frag_k).permute(0, 2, 1, 3)
        weight = weight.reshape(
            c_frags, r_frags, self.n_pack_size, self.num_n_lanes, self.k_pack_size, self.num_k_lanes, lane_k
        )
        weight = weight.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        return weight.view(c, r)

    def unpack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        """Unpack Low-Rank Weight.

        Args:
            weight (`torch.Tensor`):
                low-rank weight tensor.
            down (`bool`):
                whether the weight is for down projection in low-rank branch.
        """
        c, r = weight.shape
        assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
        lane_n, lane_k = 1, 2  # lane_n is always 1, lane_k is 32 bits // 16 bits = 2
        frag_n = self.n_pack_size * self.num_n_lanes * lane_n
        frag_k = self.k_pack_size * self.num_k_lanes * lane_k
        if down:
            r_frags, c_frags = r // frag_n, c // frag_k
        else:
            c_frags, r_frags = c // frag_n, r // frag_k
        weight = weight.view(
            c_frags, r_frags, self.num_n_lanes, self.num_k_lanes, self.n_pack_size, self.k_pack_size, lane_k
        )
        weight = weight.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        weight = weight.view(c_frags, r_frags, frag_n, frag_k)
        if down:
            weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
        else:
            weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
        return weight


def convert_to_nunchaku_w4x4y16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    assert weight.ndim == 2, "weight tensor should be 2D."
    device, dtype = weight.device, weight.dtype
    assert dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
    assert scale is not None, "scale tensor is required for quantization."

    oc, ic = weight.shape
    if scale.numel() == 1:
        scale = scale.view(-1).expand(oc).reshape(oc, 1, 1, 1)
        ng, gs = 1, ic
    else:
        assert scale.ndim == 4, "scale tensor should be 4D."
        assert scale.shape[1] == scale.shape[3] == 1
        assert scale.shape[0] == oc
        ng, gs = scale.shape[2], ic // scale.shape[2]
        assert ic == gs * ng, "input channel size should be equal to group size times number of groups."
    # region quantize and pack weight tensor
    weight = weight.to(dtype=torch.float32).view(oc, 1, ng, gs)
    weight = weight.div_(scale.to(dtype=torch.float32, device=device)).round_().to(torch.int32).view(oc, ic)
    assert weight.min() >= -8 and weight.max() <= 7, "quantized weight should be in [-8, 7]."
    weight = pad(weight, divisor=128, dim=(0, 1))
    # endregion

    scale = pad(scale.to(dtype=dtype), divisor=(128, 2), dim=(0, 2), fill_value=1)
    bias = torch.zeros([weight.shape[0]], dtype=dtype) if bias is None else pad(bias, divisor=128, dim=0)
    smooth = torch.ones([weight.shape[1]], dtype=dtype) if smooth is None else pad(smooth, divisor=128, dim=0)

    packer = NunchakuWeightPacker(bits=4)
    weight = packer.pack_weight(weight)
    scale = packer.pack_scale(scale)
    bias = packer.pack_scale(bias.to(dtype=dtype).view(-1, 1)).view(-1)
    smooth = packer.pack_scale(smooth.to(dtype=dtype).view(-1, 1)).view(-1)
    if lora is not None:
        lora_down = packer.pack_lowrank_weight(pad(lora[0], divisor=128, dim=1), down=True)
        lora_up = packer.pack_lowrank_weight(pad(lora[1], divisor=128, dim=0), down=False)
        lora = (lora_down, lora_up)

    return weight, scale, bias, smooth, lora


def convert_to_nunchaku_w8x8y16_linear_weight(
    weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weight.ndim == 2, "weight tensor should be 2D."
    device, dtype = weight.device, weight.dtype
    assert dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
    assert scale is not None, "scale tensor is required for quantization."
    oc, ic = weight.shape
    if scale.numel() == 1:
        scale = scale.view(-1).expand(oc)
    scale = scale.reshape(oc, 1)
    weight = weight.to(dtype=torch.float32)
    weight = weight.div_(scale.to(dtype=torch.float32, device=device)).round_().to(torch.int32).view(oc, ic)
    assert weight.min() >= -128 and weight.max() <= 127, "quantized weight should be in [-128, 127]."
    weight = pad(weight, divisor=128, dim=(0, 1))
    # endregion
    scale = pad(scale.to(dtype=dtype), divisor=128, dim=0, fill_value=1)
    bias = torch.zeros([weight.shape[0]], dtype=dtype) if bias is None else pad(bias, divisor=128, dim=0)
    packer = NunchakuWeightPacker(bits=8)
    weight = packer.pack_weight(weight)
    scale = packer.pack_scale(scale)
    bias = packer.pack_scale(bias.to(dtype=dtype).view(-1, 1)).view(-1)
    return weight, scale, bias


def convert_to_nunchaku_w4x16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    adanorm_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    oc, ic = weight.shape
    assert scale.ndim == 4, "scale tensor should be 4D."
    assert scale.shape[0] == oc
    assert scale.shape[1] == scale.shape[3] == 1
    ng = scale.shape[2]
    if bias is None:
        bias = torch.zeros([oc], dtype=weight.dtype, device=weight.device)
    assert oc % adanorm_splits == 0, "output channel size should be divisible by splits."
    if adanorm_splits > 1:
        weight = weight.view(adanorm_splits, oc // adanorm_splits, ic).transpose(0, 1).reshape(oc, ic)
        scale = scale.view(adanorm_splits, oc // adanorm_splits, ng).transpose(0, 1).reshape(oc, 1, ng, 1)
        bias = bias.reshape(adanorm_splits, oc // adanorm_splits).transpose(0, 1)
        delta = [0] * adanorm_splits
        delta[1] = delta[-2] = 1
        bias = bias.add_(torch.tensor(delta, dtype=bias.dtype, device=bias.device))
        bias = bias.reshape(oc)
    weight, scale, zero = convert_to_tinychat_w4x16y16_linear_weight(
        weight=weight,
        scale=scale,
        zero=torch.full_like(scale, 7) if zero is None else zero,
        zero_pre_scaled=True,
    )
    weight = weight.view(torch.int32)
    return weight, scale, zero, bias
