# -*- coding: utf-8 -*-
"""QServe backend utilities."""

import torch

from ..utils import MmaWeightPackerBase

__all__ = ["convert_to_qserve_w4x8y16_linear_weight", "convert_to_qserve_w8x8y16_linear_weight"]


class QServePacker(MmaWeightPackerBase):
    def __init__(self):
        super().__init__(bits=8, warp_n=32)
        assert self.num_n_frags == 2

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.min() >= 0, "quantized weight should be non-negative."
        assert weight.max() <= 15, "quantized weight should be less than 16."
        assert weight.dtype == torch.uint8, f"quantized weight should be torch.uint8, but got {weight.dtype}."
        n, k = weight.shape
        assert n % self.warp_n == 0, "output channel size should be divisible by warp_n."
        assert k % self.warp_k == 0, "input channel size should be divisible by warp_k."
        n_warps, k_warps = n // self.warp_n, k // self.warp_k
        weight = weight.reshape(
            n_warps,
            self.num_n_frags,  # always 2
            self.n_pack_size,  # always 2
            self.num_n_lanes,  # constant 8
            # self.lane_n,  # constant 1
            k_warps,
            # self.num_k_frags,  # constant 1
            self.k_pack_size,  # always 2
            self.num_k_lanes,  # constant 4
            self.lane_k,  # always 4 = 32 bits / 8 bits
        )
        weight = weight.permute(1, 0, 4, 3, 6, 5, 2, 7).contiguous()
        assert weight.shape[3:-1] == (8, 4, 2, 2)
        weight = (weight[1] << 4) + weight[0]
        weight = weight.view(n_warps, k_warps, self.num_lanes, self.pack_size, self.lane_k).view(torch.int8)
        return weight.view(n, k // 2)

    def pack_scale(
        self, scale: torch.Tensor, zero: torch.Tensor | None = None, subscale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        scale = scale.view(-1)
        n = scale.shape[0]
        if subscale is None:
            zero = zero.view(-1)
        else:
            assert subscale.dtype == torch.int8, f"subscale should be torch.int8, but got {subscale.dtype}."
            view_shape = (n // self.warp_n, self.num_n_frags * self.n_pack_size, self.num_n_lanes, -1)
            subscale = subscale.view(view_shape).permute(3, 0, 2, 1).contiguous().view(-1, n)
            zero = zero.view(view_shape).permute(3, 0, 2, 1).contiguous().view(-1, n)
        return scale, zero, subscale


def convert_to_qserve_w4x8y16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    subscale: torch.Tensor | None = None,
    zero_pre_scaled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert a weight tensor to QServe W4-X8-Y16 linear weight format.

    Args:
        weight (`torch.Tensor`):
            weight tensor to be converted.
        scale (`torch.Tensor`):
            scale tensor for the weight tensor.
        zero (`torch.Tensor`):
            zero point tensor for the weight tensor.
        subscale (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            subscale tensor for the weight tensor.
        zero_pre_scaled (`bool`, *optional*, defaults to `False`):
            whether zero point tensor is pre-scaled.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]`:
            packed quantized weight tensor, scale tensor, zero point tensor, and subscale tensor.
    """
    dtype = weight.dtype
    assert dtype == torch.float16, "currently qserve only supports fp16."
    assert scale is not None, "scale tensor is required for quantization."
    assert zero is not None, "zero point tensor is required for quantization."
    weight = weight.to(dtype=torch.float32)
    scale = scale.to(dtype=torch.float32, device=weight.device)
    zero = zero.to(dtype=torch.float32, device=weight.device)
    oc, ic = weight.shape
    if subscale is not None:  # per-group quantization
        subscale = subscale.to(dtype=weight.dtype, device=weight.device)
        # region reshape scale and zero point
        if scale.numel() == 1:
            scale = scale.view(-1).expand(oc)
        scale = scale.reshape(oc).contiguous().view(oc, 1)
        assert subscale.numel() > 1, "subscale tensor is required for per-group quantization."
        subscale = subscale.view(oc, -1, 1).round_()
        ng = subscale.shape[1]
        gs = ic // ng
        assert ic == ng * gs, "input channel size should be divisible by group size."
        if zero.numel() == 1:
            zero = zero.view(1, 1).expand(oc, ng)
        zero = zero.reshape(oc, ng).contiguous().view(oc, ng, 1).round_()
        # endregion
        # region quantize weight tensor
        weight = weight.div_(scale).round_()
        assert weight.min() >= -128, "first-level quantized weight should be greater than or equal to -128."
        assert weight.max() <= 127, "first-level quantized weight should be less than or equal to 127."
        weight = weight.view(oc, ng, gs)
        if not zero_pre_scaled:  # zero point is int8
            weight = weight.add_(zero)
        weight = weight.div_(subscale)
        if zero_pre_scaled:  # zero point is int4
            if zero.min() < 0:  # sint4 zero point
                zero = zero.add_(8)  # convert to uint4 zero point
            assert zero.min() >= 0, "quantized zero point should be non-negative."
            assert zero.max() <= 15, "quantized zero point should be less than 16."
            weight = weight.add_(zero)
            zero = zero.mul_(subscale)
        else:
            if weight.min() < 0:  # sint4 weight
                weight = weight.add_(8)  # convert to uint4 weight
                zero = zero.add_(8 * subscale)
        _weight = weight.mul(subscale)
        assert _weight.min() >= 0, "first-level dequantize weight should be non-negative."
        assert _weight.max() <= 255, "first-level dequantize weight should be less than 256."
        del _weight
        assert subscale.min() >= 0, "subscale should be non-negative."
        assert subscale.max() <= 127, "subscale should be less than or equal to 127."
        assert zero.min() >= 0, "quantized zero point should be non-negative."
        assert zero.max() <= 255, "quantized zero point should be less than 256."
        assert weight.min() >= 0, "quantized weight should be non-negative."
        assert weight.max() <= 15, "quantized weight should be less than 16."
        # endregion
        zero = -zero  # ! for group quant, qserve uses q*s+z=r instead of q*s-z=r
        subscale = subscale.to(torch.int8)
        zero = zero.to(torch.int8)
    else:  # per-channel quantization
        assert subscale is None, "subscale tensor is not required for per-channel quantization."
        # region reshape scale and zero point
        if scale.numel() == 1:
            scale = scale.view(-1).expand(oc)
        scale = scale.reshape(oc).contiguous().view(oc, 1)
        if zero.numel() == 1:
            zero = zero.view(-1).expand(oc)
        zero = zero.reshape(oc).contiguous().view(oc, 1)
        # endregion
        # region quantize weight tensor
        if not zero_pre_scaled:  # zero point is fp16
            weight = weight.add_(zero)
        weight = weight.div_(scale).round_()
        if zero_pre_scaled:  # zero point is int4
            zero = zero.round_()
            if zero.min() < 0:  # sint4 zero point
                zero = zero.add_(8)  # convert to uint4 zero point
            assert zero.min() >= 0, "quantized zero point should be non-negative."
            assert zero.max() <= 15, "quantized zero point should be less than 16."
            weight = weight.add_(zero)
            zero = zero.mul_(scale)
        else:
            if weight.min() < 0:  # sint4 weight
                weight = weight.add_(8)  # convert to uint4 weight
                zero = zero.add_(8 * scale)
        assert weight.min() >= 0, "quantized weight should be non-negative."
        assert weight.max() <= 15, "quantized weight should be less than 16."
        # endregion
        zero = zero.to(dtype=dtype)
    scale = scale.to(dtype=dtype)
    packer = QServePacker()
    weight = packer.pack_weight(weight.view(oc, ic).to(torch.uint8))
    scale, zero, subscale = packer.pack_scale(scale=scale, zero=zero, subscale=subscale)
    return weight, scale, zero, subscale


def convert_to_qserve_w8x8y16_linear_weight(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a weight tensor to QServe W8-X8-Y16 linear weight format.

    Args:
        weight (`torch.Tensor`):
            weight tensor to be converted.
        scale (`torch.Tensor`):
            scale tensor for the weight tensor.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`:
            packed quantized weight tensor and scale tensor.
    """
    dtype = weight.dtype
    assert dtype == torch.float16, "currently qserve only supports fp16."
    assert scale is not None, "scale tensor is required for quantization."
    weight = weight.to(dtype=torch.float32)
    scale = scale.to(dtype=torch.float32, device=weight.device)
    oc = weight.shape[0]
    if scale.numel() == 1:
        scale = scale.view(-1).expand(oc)
    scale = scale.reshape(oc).contiguous().view(oc, 1)
    weight = weight.div_(scale).round_()
    assert weight.min() >= -128, "quantized weight should be greater than or equal to -128."
    assert weight.max() <= 127, "quantized weight should be less than or equal to 127."
    weight = weight.contiguous().to(torch.int8)
    scale = scale.view(oc).to(dtype=dtype)
    return weight, scale
