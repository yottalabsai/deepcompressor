# -*- coding: utf-8 -*-
"""Backend utilities."""

import typing as tp

import safetensors
import torch

__all__ = ["ceil_divide", "pad", "MmaWeightPackerBase"]


def ceil_divide(x: int, divisor: int) -> int:
    """Ceiling division.

    Args:
        x (`int`):
            dividend.
        divisor (`int`):
            divisor.

    Returns:
        `int`:
            ceiling division result.
    """
    return (x + divisor - 1) // divisor


def pad(
    tensor: tp.Optional[torch.Tensor],
    divisor: int | tp.Sequence[int],
    dim: int | tp.Sequence[int],
    fill_value: float | int = 0,
) -> torch.Tensor:
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result


def load_state_dict_in_safetensors(
    path: str, device: str | torch.device = "cpu", filter_prefix: str = ""
) -> dict[str, torch.Tensor]:
    """Load state dict in SafeTensors.

    Args:
        path (`str`):
            file path.
        device (`str` | `torch.device`, optional, defaults to `"cpu"`):
            device.
        filter_prefix (`str`, optional, defaults to `""`):
            filter prefix.

    Returns:
        `dict`:
            loaded SafeTensors.
    """
    state_dict = {}
    with safetensors.safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            if filter_prefix and not k.startswith(filter_prefix):
                continue
            state_dict[k.removeprefix(filter_prefix)] = f.get_tensor(k)
    return state_dict


class MmaWeightPackerBase:
    def __init__(self, bits: int, warp_n: int, insn_k: int = None):
        self.bits = bits

        # compute tile size at the instruction level
        self.insn_k = insn_k if insn_k is not None else 256 // self.bits
        self.insn_n = 8
        # memory tile size at the lane level
        self.lane_k = 32 // self.bits
        self.lane_n = 1
        self.num_lanes = 32  # there are 32 lanes (or threds) in a warp
        self.num_k_lanes = 4
        self.num_n_lanes = 8
        self.pack_size = 4  # every 4 32-bit data is packed into one 128-bit data
        self.k_pack_size = self.insn_k // (self.num_k_lanes * self.lane_k)
        self.n_pack_size = self.pack_size // self.k_pack_size
        assert self.k_pack_size * self.n_pack_size == self.pack_size
        assert self.k_pack_size * self.num_k_lanes * self.lane_k == self.insn_k
        # memory tile size at the warp level
        self.warp_k = self.insn_k
        self.warp_n = warp_n
        self.num_k_frags = self.warp_k // (self.k_pack_size * self.num_k_lanes * self.lane_k)
        self.num_n_frags = self.warp_n // (self.n_pack_size * self.num_n_lanes * self.lane_n)
        assert self.num_k_frags == 1

    def get_view_shape(self, n: int, k: int) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        assert n % self.warp_n == 0, "output channel size should be divisible by warp_n."
        assert k % self.warp_k == 0, "input channel size should be divisible by warp_k."
        return (
            n // self.warp_n,
            self.num_n_frags,
            self.n_pack_size,
            self.num_n_lanes,
            self.lane_n,
            k // self.warp_k,
            self.num_k_frags,
            self.k_pack_size,
            self.num_k_lanes,
            self.lane_k,
        )
