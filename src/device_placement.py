"""Hybrid GPU+CPU model placement.

Splits a model's submodules between CUDA and CPU so that models too large
for VRAM can still train: layers are placed on the GPU in forward order
until a memory budget is reached, and the remainder stays on the CPU.
Forward pre-hooks move activations across the device boundary, so the
model's own forward() code needs no changes.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def module_nbytes(module: nn.Module) -> int:
    """Total bytes of a module's parameters and buffers."""
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def placement_units(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Yield the (name, module) pairs that act as placement units.

    Top-level children are used directly, except containers without a
    forward (ModuleList/ModuleDict), which are expanded into their own
    children — e.g. ConvLSTM.cell_list becomes one unit per cell.
    """
    units = []
    for name, child in model.named_children():
        if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
            for sub_name, sub_child in child.named_children():
                units.append((f"{name}.{sub_name}", sub_child))
        else:
            units.append((name, child))
    return units


def _move_to_device(obj, device: torch.device):
    """Recursively move tensors in nested tuples/lists/dicts to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [_move_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def apply_hybrid_placement(
    model: nn.Module,
    vram_budget: Optional[int] = None,
    vram_fraction: float = 0.5,
) -> torch.device:
    """
    Place model units on CUDA until the VRAM budget is exhausted, the rest
    on CPU, and register hooks that move activations to each unit's device.

    Args:
        model: Model to place (modified in place)
        vram_budget: Parameter-memory budget in bytes for the GPU. Defaults
            to vram_fraction of currently free VRAM (the rest is headroom
            for activations, gradients, and optimizer state).
        vram_fraction: Fraction of free VRAM used when vram_budget is None

    Returns:
        The device of the first unit (where inputs should be sent)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid placement requires CUDA")

    if vram_budget is None:
        free_bytes, _ = torch.cuda.mem_get_info()
        vram_budget = int(free_bytes * vram_fraction)

    cuda = torch.device("cuda")
    cpu = torch.device("cpu")

    units = placement_units(model)
    if not units:
        model.to(cuda)
        return cuda

    used = 0
    devices = []
    for name, unit in units:
        size = module_nbytes(unit)
        # Adam keeps two fp32 moments per parameter, and gradients add one
        # more copy — budget roughly 4x the raw parameter bytes per unit.
        cost = size * 4
        if used + cost <= vram_budget:
            device = cuda
            used += cost
        else:
            device = cpu
        unit.to(device)
        devices.append((name, device))

    n_gpu = sum(1 for _, d in devices if d.type == "cuda")
    print(f"🔀 Hybrid placement: {n_gpu}/{len(devices)} units on GPU")
    for name, device in devices:
        print(f"   • {name}: {device.type}")

    for (_, unit), (_, device) in zip(units, devices):
        unit.register_forward_pre_hook(
            lambda module, args, _d=device: _move_to_device(args, _d)
        )

    return devices[0][1]
